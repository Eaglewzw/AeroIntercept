"""Collect Gazebo camera frames labeled by the privileged lead-pursuit expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aerointercept.end_to_end.data import DATASET_SCHEMA_VERSION
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboInterceptEnv
from aerointercept.gazebo.dataset_integrity import recover_summaries, save_episode
from aerointercept.gazebo.expert import GazeboExpertController, expert_name
from aerointercept.gazebo.process import maybe_launch
from aerointercept.gazebo.scenarios import MODES
from aerointercept.gazebo.self_state import SELF_STATE_SOURCE, SELF_STATE_COMPONENTS
from aerointercept.gazebo.visual_supervision import CAMERA_LABEL_PROTOCOL, camera_target_xy
from aerointercept.training.collect_e2e_data import prepare_output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--mode", choices=(*MODES, "mixed"),
        default="mixed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/gazebo_experiment_c")
    parser.add_argument("--socket", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted dataset from its existing episode shards",
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--behavior-checkpoint", default=None,
                        help="optional visual policy for corrective imitation data collection")
    parser.add_argument("--expert-weight", type=float, default=.8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def collect_episode(env, expert, behavior=None, expert_weight=.8):
    frames, training, reset_info = env.reset()
    expert.reset()
    collected = {
        "frames": [], "actions": [], "future_position": [],
        "collision_risk": [], "confidence": [], "critic_obs": [],
        "camera_target_xy": [],
    }
    use_self_state = int(env.cfg.end_to_end.model.get("self_state_dim", 0)) > 0
    if use_self_state:
        collected["self_state"] = []
        collected["self_state_timestamp_ns"] = []
        collected["image_timestamp_ns"] = []
    final = None
    while final is None:
        state = env.expert_state()
        own_state = env.actor_self_state()
        if use_self_state:
            collected["self_state"].append(own_state.copy())
            collected["self_state_timestamp_ns"].append(env._last_snapshot["self_state_timestamp_ns"])
            collected["image_timestamp_ns"].append(env._last_snapshot["image_timestamp_ns"])
        action = expert.action(state)
        # Privileged supervision only: normalized coordinates in the complete
        # letterboxed image. Never returned through the Actor input interface.
        collected["camera_target_xy"].append(camera_target_xy(
            env._last_snapshot["camera_relative_frd"], float(env.cfg.gazebo.camera.horizontal_fov)))
        executed_action = action
        if behavior is not None:
            # Only the collector mixes a privileged teacher and a visual
            # behavior policy. Labels always contain the teacher correction.
            # Fade to the teacher inside 2 m to keep collection noncontact.
            distance = float(np.linalg.norm(state["target_position"]-state["interceptor_position"]))
            weight = 1. - (1.-expert_weight)*float(np.clip((distance-2.)/3., 0., 1.))
            executed_action = weight*action + (1.-weight)*behavior(frames, own_state)
        collected["frames"].append(frames[-1].copy())
        collected["actions"].append(action.copy())
        for key in (
            "future_position", "collision_risk", "confidence", "critic_obs",
        ):
            collected[key].append(np.asarray(training[key]).copy())
        frames, _, terminated, truncated, training, info = env.step(executed_action)
        if terminated or truncated:
            final = info["final"]

    arrays = {
        "frames": np.asarray(collected["frames"], dtype=np.uint8),
        "actions": np.asarray(collected["actions"], dtype=np.float32),
        "future_position": np.asarray(
            collected["future_position"], dtype=np.float32,
        ),
        "collision_risk": np.asarray(
            collected["collision_risk"], dtype=np.float32,
        ),
        "confidence": np.asarray(collected["confidence"], dtype=np.float32),
        "critic_obs": np.asarray(collected["critic_obs"], dtype=np.float32),
        "camera_target_xy": np.asarray(collected["camera_target_xy"], dtype=np.float32),
    }
    summary = {
        **final,
        "length": int(arrays["frames"].shape[0]),
        "outcome": str(final["outcome"]),
        "minimum_distance": float(final["minimum_distance"]),
        "episode_reward": float(final["episode_reward"]),
        "reset_target_distance_m": float(reset_info["target_distance_m"]),
        "scenario": reset_info.get("scenario"),
        "physical_state_source": reset_info.get("physical_state_source"),
    }
    if use_self_state:
        arrays["self_state"] = np.asarray(collected["self_state"], dtype=np.float32)
        for key in ("self_state_timestamp_ns", "image_timestamp_ns"):
            arrays[key] = np.asarray(collected[key], dtype=np.int64)
        summary["self_state_source"] = SELF_STATE_SOURCE
    return arrays, summary


def mode_plan(requested_mode: str, episode_count: int) -> list[tuple[str, int]]:
    """Split a mixed dataset across modes supported by the read-only C++ node."""
    if requested_mode != "mixed":
        return [(requested_mode, int(episode_count))]
    modes = ("circle", "sinusoidal", "random_walk")
    quotient, remainder = divmod(int(episode_count), len(modes))
    return [
        (mode, quotient + int(index < remainder))
        for index, mode in enumerate(modes)
        if quotient + int(index < remainder) > 0
    ]


def remaining_mode_plan(
    plan: list[tuple[str, int]], completed_episodes: int,
) -> list[tuple[str, int]]:
    remaining = []
    consumed = int(completed_episodes)
    for mode, count in plan:
        skipped = min(consumed, count)
        consumed -= skipped
        if count > skipped:
            remaining.append((mode, count - skipped))
    if consumed:
        raise ValueError("existing dataset has more episodes than requested")
    return remaining


def _infer_existing_summaries(episodes_dir: Path, plan) -> list[dict]:
    """Compatibility entry point; recovery now uses recorded results only."""
    return recover_summaries(episodes_dir, plan)


def _write_progress(path: Path, summaries: list[dict], total_frames: int) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "episodes": len(summaries),
        "frames": int(total_frames),
        "summaries": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    cfg = load_gazebo_config(args.config)
    episode_count = int(cfg.end_to_end.bc.episodes if args.episodes is None else args.episodes)
    if episode_count < 1:
        raise ValueError("--episodes must be positive")
    if not 0. <= args.expert_weight <= 1.:
        raise ValueError("--expert-weight must be in [0,1]")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    socket_path = args.socket or str(cfg.gazebo.bridge.socket)
    output_dir = Path(args.out)
    plan = mode_plan(args.mode, episode_count)
    contract = {
        "seed": args.seed, "mode_plan": plan,
        "task": dict(cfg.gazebo.task), "expert": dict(cfg.gazebo.expert),
        "action": dict(cfg.gazebo.action), "camera": dict(cfg.gazebo.camera),
        "labels": dict(cfg.end_to_end.labels),
        "camera_target_label": CAMERA_LABEL_PROTOCOL,
    }
    if int(cfg.end_to_end.model.get("self_state_dim", 0)):
        contract["self_state"] = {"source": SELF_STATE_SOURCE, "components": SELF_STATE_COMPONENTS,
                                  "frame": "body_frd", "units": ["m/s"]*3+["rad/s"]*3}
    behavior = None
    if args.behavior_checkpoint:
        import torch
        from aerointercept.config import DotDict
        from aerointercept.end_to_end.policy import EndToEndActorCritic
        from aerointercept.gazebo.checkpoint import load_model_weights, validate_task_checkpoint
        path = Path(args.behavior_checkpoint)
        checkpoint_bytes = path.read_bytes()
        contract["behavior"] = {
            "checkpoint": str(path.resolve()), "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "expert_weight": args.expert_weight, "protocol": "teacher_blend_fade_2m_5m_v1",
        }
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        validate_task_checkpoint(checkpoint, cfg)
        model_config = DotDict(dict(checkpoint["model_config"]))
        model_config["pretrained_weights"] = None
        model = EndToEndActorCritic(model_config).to(args.device).eval()
        load_model_weights(model, checkpoint, dict(checkpoint["model_config"]))
        def behavior(frames, own_state):
            own_tensor = (torch.from_numpy(own_state[None]).to(args.device)
                          if model.actor.self_state_dim else None)
            with torch.no_grad():
                return model.actor.act(torch.from_numpy(frames[None]).to(args.device),
                                       deterministic=True, self_state=own_tensor)[0][0].cpu().numpy()
    # JSON round-trip gives tuples the same representation before/after restart.
    contract = json.loads(json.dumps(contract))
    contract_path = output_dir / "collection_config.json"
    if args.resume:
        episodes_dir = output_dir / "episodes"
        if not episodes_dir.is_dir():
            raise FileNotFoundError(f"cannot resume missing dataset: {output_dir}")
        if not contract_path.exists() or json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "cannot resume without an identical recorded collection_config.json; "
                "use a new output directory for changed or legacy task definitions"
            )
        summaries = _infer_existing_summaries(episodes_dir, plan)
        total_frames = sum(item["length"] for item in summaries)
        print(
            f"[AeroIntercept] resuming {len(summaries)}/{episode_count} "
            f"episodes ({total_frames:,} frames)", flush=True,
        )
    else:
        episodes_dir = prepare_output(output_dir, args.overwrite)
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        summaries = []
        total_frames = 0
    progress_path = output_dir / "collection_progress.json"
    episode_index = len(summaries)
    remaining_plan = remaining_mode_plan(plan, episode_index)
    stack = maybe_launch(args, socket_path) if remaining_plan else None
    env = None
    try:
        if remaining_plan:
            env = GazeboInterceptEnv(cfg, socket_path, seed=args.seed)
            if not env._scenario_control:
                raise RuntimeError("collection requires the seeded cooperative target supervisor")
            env.episode_index = episode_index
        expert = GazeboExpertController(cfg.gazebo.expert, cfg.gazebo.action, cfg.gazebo.camera)
        for selected_mode, mode_episodes in remaining_plan:
            env.mode = selected_mode
            for _ in range(mode_episodes):
                arrays, summary = collect_episode(env, expert, behavior, args.expert_weight)
                summary["mode"] = selected_mode
                final_path = episodes_dir / f"episode_{episode_index:06d}.npz"
                save_episode(final_path, arrays, summary)
                summaries.append(summary)
                total_frames += summary["length"]
                episode_index += 1
                _write_progress(progress_path, summaries, total_frames)
                hits = sum(item["outcome"] == "hit" for item in summaries)
                print(
                    f"[{episode_index}/{episode_count}] frames={total_frames:,} "
                    f"rendezvous_success={hits / episode_index:.1%} "
                    f"mode={selected_mode} last={summary['outcome']} "
                    f"min={summary['minimum_distance']:.3f}", flush=True,
                )
                if summary["outcome"] in ("contact", "ground", "invalid") and episode_index < episode_count:
                    if stack is None:
                        raise RuntimeError("a physical failure requires --launch for simulator recovery")
                    env.close()
                    stack.close()
                    stack = maybe_launch(args, socket_path)
                    env = GazeboInterceptEnv(cfg, socket_path, seed=args.seed, mode=selected_mode)
                    env.episode_index = episode_index
    finally:
        if env is not None:
            env.close()
        if stack is not None:
            stack.close()

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "phase": 3,
        "experiment": "C",
        "backend": "gazebo_harmonic_px4_sitl",
        "teacher": expert_name(),
        "observation": "gazebo_full_rgb_frame_history",
        "camera_source": str(cfg.gazebo.camera.source),
        "camera_transform": str(cfg.gazebo.camera.transform),
        "channel_order": "RGB",
        "action_protocol": "body_velocity_yaw_rate_v1",
        "episodes": episode_count,
        "frames": total_frames,
        "image_width": int(cfg.end_to_end.render.image_width),
        "image_height": int(cfg.end_to_end.render.image_height),
        "history_frames": int(cfg.end_to_end.model.history_frames),
        "velocity_max": float(cfg.gazebo.action.velocity_max),
        "yaw_rate_max": float(cfg.gazebo.action.yaw_rate_max),
        "seed": args.seed,
        "mode": args.mode,
        "mode_plan": dict(plan),
        "collection_config": contract,
        "scenario_control": "bounded_cooperative_v1",
        "summaries": summaries,
        "self_state_source": SELF_STATE_SOURCE if int(cfg.end_to_end.model.get("self_state_dim", 0)) else None,
    }
    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(
        f"saved Gazebo Experiment C dataset to {output_dir} "
        f"({total_frames:,} frames)", flush=True,
    )


if __name__ == "__main__":
    main()
