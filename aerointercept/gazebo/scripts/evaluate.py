"""Evaluate a detector-free Actor in the same physical Gazebo/PX4 backend."""

from __future__ import annotations

import argparse
import copy
import json
import hashlib
import io
import math
from pathlib import Path
import time

import torch

from aerointercept.config import DotDict
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_model_weights, validate_task_checkpoint
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboInterceptEnv, requires_world_replacement
from aerointercept.gazebo.expert import GazeboExpertController
from aerointercept.gazebo.evaluation_metrics import summarize_episodes, acceptance_result
from aerointercept.gazebo.process import maybe_launch
from aerointercept.gazebo.protocol import BridgeProtocolError
from aerointercept.gazebo.scenarios import MODES, TRAIN_MODES, HELD_OUT_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--socket", default=None)
    parser.add_argument("--output", default="artifacts/runs/experiments/gazebo_evaluation/evaluation.json")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=(*MODES, "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--suite", action="store_true", help="evaluate each training and held-out shape, episodes per mode")
    parser.add_argument("--trace", action="store_true", help="record step-level sensor, action and diagnostic state")
    parser.add_argument("--initial-distance", type=float, default=None,
                        help="test-only initial center distance in meters; preserves checkpoint task validation")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="test-only Actor velocity command norm limit in m/s; does not change target motion")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    output = Path(args.output)
    if args.trace and output.with_suffix(".trace.jsonl").exists():
        raise FileExistsError("trace output already exists; use a distinct run output path")
    cfg = load_gazebo_config(args.config)
    checkpoint_cfg = copy.deepcopy(cfg)
    test_overrides = {}
    for flag, value in (("initial_distance_m", args.initial_distance), ("max_speed_mps", args.max_speed)):
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{flag} must be finite and positive")
            test_overrides[flag] = value
    if args.initial_distance is not None:
        cfg["gazebo"]["task"]["reset_target_distance_m"] = args.initial_distance
    if args.max_speed is not None:
        cfg["gazebo"]["action"]["velocity_max"] = args.max_speed
    # Use the resolved task distance for both world spawn and physical reset.
    args.initial_distance = float(cfg.gazebo.task.reset_target_distance_m)
    socket_path = args.socket or cfg.gazebo.bridge.socket
    stack = maybe_launch(args, socket_path)
    environment = None
    try:
        environment = GazeboInterceptEnv(cfg, socket_path, mode=args.mode, seed=args.seed)
        device = torch.device(args.device)
        checkpoint_bytes = Path(args.checkpoint).read_bytes()
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False)
        validate_task_checkpoint(checkpoint, checkpoint_cfg)
        model_config = dict(cfg.end_to_end.model)
        construction_config = DotDict(dict(model_config))
        construction_config["pretrained_weights"] = None
        model = EndToEndActorCritic(construction_config).to(device).eval()
        load_model_weights(model, checkpoint, dict(cfg.end_to_end.model))
        diagnostic_expert = (GazeboExpertController(cfg.gazebo.expert, cfg.gazebo.action, cfg.gazebo.camera)
                             if args.trace else None)
        records = []
        reset_recoveries = []
        retired_camera_frames = 0
        started = time.perf_counter()
        modes = (*TRAIN_MODES, *HELD_OUT_MODES) if args.suite else (args.mode,)
        schedule = [mode for mode in modes for _ in range(args.episodes)]
        output.parent.mkdir(parents=True, exist_ok=True)
        for episode, mode in enumerate(schedule):
            environment.mode = mode
            try:
                frames, _, reset_info = environment.reset()
            except (BridgeProtocolError, TimeoutError, ConnectionError, EOFError) as exc:
                if stack is None:
                    raise
                reset_recoveries.append({"before_episode": episode, "error": str(exc)})
                print(f"reset recovery before episode={episode + 1}: {exc}", flush=True)
                old = environment
                retired_camera_frames += old.camera_frames
                old.close()
                stack.restart()
                environment = GazeboInterceptEnv(cfg, socket_path, mode=mode, seed=args.seed)
                environment.episode_index = old.episode_index
                frames, _, reset_info = environment.reset()
            episode_reward = 0.0
            print(f"episode={episode + 1}/{len(schedule)} reset "
                  f"requested_distance={cfg.gazebo.task.reset_target_distance_m:.2f} m "
                  f"measured_center_distance={reset_info['target_distance_m']:.3f} m "
                  f"velocity_command_limit={cfg.gazebo.action.velocity_max:.2f} m/s", flush=True)
            model.actor.reset_memory()
            if diagnostic_expert is not None:
                diagnostic_expert.reset()
            for step_index in range(int(cfg.gazebo.task.episode_max_steps)):
                teacher_action = None
                input_center_distance = None
                if diagnostic_expert is not None:
                    # Diagnostics only: this label is written to the trace and
                    # never used to modify the visual Actor's command.
                    diagnostic_state = environment.expert_state()
                    teacher_action = diagnostic_expert.action(diagnostic_state).tolist()
                    input_center_distance = math.dist(diagnostic_state["target_position"],
                                                      diagnostic_state["interceptor_position"])
                # Deployment path calls Actor directly: no Critic or simulator truth argument.
                own_state = environment.actor_self_state()
                input_image_timestamp_ns = environment._last_snapshot.get("image_timestamp_ns")
                input_camera_relative = environment._last_snapshot.get("camera_relative_frd")
                own_tensor = None if own_state is None else torch.from_numpy(own_state[None]).to(device)
                inference_started = time.perf_counter()
                with torch.no_grad():
                    policy_outputs = model.actor.act(
                        torch.from_numpy(frames[None]).to(device), deterministic=True, self_state=own_tensor
                    )
                    action = policy_outputs[0][0].cpu().numpy()
                inference_wall_seconds = time.perf_counter()-inference_started
                predicted_image_xy = None
                if args.trace:
                    weights = policy_outputs[-1][0].detach().cpu()
                    h, w = weights.shape
                    x = (torch.arange(w)+.5)*2/w-1
                    y = (torch.arange(h)+.5)*2/h-1
                    predicted_image_xy = [float((weights*x[None, :]).sum()), float((weights*y[:, None]).sum())]
                try:
                    frames, reward, terminated, truncated, _, info = environment.step(action)
                    episode_reward += reward
                except (BridgeProtocolError, TimeoutError, ConnectionError, EOFError) as exc:
                    # Count a lost physical episode as a failed evaluation.
                    # We have no terminal measurement: keep those fields null,
                    # rather than fabricate a post-crash image or a success.
                    terminated, truncated = True, False
                    info = {"final": {
                        "outcome": "simulator_error", "error": str(exc),
                        "episode_reward": None, "episode_length": environment._episode.length,
                        "minimum_distance": None, "contact_count": None,
                        "last_observed_minimum_distance": environment._episode.minimum_distance,
                        "target_contact": None,
                    }}
                if args.trace:
                    snapshot = environment._last_snapshot
                    with output.with_suffix(".trace.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "episode": episode, "step": step_index, "mode": mode,
                            "input_image_timestamp_ns": input_image_timestamp_ns,
                            "input_camera_relative_frd": input_camera_relative,
                            "predicted_image_xy": predicted_image_xy,
                            "diagnostic_teacher_action": teacher_action,
                            "input_center_distance_m": input_center_distance,
                            "inference_wall_seconds": inference_wall_seconds,
                            "result_image_timestamp_ns": snapshot.get("image_timestamp_ns"),
                            "action": action.tolist(), "self_state_input": None if own_state is None else own_state.tolist(),
                            "observed_distance": info.get("distance"), "visible": info.get("visible"),
                            "fov_center_error": info.get("fov_center_error"),
                            "last_available_interceptor_position": snapshot["interceptor_position"],
                            "last_available_target_position": snapshot["target_position"],
                            "camera_relative_frd": snapshot.get("camera_relative_frd"),
                            "decoded_action": info.get("decoded_action"),
                        }, allow_nan=False) + "\n")
                if terminated or truncated:
                    final = info["final"]
                    actual_mode = reset_info["scenario"]["mode"]
                    records.append({**final, "episode": episode, "reset": reset_info,
                                    "mode": actual_mode, "split": "held_out_shape" if actual_mode in HELD_OUT_MODES else "new_seed"})
                    progress = output.with_suffix(".progress.json")
                    temporary = progress.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"complete": False, "episode_records": records,
                                                     "checkpoint": str(Path(args.checkpoint).resolve()),
                                                     "checkpoint_sha256": checkpoint_sha256,
                                                     "task_config": dict(cfg.gazebo.task),
                                                     "action_config": dict(cfg.gazebo.action),
                                                     "test_overrides": test_overrides,
                                                     "requested_seed": args.seed,
                                                     "reset_recoveries": reset_recoveries}, indent=2, allow_nan=False))
                    temporary.replace(progress)
                    print(
                        f"episode={episode + 1}/{len(schedule)} mode={actual_mode} outcome={final['outcome']} "
                        f"reward_observed={episode_reward:.2f} min={final['minimum_distance']}",
                        flush=True,
                    )
                    # A physical contact can leave PX4/Gazebo in a crashed
                    # state, successful interceptions included.  When this
                    # script owns the launch, replace the world before asking it
                    # to reset; otherwise a later episode would silently measure
                    # a broken simulator.
                    if ((final["outcome"] == "simulator_error" or requires_world_replacement(final))
                            and stack is not None and episode + 1 < len(schedule)):
                        old = environment
                        retired_camera_frames += old.camera_frames
                        episode_index = old.episode_index
                        old.close()
                        stack.restart()
                        environment = GazeboInterceptEnv(
                            cfg, socket_path, mode=mode, seed=args.seed,
                        )
                        environment.episode_index = episode_index
                    elif final["outcome"] == "simulator_error" and episode + 1 < len(schedule):
                        raise RuntimeError("evaluation recovery requires an owned --launch stack")
                    break
            else:
                raise RuntimeError("environment failed to emit its configured timeout")
        elapsed = time.perf_counter() - started
        report = {
            "backend": "gazebo_harmonic_px4_sitl",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "actor_inputs": ["frames", "self_state"] if model.actor.self_state_dim else ["frames"],
            "model_config": dict(cfg.end_to_end.model),
            **summarize_episodes(records),
            "requested_mode": args.mode,
            "requested_seed": args.seed,
            "target_seed_controlled": environment._scenario_control,
            "complete": True,
            "reset_recoveries": reset_recoveries,
            "acceptance": acceptance_result(records, (*TRAIN_MODES, *HELD_OUT_MODES), reset_recoveries,
                                            hit_radius=float(cfg.gazebo.task["hit_radius"])),
            "scenario_results": {mode: summarize_episodes([r for r in records if r["mode"] == mode])
                                 for mode in sorted({r["mode"] for r in records})},
            "task_config": dict(cfg.gazebo.task),
            "checkpoint_task_config": dict(checkpoint_cfg.gazebo.task),
            "action_config": dict(cfg.gazebo.action),
            "test_overrides": test_overrides,
            "reward_config": dict(cfg.gazebo.rewards),
            "camera_fps": (retired_camera_frames + environment.camera_frames) / elapsed,
            "elapsed_seconds": elapsed,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded_report = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(encoded_report, encoding="utf-8")
        temporary.replace(output)
        progress = output.with_suffix(".progress.json")
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(encoded_report, encoding="utf-8")
        temporary.replace(progress)
        print("AEROINTERCEPT_GAZEBO_EVAL=" + json.dumps(report), flush=True)
    finally:
        if environment is not None:
            environment.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
