"""Offline per-axis visual policy error on recorded validation episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aerointercept.config import DotDict
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_model_weights, validate_task_checkpoint
from aerointercept.gazebo.config import load_gazebo_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_gazebo_config()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    validate_task_checkpoint(ckpt, cfg)
    model_cfg = DotDict(dict(ckpt["model_config"]))
    model_cfg["pretrained_weights"] = None
    model = EndToEndActorCritic(model_cfg).to(args.device).eval()
    load_model_weights(model, ckpt, dict(ckpt["model_config"]))
    files = ckpt["dataset_split"]["validation"]
    errors, targets, distances = [], [], []
    image_errors = []
    for filename in files:
        with np.load(Path(args.data)/"episodes"/filename, allow_pickle=False) as shard:
            frames = shard["frames"]
            actions = shard["actions"]
            own_states = shard["self_state"] if model.actor.self_state_dim else None
            image_xy = shard["camera_target_xy"] if "camera_target_xy" in shard else None
            visible = shard["confidence"]
            distance = shard["critic_obs"][:, 13]*float(cfg.end_to_end.labels.position_norm)
        indices = np.unique(np.linspace(0, len(frames)-1, min(64, len(frames)), dtype=int))
        for start in range(0, len(indices), 8):
            batch_indices = indices[start:start+8]
            histories = np.stack((np.maximum(0, batch_indices-1), batch_indices), axis=1)
            with torch.no_grad():
                outputs = model.actor.act(torch.from_numpy(frames[histories]).to(args.device),
                                              deterministic=True,
                                              self_state=None if own_states is None else torch.from_numpy(
                                                  own_states[batch_indices]).to(args.device))
                predictions = outputs[0].cpu().numpy()
                if image_xy is not None:
                    attention = outputs[-1].cpu().numpy()
                    h, w = attention.shape[-2:]
                    xx = (np.arange(w)+.5)*2/w-1
                    yy = (np.arange(h)+.5)*2/h-1
                    xy = np.stack(((attention*xx[None, None, :]).sum((1, 2)),
                                   (attention*yy[None, :, None]).sum((1, 2))), axis=-1)
                    truth_xy = image_xy[batch_indices]
                    center_visible = ((visible[batch_indices] > .5)
                                      & (np.abs(truth_xy[:, 0]) <= 1.)
                                      & (np.abs(truth_xy[:, 1]) <= 9./16.))
                    image_errors.append((xy-truth_xy)[center_visible])
            errors.append(predictions-actions[batch_indices])
            targets.append(actions[batch_indices])
            distances.append(distance[batch_indices])
    errors, targets, distances = map(np.concatenate, (errors, targets, distances))
    scale = np.array([cfg.gazebo.action.velocity_max]*3 + [cfg.gazebo.action.yaw_rate_max])
    report = {"checkpoint": args.checkpoint, "validation_files": files,
              "units": ["m/s", "m/s", "m/s", "rad/s"],
              "axes": ["forward", "right", "down", "yaw_rate"], "groups": {}}
    if image_errors:
        error = np.concatenate(image_errors)
        report["visible_localization"] = {"samples": len(error), "rmse_pixels_xy": (
            np.sqrt(np.mean(error**2, axis=0))*frames.shape[-1]/2).tolist() if len(error) else None}
    for name, mask in (("all", np.ones(len(errors), bool)), ("near_2m", distances < 2),
                       ("far_2m", distances >= 2)):
        report["groups"][name] = {"samples": int(mask.sum())}
        if mask.any():
            report["groups"][name].update({
                "rmse": (np.sqrt(np.mean(errors[mask]**2, axis=0))*scale).tolist(),
                "bias": (np.mean(errors[mask], axis=0)*scale).tolist(),
                "teacher_std": (np.std(targets[mask], axis=0)*scale).tolist(),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
