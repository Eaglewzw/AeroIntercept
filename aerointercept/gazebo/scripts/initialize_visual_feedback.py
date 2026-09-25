"""Explicitly initialize an RGB yaw-feedback Actor; this is not a trained result."""
import argparse
import hashlib
import io
from pathlib import Path

import torch

from aerointercept.config import DotDict
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import architecture_config, validate_task_checkpoint
from aerointercept.gazebo.config import load_gazebo_config


def initialize(source, cfg, source_path, source_sha256):
    validate_task_checkpoint(source, cfg)
    target = dict(cfg.end_to_end.model)
    changed = {"visual_yaw_gain", "camera_horizontal_fov"}
    comparable = lambda d: {k: v for k, v in architecture_config(d).items() if k not in changed}
    if comparable(source["model_config"]) != comparable(target):
        raise ValueError("feedback initialization must preserve every learned architecture component")
    if source["model_config"].get("visual_yaw_gain", 0.) or not target.get("visual_yaw_gain", 0.):
        raise ValueError("requires a zero-feedback source and a positive-feedback destination")
    if target["camera_horizontal_fov"] != cfg.gazebo.camera.horizontal_fov:
        raise ValueError("feedback camera calibration differs from the active camera")
    construction = DotDict({**target, "pretrained_weights": None})
    model = EndToEndActorCritic(construction).eval()
    model.load_state_dict(source["model"], strict=True)
    copied = ("phase", "backend", "render_config", "action_config", "label_config",
              "task_config", "safety_config", "self_state_source")
    result = {k: source[k] for k in copied if k in source}
    result.update(checkpoint_schema=7, model=model.state_dict(), model_config=target,
                  training_stage="visual_feedback_initialization",
                  lineage={"source_checkpoint": source_path, "source_sha256": source_sha256,
                           "change": "RGB attention calibrated yaw feedback; learned weights unchanged"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint).resolve()
    contents = path.read_bytes()
    source = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
    result = initialize(source, load_gazebo_config(args.config), str(path), hashlib.sha256(contents).hexdigest())
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("use a new checkpoint path for this explicit architecture change")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"initialized {output}; requires new closed-loop evaluation", flush=True)


if __name__ == "__main__":
    main()
