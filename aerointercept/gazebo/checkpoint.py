"""Self-describing Gazebo PPO checkpoint helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


def _command_version(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5.0
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def architecture_config(model_config: dict) -> dict:
    return {key: value for key, value in model_config.items() if key != "encoder_chunk_size"}


def load_model_weights(model, checkpoint: dict, model_config: dict) -> None:
    stored = checkpoint.get("model_config")
    if stored is not None and architecture_config(stored) != architecture_config(model_config):
        raise ValueError("checkpoint visual architecture differs from current configuration")
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict model restore failed: {result}")


def validate_task_checkpoint(checkpoint: dict, cfg) -> None:
    if cfg.gazebo.task.get("task_version") != "noncontact_rendezvous_v1":
        return
    task = checkpoint.get("task_config")
    if task is None:
        task = checkpoint.get("config", {}).get("gazebo", {}).get("task")
    if task != dict(cfg.gazebo.task):
        raise ValueError("checkpoint task differs from the current noncontact center-distance task; train with new data")


def load_visual_initialization(model, checkpoint: dict, model_config: dict) -> None:
    """Explicit migration: preserve learned visual weights, add own-state fusion."""
    new_keys = {"self_state_dim", "self_velocity_scale", "self_angular_velocity_scale"}
    stored = checkpoint.get("model_config", {})
    visual = lambda cfg: {k: v for k, v in architecture_config(cfg).items() if k not in new_keys}
    if stored.get("self_state_dim", 0) != 0 or model_config.get("self_state_dim") != 6:
        raise ValueError("visual migration requires a legacy visual Actor and a six-state destination")
    if visual(stored) != visual(model_config):
        raise ValueError("visual architecture differs; cannot initialize its weights")
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.unexpected_keys or any(not key.startswith(("actor.self_state_encoder.", "actor.sensor_fusion."))
                                     for key in result.missing_keys):
        raise ValueError(f"unexpected visual migration mismatch: {result}")


def load_spatial_initialization(model, checkpoint: dict, model_config: dict) -> None:
    """Explicit architecture migration with initially zero spatial residual."""
    stored = checkpoint.get("model_config", {})
    visual = lambda cfg: {k: v for k, v in architecture_config(cfg).items() if k != "spatial_coordinates"}
    if stored.get("spatial_coordinates", False) or not model_config.get("spatial_coordinates", False):
        raise ValueError("spatial migration requires disabled-to-enabled position features")
    if visual(stored) != visual(model_config):
        raise ValueError("non-spatial architecture differs")
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.unexpected_keys or result.missing_keys != ["actor.spatial_projection.weight"]:
        raise ValueError(f"unexpected spatial migration mismatch: {result}")


def payload(model, optimizer, cfg, global_step, seed, best_hit_rate, metrics,
            lineage=None):
    result = {
        "phase": 3,
        "checkpoint_schema": 6 if cfg.end_to_end.model.get("self_state_dim") else 5,
        "backend": "gazebo_harmonic_px4_sitl",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "config": {
            "model": dict(cfg.end_to_end.model),
            "ppo": dict(cfg.end_to_end.ppo),
            "auxiliary": dict(cfg.end_to_end.auxiliary),
            "gazebo": dict(cfg.gazebo),
        },
        "model_config": dict(cfg.end_to_end.model),
        "render_config": dict(cfg.end_to_end.render),
        "action_config": dict(cfg.gazebo.action),
        "label_config": dict(cfg.end_to_end.labels),
        "task_config": dict(cfg.gazebo.task),
        "self_state_source": "px4_vehicle_odometry_body_frd_v1" if cfg.end_to_end.model.get("self_state_dim") else None,
        "safety_config": dict(cfg.end_to_end.safety),
        "random_seed": int(seed),
        "image_size": [640, 640],
        "camera_transform": "full_frame_letterbox_v1",
        "gazebo_version": _command_version(["gz", "sim", "--versions"]),
        "ros_distribution": "humble",
        "px4_interface": "PX4 SITL / px4_msgs",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "best_hit_rate": float(best_hit_rate),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "metrics": dict(metrics),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        },
    }
    if lineage is not None:
        result["lineage"] = dict(lineage)
    return result


def save(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload(**kwargs), temporary)
    temporary.replace(path)
