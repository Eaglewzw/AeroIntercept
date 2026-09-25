"""Configuration loading for the Gazebo overlay."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml

from aerointercept.config import DEFAULT_CONFIG, DotDict


DEFAULT_GAZEBO_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "gazebo_e2e.yaml"
)


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_gazebo_config(path: str | Path | None = None) -> DotDict:
    """Deep-merge the Gazebo contract over the unchanged legacy defaults."""
    with Path(DEFAULT_CONFIG).open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    overlay_path = Path(path) if path is not None else DEFAULT_GAZEBO_CONFIG
    with overlay_path.open("r", encoding="utf-8") as stream:
        overlay = yaml.safe_load(stream)
    result = DotDict(_merge(base, overlay))
    render = result.end_to_end.render
    if (int(render.image_width), int(render.image_height)) != (640, 640):
        raise ValueError("Gazebo actor input must remain 640x640")
    if render.channel_order != "RGB" or int(render.history_frames) != 2:
        raise ValueError("Gazebo observation protocol must be two RGB frames")
    task = result.gazebo.task
    model = result.end_to_end.model
    if int(model.get("self_state_dim", 0)) not in (0, 6):
        raise ValueError("Actor self-state dimension must be 0 or 6")
    for key in ("self_velocity_scale", "self_angular_velocity_scale"):
        if key in model and (not math.isfinite(float(model[key])) or float(model[key]) <= 0):
            raise ValueError(f"model.{key} must be finite and positive")
    for key in ("hit_radius", "reset_target_distance_m", "reset_timeout_s"):
        if not math.isfinite(float(task[key])) or float(task[key]) <= 0:
            raise ValueError(f"gazebo.task.{key} must be finite and positive")
    if task.get("task_version") == "noncontact_rendezvous_v1":
        if task.get("center_reference") != "gazebo_base_link_center_enu_to_ned_v2":
            raise ValueError("noncontact task requires the corrected base_link center reference")
        if not task.get("require_contact_monitor"):
            raise ValueError("noncontact rendezvous requires contact monitoring")
        for key in ("rendezvous_max_relative_speed_mps", "rendezvous_hold_seconds"):
            if not math.isfinite(float(task[key])) or float(task[key]) <= 0:
                raise ValueError(f"gazebo.task.{key} must be finite and positive")
    return result
