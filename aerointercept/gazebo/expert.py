"""Privileged analytical expert used only to label Gazebo camera frames."""

from __future__ import annotations

import numpy as np

from aerointercept.end_to_end.actions import encode_velocity_command
from aerointercept.gazebo.task_logic import camera_target_yaw_geometry


class GazeboExpertController:
    """Privileged analytical expert used only to label Gazebo camera frames.

    Single-phase full-speed pursuit: every command aims at the velocity- and
    acceleration-led prediction of the target position and saturates the
    velocity protocol at ``velocity_max``, so the labels teach charging
    through the target at the airframe's physical top speed.  The retired
    cooperative-formation docking law (fixed offset, relative-velocity
    damping, ``maximum_relative_speed`` cap) had the interceptor slow down to
    match the leader and is gone entirely.

    The controller consumes simulator truth and therefore must never be used
    as an Actor input or deployment fallback.  Its output follows the exact
    same normalized body-velocity/yaw-rate action protocol as the learned
    Actor.
    """

    def __init__(self, expert_cfg, action_cfg, camera_cfg):
        self.cfg = expert_cfg
        self.action_cfg = action_cfg
        self.camera_cfg = camera_cfg
        if not 0.0 < float(self.cfg.get("lead_seconds", 0.0)):
            raise ValueError("expert configuration requires a positive lead_seconds")
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._previous_target_velocity = None
        self._previous_timestamp = None
        self._target_acceleration = np.zeros(3)

    def reset(self) -> None:
        self._previous_action.fill(0.0)
        self._previous_target_velocity = None
        self._previous_timestamp = None
        self._target_acceleration.fill(0.0)

    def action(self, state: dict) -> np.ndarray:
        interceptor_position = np.asarray(
            state["interceptor_position"], dtype=np.float64,
        )
        target_position = np.asarray(
            state["target_position"], dtype=np.float64,
        )
        target_velocity = np.asarray(state["target_velocity"], dtype=np.float64)
        yaw = float(state["interceptor_yaw"])
        values = np.concatenate((
            interceptor_position, target_position, target_velocity,
            np.array([yaw]),
        ))
        if not np.isfinite(values).all():
            raise ValueError("Gazebo expert state contains NaN or Inf")

        if self.cfg.get("controller") != "full_speed_pursuit_v1":
            raise ValueError("the Gazebo expert requires the full-speed pursuit configuration")

        # EMA estimate of the target acceleration, folded into the lead-point
        # prediction so the chase aims where a maneuvering target will be.
        timestamp = state.get("timestamp_seconds")
        if timestamp is not None and self._previous_timestamp is not None:
            dt = float(timestamp) - self._previous_timestamp
            if .01 <= dt <= 1.:
                acceleration = np.clip(
                    (target_velocity - self._previous_target_velocity) / dt, -2., 2.)
                self._target_acceleration = (
                    .7 * self._target_acceleration + .3 * acceleration)
        self._previous_target_velocity = target_velocity.copy()
        self._previous_timestamp = None if timestamp is None else float(timestamp)

        # Full-speed pursuit of the lead point: no docking offset and no
        # relative-speed cap — only the airframe's velocity_max applies.
        lead_seconds = float(self.cfg.lead_seconds)
        lead_point = (
            target_position
            + target_velocity * lead_seconds
            + 0.5 * self._target_acceleration * lead_seconds ** 2
        )
        chase = lead_point - interceptor_position
        chase_norm = float(np.linalg.norm(chase))
        if chase_norm > 1e-6:
            velocity_ned = (
                float(self.action_cfg.velocity_max) * chase / chase_norm
            )
        else:
            velocity_ned = np.zeros(3)

        _, _, yaw_error = camera_target_yaw_geometry(
            interceptor_position, target_position, yaw,
            float(self.camera_cfg.mount_yaw_offset_rad),
        )
        yaw_rate = float(np.clip(
            float(self.cfg.yaw_gain) * yaw_error,
            -float(self.action_cfg.yaw_rate_max),
            float(self.action_cfg.yaw_rate_max),
        ))
        action = encode_velocity_command(
            velocity_ned, yaw_rate, yaw,
            velocity_max=float(self.action_cfg.velocity_max),
            yaw_rate_max=float(self.action_cfg.yaw_rate_max),
        )
        smoothing = float(self.cfg.action_smoothing)
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("expert action_smoothing must be in [0,1)")
        action = smoothing * self._previous_action + (1.0 - smoothing) * action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise RuntimeError("Gazebo expert produced an invalid action")
        self._previous_action = action.copy()
        return action


def expert_name() -> str:
    return "gazebo_full_speed_pursuit_v1"
