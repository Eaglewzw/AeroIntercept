"""Privileged analytical expert used only to label Gazebo camera frames."""

from __future__ import annotations

import numpy as np

from aerointercept.end_to_end.actions import encode_velocity_command
from aerointercept.gazebo.task_logic import camera_target_yaw_geometry


class GazeboExpertController:
    """Cooperative formation-following expert for noncontact behavior cloning.

    The controller consumes simulator truth and therefore must never be used as
    an Actor input or deployment fallback.  Its output follows the exact same
    normalized body-velocity/yaw-rate action protocol as the learned Actor.
    """

    def __init__(self, expert_cfg, action_cfg, camera_cfg):
        self.cfg = expert_cfg
        self.action_cfg = action_cfg
        self.camera_cfg = camera_cfg
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
        target_velocity = np.asarray(
            state["target_velocity"], dtype=np.float64,
        )
        yaw = float(state["interceptor_yaw"])
        values = np.concatenate((
            interceptor_position, target_position, target_velocity,
            np.array([yaw]),
        ))
        if not np.isfinite(values).all():
            raise ValueError("Gazebo expert state contains NaN or Inf")

        if self.cfg.get("controller") != "cooperative_rendezvous_v1":
            raise ValueError("the Gazebo expert requires the cooperative rendezvous configuration")
        desired = target_position + np.asarray(self.cfg.offset_ned, dtype=np.float64)
        relative_velocity = np.asarray(state["interceptor_velocity"])-target_velocity
        correction = float(self.cfg.position_gain)*(desired-interceptor_position)
        correction -= float(self.cfg.relative_velocity_gain)*relative_velocity
        norm = np.linalg.norm(correction)
        correction *= min(1., float(self.cfg.maximum_relative_speed)/max(norm, 1e-9))
        velocity_ned = target_velocity + correction
        timestamp = state.get("timestamp_seconds")
        if timestamp is not None and self._previous_timestamp is not None:
            dt = float(timestamp)-self._previous_timestamp
            if .01 <= dt <= 1.:
                acceleration = np.clip((target_velocity-self._previous_target_velocity)/dt, -2., 2.)
                self._target_acceleration = .7*self._target_acceleration + .3*acceleration
                velocity_ned += float(self.cfg.target_acceleration_feedforward_seconds)*self._target_acceleration
        self._previous_target_velocity = target_velocity.copy()
        self._previous_timestamp = None if timestamp is None else float(timestamp)
        horizontal = (target_position-interceptor_position)[:2]
        separation = float(np.linalg.norm(horizontal))
        if separation > 1e-6:
            direction = horizontal/separation
            actual_closing = float(np.dot(relative_velocity[:2], direction))
            predicted = separation-max(0., actual_closing)*float(self.cfg.braking_lookahead_seconds)
            limit = float(self.cfg.close_approach_gain)*(predicted-float(self.cfg.minimum_horizontal_separation_m))
            # Establish the vertical offset before the close formation approach.
            if separation < 2. and abs(float(desired[2]-interceptor_position[2])) > .04:
                limit = min(limit, 0.)
            requested = float(np.dot((velocity_ned-target_velocity)[:2], direction))
            if requested > limit:
                velocity_ned[:2] -= (requested-limit)*direction
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
    return "gazebo_cooperative_rendezvous_v1"
