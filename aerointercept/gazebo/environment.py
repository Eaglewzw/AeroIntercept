"""Gazebo/PX4 environment separating deployable sensors from target truth."""

from __future__ import annotations

import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .client import GazeboBridgeClient
from .protocol import BridgeProtocolError, image_from_snapshot
from .scenarios import Scenario, TRAIN_MODES
from .self_state import SELF_STATE_SOURCE
from .visual_supervision import camera_target_xy
from .task_logic import (
    build_training_info,
    compute_reward,
    segment_minimum_distance,
    camera_target_yaw_geometry,
    target_visibility,
    termination_flags,
    vertical_clearance_penalty,
)


@dataclass
class EpisodeStats:
    reward: float = 0.0
    length: int = 0
    minimum_distance: float = float("inf")


class GazeboInterceptEnv:
    """One physical PX4 pair in one running Gazebo world.

    Simulator truth is returned separately in ``training_info`` and never
    enters the Actor observation.  ``reset`` is a physical PX4 position
    setpoint return, not an object teleport or renderer reset.
    """

    observation_shape = (2, 3, 640, 640)
    action_shape = (4,)

    def __init__(
        self,
        cfg,
        socket_path: str | Path,
        *,
        client_factory: Callable[..., GazeboBridgeClient] = GazeboBridgeClient,
        connect: bool = True,
        mode: str | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg
        self.task_cfg = cfg.gazebo.task
        self.reward_cfg = cfg.gazebo.rewards
        self.label_cfg = cfg.end_to_end.labels
        self.client = client_factory(socket_path, timeout=float(cfg.gazebo.bridge.timeout_s))
        status = {}
        if connect:
            status = self.client.connect(float(cfg.gazebo.bridge.startup_timeout_s))
            if status.get("backend") != "gazebo_px4":
                raise BridgeProtocolError("connected process is not the Gazebo/PX4 bridge")
        self._scenario_control = status.get("scenario_control") == "bounded_cooperative_v1"
        self.mode = mode or status.get("default_mode", "mixed")
        self.seed = int(status.get("default_seed", 0) if seed is None else seed)
        self.episode_index = 0
        self._hold_started_ns = None
        self._history = np.empty(self.observation_shape, dtype=np.uint8)
        self._last_snapshot: dict | None = None
        self._last_action = np.zeros(4, dtype=np.float32)
        self._previous_distance = float("inf")
        self._lost_count = 0
        self._episode = EpisodeStats()
        self._episode_started_wall = None
        self._episode_started_image_ns = None
        self.camera_frames = 0

    @staticmethod
    def _physical_state(snapshot: dict) -> dict:
        keys = (
            "interceptor_position", "interceptor_velocity",
            "target_position", "target_velocity", "interceptor_yaw",
        )
        missing = [key for key in keys if key not in snapshot]
        if missing:
            raise BridgeProtocolError(f"snapshot missing physical state: {missing}")
        return {key: snapshot[key] for key in keys}

    def _observation_and_training(self, snapshot: dict) -> tuple[np.ndarray, dict, dict]:
        image = image_from_snapshot(snapshot)
        self.camera_frames += 1
        state = self._physical_state(snapshot)
        relative = np.asarray(state["target_position"], dtype=np.float64) - np.asarray(
            state["interceptor_position"], dtype=np.float64
        )
        yaw = float(state["interceptor_yaw"])
        camera_yaw = yaw + float(self.cfg.gazebo.camera.mount_yaw_offset_rad)
        cy, sy = np.cos(camera_yaw), np.sin(camera_yaw)
        relative_body = np.array([
            cy * relative[0] + sy * relative[1],
            -sy * relative[0] + cy * relative[1],
            relative[2],
        ])
        if "camera_relative_frd" in snapshot:
            relative_body = np.asarray(snapshot["camera_relative_frd"], dtype=np.float64)
        visible, center_error = target_visibility(
            relative_body,
            float(self.cfg.gazebo.camera.horizontal_fov),
            float(self.cfg.gazebo.camera.vertical_fov),
            float(self.task_cfg.get("target_bounding_radius_m", 0.0)),
        )
        training = build_training_info(state, self.label_cfg, visible)
        if float(self.cfg.end_to_end.auxiliary.get("spatial_coef", 0.)) > 0:
            xy = camera_target_xy(relative_body, float(self.cfg.gazebo.camera.horizontal_fov))
            training["camera_target_xy"] = xy
            aspect = float(self.cfg.gazebo.camera.source_height)/float(self.cfg.gazebo.camera.source_width)
            training["camera_target_valid"] = np.float32(
                visible and relative_body[0] > 0 and abs(xy[0]) <= 1 and abs(xy[1]) <= aspect)
        metrics = {
            "visible": visible,
            "fov_center_error": center_error,
            "distance": float(np.linalg.norm(relative)),
        }
        return image, training, metrics

    def reset(self) -> tuple[np.ndarray, dict, dict]:
        home = np.asarray(self.task_cfg.reset_position_ned, dtype=np.float64)
        look_at_target = bool(getattr(self.task_cfg, "look_at_target", True))
        mount_yaw_offset = float(self.cfg.gazebo.camera.mount_yaw_offset_rad)
        reset_kwargs = {"look_at_target": look_at_target}
        if self._scenario_control:
            selected_mode = TRAIN_MODES[self.episode_index % len(TRAIN_MODES)] if self.mode == "mixed" else self.mode
            scenario = Scenario.sample(selected_mode, self.seed+self.episode_index, home,
                                       float(self.task_cfg.reset_target_distance_m))
            from dataclasses import asdict
            reset_kwargs["scenario"] = asdict(scenario)
        self.client.reset(home, float(self.task_cfg.reset_yaw), **reset_kwargs)
        deadline = time.monotonic() + float(self.task_cfg.reset_timeout_s)
        sequence = -1
        snapshot = None
        first_snapshot = None
        last_position = None
        last_velocity = None
        last_target_distance = None
        while time.monotonic() < deadline:
            snapshot = self.client.snapshot(sequence, timeout=2.0)
            sequence = int(snapshot["sequence"])
            position = np.asarray(snapshot["interceptor_position"], dtype=np.float64)
            velocity = np.asarray(snapshot["interceptor_velocity"], dtype=np.float64)
            last_position = position
            last_velocity = velocity
            target_position = np.asarray(snapshot["target_position"], dtype=np.float64)
            target_distance = float(np.linalg.norm(target_position - position))
            last_target_distance = target_distance
            target_status = snapshot.get("target_vehicle_status")
            _, desired_yaw, yaw_error = camera_target_yaw_geometry(
                position, target_position, float(snapshot["interceptor_yaw"]),
                mount_yaw_offset,
            )
            target_ready = (
                -float(target_position[2]) >= float(self.task_cfg.target_minimum_altitude)
                and (not self._scenario_control or np.linalg.norm(snapshot["target_velocity"])
                     <= float(self.task_cfg.reset_speed_tolerance))
                and (
                    target_status is None
                    or (target_status.get("armed") and target_status.get("offboard"))
                )
            )
            if (
                np.linalg.norm(position - home) <= float(self.task_cfg.reset_tolerance_m)
                and np.linalg.norm(velocity) <= float(self.task_cfg.reset_speed_tolerance)
                and abs(
                    target_distance - float(self.task_cfg.reset_target_distance_m)
                ) <= float(self.task_cfg.reset_target_distance_tolerance_m)
                and target_ready
                and (
                    not look_at_target
                    or abs(yaw_error) <= float(self.task_cfg.reset_yaw_tolerance_rad)
                )
            ):
                # Both real images entering the Actor must satisfy reset
                # constraints. A valid first image alone does not establish
                # the actual start distance of the second image.
                if first_snapshot is not None:
                    break
                first_snapshot = snapshot
            else:
                first_snapshot = None
        else:
            raise TimeoutError(
                "PX4 did not physically return to the reset setpoint; "
                f"last_position={last_position} last_velocity={last_velocity} "
                f"status={None if snapshot is None else snapshot.get('vehicle_status')} "
                f"target={None if snapshot is None else snapshot.get('target_position')} "
                f"target_distance={last_target_distance} "
                f"camera_target_yaw_error={None if snapshot is None else snapshot.get('camera_target_yaw_error')} "
                f"target_status={None if snapshot is None else snapshot.get('target_vehicle_status')}"
            )
        assert snapshot is not None and first_snapshot is not None
        first, _, _ = self._observation_and_training(first_snapshot)
        second_snapshot = snapshot
        second, training, metrics = self._observation_and_training(second_snapshot)
        if bool(self.task_cfg.get("require_contact_monitor", False)) and not second_snapshot.get("contact_monitor_ready"):
            raise BridgeProtocolError("noncontact task requires a verified Gazebo contact monitor")
        if self.task_cfg.get("task_version") == "noncontact_rendezvous_v1" and second_snapshot.get(
            "physical_state_source"
        ) != "gazebo_base_link_center_enu_to_ned_v2":
            raise BridgeProtocolError("noncontact task requires Gazebo base_link center truth")
        self._history[0] = first
        self._history[1] = second
        self._last_snapshot = second_snapshot
        self._last_action.fill(0.0)
        self._previous_distance = metrics["distance"]
        self._lost_count = 0 if metrics["visible"] else 1
        self._episode = EpisodeStats(minimum_distance=metrics["distance"])
        self._episode_started_wall = time.monotonic()
        self._episode_started_image_ns = second_snapshot.get("image_timestamp_ns")
        self._hold_started_ns = None
        self.episode_index += 1
        return self._history.copy(), training, {
            "camera": second_snapshot.get("camera_metadata", {}),
            "backend": "gazebo_px4",
            "scenario": second_snapshot.get("scenario"),
            "physical_state_source": second_snapshot.get("physical_state_source"),
            "look_at_target": look_at_target,
            "target_bearing_rad": float(second_snapshot.get("target_bearing", desired_yaw)),
            "interceptor_yaw_rad": float(second_snapshot["interceptor_yaw"]),
            "camera_target_yaw_error_rad": float(
                second_snapshot.get("camera_target_yaw_error", yaw_error)
            ),
            "target_distance_m": float(metrics["distance"]),
            "requested_target_distance_m": float(
                self.task_cfg.reset_target_distance_m
            ),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict, dict]:
        if self._last_snapshot is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_shape or not np.isfinite(action).all():
            raise ValueError("Actor action must be a finite [forward,right,down,yaw_rate] vector")
        action = np.clip(action, -1.0, 1.0)
        decoded = self.client.action(action)
        previous_snapshot = self._last_snapshot
        snapshot_kwargs = {}
        if "after_image_ns" in decoded:
            snapshot_kwargs["after_image_ns"] = decoded["after_image_ns"]
        snapshot = self.client.snapshot(
            max(int(previous_snapshot["sequence"]), int(decoded.get("after_sequence", -1))),
            timeout=3.0, **snapshot_kwargs,
        )
        image, training, metrics = self._observation_and_training(snapshot)
        previous_state = self._physical_state(previous_snapshot)
        state = self._physical_state(snapshot)
        finite = all(
            np.isfinite(np.asarray(value, dtype=np.float64)).all()
            for value in state.values()
        )
        minimum = segment_minimum_distance(
            previous_state["interceptor_position"],
            state["interceptor_position"],
            previous_state["target_position"],
            state["target_position"],
        ) if finite else float("inf")
        self._lost_count = 0 if metrics["visible"] else self._lost_count + 1
        self._episode.length += 1
        self._episode.minimum_distance = min(self._episode.minimum_distance, minimum)
        relative_speed = float(np.linalg.norm(
            np.asarray(state["target_velocity"])-state["interceptor_velocity"]
        ))
        contact = int(snapshot.get("contact_count", 0)) > 0
        timestamp_ns = int(snapshot.get("interceptor_timestamp", 0))*1000
        eligible = (metrics["distance"] <= float(self.task_cfg.hit_radius)
                    and relative_speed <= float(self.task_cfg.get("rendezvous_max_relative_speed_mps", .5))
                    and not contact and finite)
        if eligible and self._hold_started_ns is None:
            self._hold_started_ns = timestamp_ns
        if not eligible:
            self._hold_started_ns = None
        held_seconds = 0.0 if self._hold_started_ns is None else (timestamp_ns-self._hold_started_ns)*1e-9
        flags = termination_flags(
            step_minimum_distance=minimum,
            lost_count=self._lost_count,
            interceptor_position=state["interceptor_position"],
            invalid=not finite,
            episode_step=self._episode.length,
            cfg=self.task_cfg,
            current_distance=metrics["distance"], relative_speed=relative_speed,
            held_seconds=held_seconds, contact=contact,
            contact_monitor_ready=bool(snapshot.get("contact_monitor_ready", False)),
        )
        settle_seconds = float(self.task_cfg.get("rendezvous_settle_seconds", 0.0))
        trigger_distance = metrics["distance"] if flags["hit"] else None
        trigger_relative_speed = relative_speed if flags["hit"] else None
        if flags["hit"] and self._scenario_control and settle_seconds > 0:
            # Verify the terminal braking maneuver too. A delayed contact cannot
            # turn a physical failure into a recorded successful demonstration.
            self.client.hold()
            deadline = time.monotonic() + 10.0
            settle_start_ns = int(snapshot["interceptor_timestamp"])*1000
            while time.monotonic() < deadline:
                snapshot = self.client.snapshot(int(snapshot["sequence"]), timeout=3.)
                state = self._physical_state(snapshot)
                image, training, metrics = self._observation_and_training(snapshot)
                contact = int(snapshot.get("contact_count", 0)) > 0
                relative_speed = float(np.linalg.norm(
                    np.asarray(state["target_velocity"])-state["interceptor_velocity"]
                ))
                self._episode.minimum_distance = min(self._episode.minimum_distance, metrics["distance"])
                elapsed = (int(snapshot["interceptor_timestamp"])*1000-settle_start_ns)*1e-9
                exit_flags = termination_flags(
                    step_minimum_distance=metrics["distance"], lost_count=0,
                    interceptor_position=state["interceptor_position"],
                    invalid=not all(np.isfinite(np.asarray(value)).all() for value in state.values()),
                    episode_step=0, cfg=self.task_cfg, contact=contact,
                    contact_monitor_ready=bool(snapshot.get("contact_monitor_ready", False)),
                )
                physical_failures = ("contact", "ground", "invalid", "out_of_bounds")
                if any(exit_flags[key] for key in physical_failures):
                    flags["hit"] = False
                    for key in physical_failures:
                        flags[key] = exit_flags[key]
                    break
                if elapsed >= settle_seconds and snapshot.get("hold_complete", False):
                    break
            else:
                raise TimeoutError("no simulator progress during terminal noncontact verification")
        reward, reward_terms = compute_reward(
            previous_distance=self._previous_distance,
            distance=metrics["distance"],
            visible=metrics["visible"],
            center_error=metrics["fov_center_error"],
            action=action,
            previous_action=self._last_action,
            flags=flags,
            cfg=self.reward_cfg,
        )
        if self.task_cfg.get("task_version") == "noncontact_rendezvous_v1":
            reward_terms["relative_speed"] = -.2*relative_speed**2 if metrics["distance"] < 2. else 0.
            reward += reward_terms["relative_speed"]
            reward_terms["vertical_clearance"] = vertical_clearance_penalty(
                np.asarray(state["target_position"])-np.asarray(state["interceptor_position"]), self.reward_cfg)
            reward += reward_terms["vertical_clearance"]
        self._episode.reward += reward
        self._history[0] = self._history[1]
        self._history[1] = image
        self._last_snapshot = snapshot
        self._last_action = action.copy()
        self._previous_distance = metrics["distance"]
        terminated = bool(flags["terminated"])
        truncated = bool(flags["timed_out"])
        info = {
            **metrics,
            "reward_terms": reward_terms,
            "termination": flags,
            "decoded_action": decoded,
            "relative_speed": relative_speed,
            "rendezvous_held_seconds": held_seconds,
        }
        if terminated or truncated:
            outcome_order = ("invalid", "contact", "ground", "out_of_bounds", "fov_lost", "hit")
            outcome = next((name for name in outcome_order if flags[name]), "timeout")
            info["final"] = {
                "outcome": outcome,
                "episode_reward": self._episode.reward,
                "episode_length": self._episode.length,
                "minimum_distance": self._episode.minimum_distance,
                "final_center_distance_m": metrics["distance"],
                "rendezvous_center_distance_m": trigger_distance,
                "rendezvous_relative_speed_mps": trigger_relative_speed,
                "final_relative_speed_mps": relative_speed,
                "rendezvous_held_seconds": held_seconds,
                "contact_count": int(snapshot.get("contact_count", 0)),
                "last_contact": snapshot.get("last_contact") if contact else None,
                "episode_wall_seconds": time.monotonic() - self._episode_started_wall,
                "episode_simulation_seconds": image_elapsed_seconds(
                    self._episode_started_image_ns, snapshot.get("image_timestamp_ns"),
                ),
            }
            if self._scenario_control:
                self.client.hold()
        return self._history.copy(), reward, terminated, truncated, training, info

    def expert_state(self) -> dict[str, np.ndarray | float]:
        """Return copied simulator truth exclusively for offline label generation.

        This method is deliberately separate from both the observation and the
        Actor API.  PPO/evaluation never call it; the Experiment C dataset
        collector uses it to query an analytical Gazebo expert.
        """
        if self._last_snapshot is None:
            raise RuntimeError("reset must be called before requesting expert state")
        state = self._physical_state(self._last_snapshot)
        result = {
            key: float(value) if key == "interceptor_yaw" else np.asarray(
                value, dtype=np.float64,
            ).copy()
            for key, value in state.items()
        }
        result["timestamp_seconds"] = float(self._last_snapshot.get("interceptor_timestamp", 0))*1e-6
        return result

    def actor_self_state(self) -> np.ndarray | None:
        """Copy only the causal PX4 own-vehicle sample for the Actor."""
        if not int(self.cfg.end_to_end.model.get("self_state_dim", 0)):
            return None
        snapshot = self._last_snapshot
        if snapshot is None or snapshot.get("self_state_source") != SELF_STATE_SOURCE:
            raise BridgeProtocolError("Actor requires measured PX4 own velocity and angular velocity")
        values = np.asarray(snapshot.get("self_state"), dtype=np.float32)
        age = float(snapshot.get("self_state_age_seconds", float("inf")))
        if values.shape != (6,) or not np.isfinite(values).all() or not 0 <= age <= .2:
            raise BridgeProtocolError("invalid or stale PX4 Actor self-state")
        return values.copy()

    def close(self) -> None:
        self.client.close()


def image_elapsed_seconds(start_ns: int | None, end_ns: int | None) -> float | None:
    """Use sensor timestamps; missing legacy clocks must not imply 20 Hz."""
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return None
    return (end_ns - start_ns) / 1_000_000_000.0


class GazeboVectorEnv:
    """Vector façade over independently launched Gazebo/PX4 worlds.

    Each socket must represent a distinct Gazebo partition, ROS_DOMAIN_ID,
    Micro XRCE port and PX4 pair.  This avoids pretending one asynchronous
    physical world is an in-process cloned environment.
    """

    def __init__(self, cfg, socket_paths: Sequence[str | Path], *, mode=None, seed=0, restart_world=None):
        if not socket_paths:
            raise ValueError("at least one Gazebo bridge socket is required")
        if len(set(map(str, socket_paths))) != len(socket_paths):
            raise ValueError("each vector environment requires a distinct socket")
        self.environments = [GazeboInterceptEnv(cfg, path, mode=mode, seed=seed+index*100000)
                             for index, path in enumerate(socket_paths)]
        self.num_envs = len(self.environments)
        self._restart_world = restart_world
        self._retired_camera_frames = 0

    def reset(self):
        results = [environment.reset() for environment in self.environments]
        frames = np.stack([result[0] for result in results])
        training = self._stack_training([result[1] for result in results])
        return frames, training, [result[2] for result in results]

    @staticmethod
    def _stack_training(items: list[dict]) -> dict[str, np.ndarray]:
        return {key: np.stack([item[key] for item in items]).astype(np.float32)
                for key in items[0]}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"expected actions [{self.num_envs},4], got {actions.shape}")
        results = [env.step(action) for env, action in zip(self.environments, actions)]
        frames, rewards, terminated, truncated, training, infos = map(list, zip(*results))
        dones = np.logical_or(terminated, truncated)
        # Preserve final episode data, then physically return completed PX4 vehicles home.
        for index in np.flatnonzero(dones):
            final = infos[index].get("final")
            if final and final["outcome"] in ("contact", "ground", "invalid") and self._restart_world is not None:
                self._replace_world(index)
            try:
                reset_frame, reset_training, reset_info = self.environments[index].reset()
            except (BridgeProtocolError, TimeoutError, ConnectionError, EOFError) as exc:
                if self._restart_world is None:
                    raise
                # Reset is outside the policy episode. Preserve its recorded
                # outcome and explicitly log a simulator recovery; never
                # fabricate an observation or a successful transition.
                infos[index]["reset_recovery"] = str(exc)
                self._replace_world(index)
                reset_frame, reset_training, reset_info = self.environments[index].reset()
            frames[index] = reset_frame
            training[index] = reset_training
            infos[index]["reset"] = reset_info
            infos[index]["final"] = final
        return (
            np.stack(frames),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            self._stack_training(training),
            infos,
        )

    def _replace_world(self, index: int) -> None:
        previous = self.environments[index]
        self._retired_camera_frames += previous.camera_frames
        previous.close()
        self._restart_world(index)
        replacement = GazeboInterceptEnv(previous.cfg, previous.client.socket_path,
                                         mode=previous.mode, seed=previous.seed)
        replacement.episode_index = previous.episode_index
        self.environments[index] = replacement

    @property
    def camera_frames(self) -> int:
        return self._retired_camera_frames + sum(environment.camera_frames for environment in self.environments)

    def actor_self_states(self):
        values = [env.actor_self_state() for env in self.environments]
        return None if values[0] is None else np.stack(values)

    def close(self) -> None:
        for environment in self.environments:
            environment.close()

    def set_paused(self, paused: bool) -> None:
        for environment in self.environments:
            environment.client.set_paused(paused)

    @contextmanager
    def paused(self):
        self.set_paused(True)
        try:
            yield
        finally:
            self.set_paused(False)
