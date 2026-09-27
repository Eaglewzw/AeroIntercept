import json
from pathlib import Path

import numpy as np
import pytest

from aerointercept.gazebo.dataset_integrity import recover_summaries, save_episode, merge_collections
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.scripts.collect_bc_data import collect_episode
from aerointercept.training.train_e2e_bc import validate_initialization_split


def test_recovery_keeps_terminal_result_when_progress_was_not_written(tmp_path):
    summary = {"length": 2, "mode": "circle", "outcome": "timeout",
               "minimum_distance": 2.3, "episode_reward": -5.0}
    save_episode(tmp_path / "episode_000000.npz", {"actions": np.zeros((2, 4))}, summary)
    assert recover_summaries(tmp_path, [("circle", 1)]) == [summary]


def test_legacy_recovery_does_not_fabricate_success(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    np.savez(episodes / "episode_000000.npz", actions=np.zeros((3, 4)))
    (tmp_path / "collection_progress.json").write_text(json.dumps({"summaries": [
        {"length": 3, "mode": "circle", "outcome": "hit",
         "recovered_after_interruption": True},
    ]}))
    result = recover_summaries(episodes, [("circle", 1)])[0]
    assert result["outcome"] == "unknown"
    assert result["minimum_distance"] is None


def test_recovery_rejects_changed_modes_and_excess_shards(tmp_path):
    summary = {"length": 1, "mode": "circle", "outcome": "timeout"}
    save_episode(tmp_path / "episode_000000.npz", {"actions": np.zeros((1, 4))}, summary)
    with pytest.raises(ValueError, match="does not match"):
        recover_summaries(tmp_path, [("sinusoidal", 1)])
    with pytest.raises(ValueError, match="more episodes"):
        recover_summaries(tmp_path, [])


class _CollectionEnv:
    def __init__(self, outcome="fov_lost"):
        self.cfg = load_gazebo_config("configs/gazebo_feedback.yaml")
        self.outcome = outcome
        self.index = 0
        self._last_snapshot = {}

    def observation(self):
        stamp = (self.index+1)*200_000_000
        self._last_snapshot.update(self_state_timestamp_ns=stamp, image_timestamp_ns=stamp,
                                   contact_monitor_ready=True, contact_count=0,
                                   camera_relative_frd=[1.5, 0., 0.])
        return np.zeros((2, 3, 4, 6), dtype=np.uint8), {
            "future_position": np.zeros(3), "collision_risk": 0.,
            "confidence": float(self.index < 8), "critic_obs": np.zeros(15),
        }

    def reset(self):
        frames, labels = self.observation()
        return frames, labels, {"target_distance_m": 10.}

    def actor_self_state(self):
        return np.zeros(6, dtype=np.float32)

    def expert_state(self):
        return {"target_position": np.array([1.5, 0., 0.]),
                "interceptor_position": np.zeros(3), "target_velocity": np.zeros(3),
                "interceptor_velocity": np.zeros(3)}

    def step(self, action):
        self.index += 1
        frames, labels = self.observation()
        done = self.index == 12
        final = {"outcome": self.outcome, "minimum_distance": 1.5, "episode_reward": -1.,
                 "contact_count": int(self.outcome == "contact")}
        return frames, 0., done, False, labels, {"final": final} if done else {}


class _CollectionExpert:
    def reset(self):
        pass

    def action(self, state):
        return np.ones(4, dtype=np.float32)*.2


def test_correction_uses_constant_mixture_and_preserves_negative_observations():
    arrays, summary = collect_episode(_CollectionEnv(), _CollectionExpert(),
                                     lambda frames, own: np.zeros(4), .3)
    assert summary["outcome"] == "fov_lost"
    assert summary["valid_supervision_frames"] == 12
    assert np.allclose(arrays["actions"], .2)
    assert np.allclose(arrays["executed_actions"], .06)
    assert np.allclose(arrays["teacher_weight"], .3)
    assert summary["near_2m_policy_frames"] == 12
    assert np.count_nonzero(arrays["confidence"] == 0) == 4
    assert np.allclose(arrays["center_distance_m"], 1.5)


def test_physical_failure_excludes_preceding_second_without_relabelling():
    arrays, summary = collect_episode(_CollectionEnv("contact"), _CollectionExpert())
    assert summary["outcome"] == "contact"
    assert summary["contact_count"] == 1
    assert arrays["supervision_valid"].tolist() == [1]*6+[0]*6
    assert summary["invisible_frames"] == 4
    assert summary["valid_invisible_frames"] == 0


def test_touching_the_target_keeps_the_terminal_approach_under_supervision():
    """A target contact is the success this task asks for, not a failure tail."""
    class TargetContact(_CollectionEnv):
        def step(self, action):
            frames, reward, done, truncated, labels, info = super().step(action)
            if done:
                info["final"] = {**info["final"], "target_contact": True}
            return frames, reward, done, truncated, labels, info

    arrays, summary = collect_episode(TargetContact("contact"), _CollectionExpert())
    assert summary["outcome"] == "contact"
    assert summary["target_contact"] is True
    # The whole approach is kept; a contacted input state would still be masked.
    assert arrays["supervision_valid"].tolist() == [1]*12
    assert summary["valid_supervision_frames"] == 12


@pytest.mark.parametrize("weight,valid_count", [(0., 12), (.5, 12), (1., 6)])
def test_policy_failure_retains_teacher_corrections_not_executed_actions(weight, valid_count):
    arrays, summary = collect_episode(_CollectionEnv("contact"), _CollectionExpert(),
                                     lambda frames, own: np.zeros(4), weight)
    assert summary["outcome"] == "contact"
    assert summary["contact_count"] == 1
    assert summary["valid_supervision_frames"] == valid_count
    np.testing.assert_allclose(arrays["actions"], .2)
    np.testing.assert_allclose(arrays["executed_actions"], .2*weight)


def test_retaining_corrections_never_unmasks_contacted_input_states():
    class ContactedInputs(_CollectionEnv):
        def observation(self):
            result = super().observation()
            self._last_snapshot["contact_count"] = int(self.index >= 10)
            return result

    arrays, summary = collect_episode(ContactedInputs("contact"), _CollectionExpert(),
                                     lambda frames, own: np.zeros(4), 0.)
    assert arrays["supervision_valid"].tolist() == [1]*10 + [0]*2
    assert summary["outcome"] == "contact"


def test_warm_start_cannot_move_trained_scenarios_into_validation():
    manifest = {"summaries": [{"scenario": {"seed": 12, "mode": "circle"}}]}
    source = {"dataset_manifest": manifest, "dataset_split": {"train": ["episode_000000.npz"]}}
    with pytest.raises(ValueError, match="already trained"):
        validate_initialization_split(source, manifest, [Path("episode_000000.npz")])
    validate_initialization_split(source, {"summaries": [{"scenario": {"seed": 13, "mode": "circle"}}]},
                                  [Path("episode_000000.npz")])


def _write_collection(root, outcome, task, seed):
    contract = {key: {} for key in ("task", "expert", "action", "camera", "labels", "self_state")}
    contract.update(task=task, camera_target_label="measured",
                    supervision_protocol="measured_safe_fragments_v1")
    (root/"episodes").mkdir(parents=True)
    summary = {"length": 2, "mode": "circle", "outcome": outcome,
               "scenario": {"mode": "circle", "seed": seed}}
    save_episode(root/"episodes/episode_000000.npz", {"actions": np.zeros((2, 4))}, summary)
    (root/"manifest.json").write_text(json.dumps(
        {"collection_config": contract, "episodes": 1, "summaries": [summary]}))


def test_collection_merge_preserves_outcomes_provenance_and_rejects_duplicates(tmp_path):
    for index, outcome in enumerate(("hit", "fov_lost")):
        _write_collection(tmp_path/str(index), outcome, {}, index)
    merged = merge_collections([tmp_path/"0", tmp_path/"1"], tmp_path/"merged")
    assert [row["outcome"] for row in merged["summaries"]] == ["hit", "fov_lost"]
    assert merged["summaries"][1]["source_collection"] == 1
    assert merged["frames"] == 4
    assert (tmp_path/"0/episodes/episode_000000.npz").read_bytes() == (tmp_path/"merged/episodes/episode_000000.npz").read_bytes()
    with pytest.raises(ValueError, match="duplicate scenario"):
        merge_collections([tmp_path/"0", tmp_path/"merged"], tmp_path/"duplicate")
    assert not (tmp_path/"duplicate").exists()


def test_collection_merge_accepts_legacy_rendezvous_and_rejects_a_different_task(tmp_path):
    """The replay workflow mixes recordings made before and after the migration."""
    current = load_gazebo_config("configs/gazebo_feedback.yaml").gazebo.task
    legacy = {**current, "task_version": "noncontact_rendezvous_v1",
              "rendezvous_max_relative_speed_mps": .5, "rendezvous_hold_seconds": .3,
              "rendezvous_settle_seconds": 1.}
    _write_collection(tmp_path/"legacy", "hit", legacy, 1)
    _write_collection(tmp_path/"current", "hit", dict(current), 2)
    assert merge_collections([tmp_path/"legacy", tmp_path/"current"],
                             tmp_path/"merged")["episodes"] == 2
    _write_collection(tmp_path/"other", "hit", {**current, "hit_radius": .9}, 3)
    with pytest.raises(ValueError, match="incompatible collection tasks"):
        merge_collections([tmp_path/"legacy", tmp_path/"other"], tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()
