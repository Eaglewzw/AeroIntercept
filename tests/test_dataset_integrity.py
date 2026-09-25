import json

import numpy as np
import pytest

from aerointercept.gazebo.dataset_integrity import recover_summaries, save_episode


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
