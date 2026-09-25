"""Lossless episode metadata and honest recovery of interrupted collections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SUMMARY_KEY = "episode_summary_json"


def save_episode(path: Path, arrays: dict, summary: dict) -> None:
    """Commit observations and their actual terminal outcome in one rename."""
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(
        temporary, **arrays,
        **{SUMMARY_KEY: np.asarray(json.dumps(summary, allow_nan=False))},
    )
    temporary.replace(path)


def recover_summaries(episodes_dir: Path, plan) -> list[dict]:
    """Never infer terminal success from pre-action training observations.

    Old shards do not contain the final transition. Their outcomes and true
    minimum distances are unknowable without a separately recorded result.
    Keep these explicitly unknown, even when an older recovery guessed a hit.
    """
    mode_sequence = [mode for mode, count in plan for _ in range(count)]
    paths = sorted(episodes_dir.glob("episode_*.npz"))
    if len(paths) > len(mode_sequence):
        raise ValueError("existing dataset has more episodes than requested")
    progress_path = episodes_dir.parent / "collection_progress.json"
    recorded = []
    if progress_path.exists():
        recorded = json.loads(progress_path.read_text())["summaries"]
    summaries = []
    for index, path in enumerate(paths):
        expected_name = f"episode_{index:06d}.npz"
        if path.name != expected_name:
            raise ValueError(
                f"resume requires contiguous shards; expected {expected_name}, got {path.name}"
            )
        with np.load(path, allow_pickle=False) as shard:
            length = int(shard["actions"].shape[0])
            if SUMMARY_KEY in shard:
                summary = json.loads(str(shard[SUMMARY_KEY].item()))
            elif index < len(recorded) and not recorded[index].get(
                "recovered_after_interruption", False,
            ):
                summary = dict(recorded[index])
            else:
                summary = {
                    "length": length, "outcome": "unknown",
                    "minimum_distance": None, "episode_reward": None,
                    "reset_target_distance_m": None,
                    "mode": mode_sequence[index],
                    "recovered_after_interruption": True,
                    "recovery_note": "legacy shard lacks recorded terminal state",
                }
        if summary["length"] != length or summary["mode"] != mode_sequence[index]:
            raise ValueError(f"recorded episode metadata does not match shard/plan: {path}")
        summaries.append(summary)
    return summaries
