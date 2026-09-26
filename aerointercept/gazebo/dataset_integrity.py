"""Lossless episode metadata and honest recovery of interrupted collections."""

from __future__ import annotations

import json
import copy
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


SUMMARY_KEY = "episode_summary_json"


def merge_collections(sources, output: Path) -> dict:
    """Combine immutable measured shards, retaining every source contract.

    Hard links avoid recompressing RGB or altering the embedded original
    outcome. The merged manifest identifies the source of each new filename.
    """
    sources = [Path(path).resolve() for path in sources]
    output = Path(output)
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("merge requires at least two distinct collections")
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((path/"manifest.json").read_text()) for path in sources]
    required_contract = ("task", "expert", "action", "camera", "labels", "camera_target_label",
                         "self_state", "supervision_protocol")
    required_manifest = ("schema_version", "backend", "image_width", "image_height", "history_frames",
                         "channel_order", "action_protocol", "velocity_max", "yaw_rate_max", "self_state_source")
    first = manifests[0]
    common = {key: first["collection_config"][key] for key in required_contract}
    if common["supervision_protocol"] != "measured_safe_fragments_v1":
        raise ValueError("merge requires recorded supervision fragments")
    for manifest in manifests:
        if any(manifest.get(key) != first.get(key) for key in required_manifest):
            raise ValueError("incompatible observation/action contracts")
        if any(manifest["collection_config"].get(key) != common[key] for key in required_contract):
            raise ValueError("incompatible collection contracts")
    merged = copy.deepcopy(first)
    merged.update(seed=None, mode="mixed", summaries=[], source_manifests=manifests)
    merged["collection_config"] = common
    common["sources"] = [{"path": str(path), "manifest_sha256": hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()}
                         for path in sources]
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_scenarios = set()
    with tempfile.TemporaryDirectory(prefix=".merge_", dir=output.parent) as temporary:
        staging = Path(temporary)/"dataset"
        (staging/"episodes").mkdir(parents=True)
        for source_index, (source, manifest) in enumerate(zip(sources, manifests)):
            paths = sorted((source/"episodes").glob("episode_*.npz"))
            summaries = manifest["summaries"]
            if len(paths) != len(summaries) or len(paths) != manifest["episodes"]:
                raise ValueError(f"incomplete source collection: {source}")
            for index, (path, item) in enumerate(zip(paths, summaries)):
                if path.name != f"episode_{index:06d}.npz":
                    raise ValueError(f"non-contiguous source collection: {source}")
                if not item.get("scenario"):
                    raise ValueError(f"missing measured scenario identity: {path}")
                scenario = json.dumps(item["scenario"], sort_keys=True)
                if scenario in seen_scenarios:
                    raise ValueError("duplicate scenario across merged episodes")
                seen_scenarios.add(scenario)
                destination_index = len(merged["summaries"])
                os.link(path, staging/"episodes"/f"episode_{destination_index:06d}.npz")
                merged["summaries"].append({**item, "episode": destination_index,
                                            "source_collection": source_index, "source_episode": index})
        merged["episodes"] = len(merged["summaries"])
        merged["frames"] = sum(item["length"] for item in merged["summaries"])
        merged["mode_plan"] = {mode: sum(item["mode"] == mode for item in merged["summaries"])
                               for mode in sorted({item["mode"] for item in merged["summaries"]})}
        (staging/"manifest.json").write_text(json.dumps(merged, indent=2, allow_nan=False))
        (staging/"collection_config.json").write_text(json.dumps(common, indent=2, allow_nan=False))
        staging.rename(output)
    return merged


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
