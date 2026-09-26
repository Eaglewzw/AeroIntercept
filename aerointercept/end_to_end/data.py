"""Episode-sharded dataset utilities for end-to-end behavior cloning."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


DATASET_SCHEMA_VERSION = 1
REQUIRED_ARRAYS = (
    "frames",
    "actions",
    "future_position",
    "collision_risk",
    "confidence",
)


def episode_files(data_dir) -> list[Path]:
    paths = sorted((Path(data_dir) / "episodes").glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards found under {data_dir}")
    return paths


def load_manifest(data_dir) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    version = int(manifest.get("schema_version", -1))
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported dataset schema {version}; expected "
            f"{DATASET_SCHEMA_VERSION}")
    return manifest


def split_episode_files(paths, validation_fraction: float, seed: int, *, strata=None, coverage=None):
    """Split at episode granularity so adjacent frames never leak to val."""
    paths = list(paths)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if strata is not None:
        if len(strata) != len(paths) or any(not item for item in strata):
            raise ValueError("one nonempty stratum is required per episode")
        train, validation = [], []
        for label in sorted(set(strata)):
            group = [path for path, value in zip(paths, strata) if value == label]
            if len(group) < 2:
                raise ValueError(f"stratum {label!r} needs at least two independent episodes")
            group_train, group_validation = split_episode_files(
                group, validation_fraction, seed)
            train.extend(group_train)
            validation.extend(group_validation)
        return _ensure_split_coverage(paths, train, validation, strata, coverage)
    if len(paths) < 2:
        raise ValueError("at least two episodes are required for train/validation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    validation_count = max(1, int(round(len(paths) * validation_fraction)))
    validation_count = min(validation_count, len(paths) - 1)
    validation_indices = set(order[:validation_count].tolist())
    train = [path for index, path in enumerate(paths)
             if index not in validation_indices]
    validation = [path for index, path in enumerate(paths)
                  if index in validation_indices]
    return _ensure_split_coverage(paths, train, validation, [None]*len(paths), coverage)


def _ensure_split_coverage(paths, train, validation, strata, coverage):
    """Keep mode counts fixed while covering rare episode classes on both sides."""
    if coverage is None:
        return train, validation
    if len(coverage) != len(paths) or len(set(paths)) != len(paths):
        raise ValueError("coverage requires one class per distinct episode")
    labels = dict(zip(paths, coverage))
    groups = dict(zip(paths, strata))
    classes = sorted(set(coverage))
    if any(coverage.count(label) < 2 for label in classes):
        raise ValueError("each coverage class needs at least two independent episodes")
    for destination, donor in ((validation, train), (train, validation)):
        for label in classes:
            if any(labels[path] == label for path in destination):
                continue
            swap = next(((new, old) for new in donor if labels[new] == label
                         for old in destination if groups[new] == groups[old]
                         and sum(labels[path] == labels[old] for path in destination) > 1
                         and sum(labels[path] == label for path in donor) > 1), None)
            if swap is None:
                raise ValueError("cannot cover episode classes without leaking or removing a mode; collect more episodes")
            new, old = swap
            destination[destination.index(old)] = new
            donor[donor.index(new)] = old
    return train, validation


class EpisodeSequenceDataset(Dataset):
    """Read fixed windows with a bounded cache of decompressed episodes.

    The final window is padded by repeating its last valid state and accompanied
    by a mask.  Frame histories are built inside an episode, never across reset
    boundaries.
    """

    def __init__(self, paths, sequence_length: int, history_frames: int = 2,
                 cache_size: int = 16, self_state_dim: int = 0,
                 camera_supervision: bool = False):
        self.paths = [Path(path) for path in paths]
        self.sequence_length = int(sequence_length)
        self.history_frames = int(history_frames)
        if self.sequence_length < 1 or self.history_frames < 1:
            raise ValueError("sequence_length and history_frames must be positive")
        self.cache_size = max(1, int(cache_size))
        self.self_state_dim = int(self_state_dim)
        self.camera_supervision = bool(camera_supervision)
        self._cache = OrderedDict()
        self._windows = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as shard:
                length = self._validate_shard(path, shard)
                if self.camera_supervision:
                    if ("camera_target_xy" not in shard or shard["camera_target_xy"].shape != (length, 2)
                            or not np.isfinite(shard["camera_target_xy"]).all()):
                        raise ValueError(f"{path}: missing or invalid measured camera projection labels")
                if self.self_state_dim:
                    if "self_state" not in shard or shard["self_state"].shape != (length, self.self_state_dim):
                        raise ValueError(f"{path}: missing or incompatible measured self_state")
                    if not np.isfinite(shard["self_state"]).all():
                        raise ValueError(f"{path}: non-finite self_state")
                    for key in ("self_state_timestamp_ns", "image_timestamp_ns"):
                        if key not in shard or shard[key].shape != (length,):
                            raise ValueError(f"{path}: missing own-state synchronization evidence")
                    ages = shard["image_timestamp_ns"]-shard["self_state_timestamp_ns"]
                    if np.any(ages < 0) or np.any(ages > 200_000_000):
                        raise ValueError(f"{path}: future or stale self-state observations")
                valid = np.ones(length, dtype=bool)
                if "center_distance_m" in shard:
                    distance = shard["center_distance_m"]
                    if distance.shape != (length,) or not np.isfinite(distance).all() or (distance < 0).any():
                        raise ValueError(f"{path}: invalid measured center distances")
                if "supervision_valid" in shard:
                    recorded = shard["supervision_valid"]
                    if recorded.shape != (length,) or not np.isin(recorded, [0, 1]).all():
                        raise ValueError(f"{path}: invalid supervision_valid mask")
                    valid = recorded.astype(bool)
            # Invalid transitions form actual boundaries, not holes through
            # which frame histories or future recurrent sequences may pass.
            edges = np.diff(np.r_[False, valid, False].astype(np.int8))
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                for start in range(int(begin), int(end), self.sequence_length):
                    self._windows.append((path, start, int(end), int(begin)))
        if not self._windows:
            raise ValueError("dataset contains no transitions")

    @staticmethod
    def _validate_shard(path, shard) -> int:
        missing = [name for name in REQUIRED_ARRAYS if name not in shard]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        # NpzFile does not cache decompressed arrays. Read RGB once; repeatedly
        # indexing frames here inflated 640x640 dataset startup I/O fivefold.
        frames = shard["frames"]
        length = int(frames.shape[0])
        if length < 1:
            raise ValueError(f"{path} is empty")
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"{path}: frames must have shape [T,3,H,W]")
        if frames.dtype != np.uint8:
            raise ValueError(f"{path}: frames must use uint8 storage")
        if shard["actions"].shape != (length, 4):
            raise ValueError(f"{path}: actions must have shape [T,4]")
        if shard["future_position"].shape != (length, 3):
            raise ValueError(f"{path}: future_position must have shape [T,3]")
        for name in ("collision_risk", "confidence"):
            if shard[name].shape != (length,):
                raise ValueError(f"{path}: {name} must have shape [T]")
        for name in REQUIRED_ARRAYS[1:]:
            if int(shard[name].shape[0]) != length:
                raise ValueError(f"{path}: {name} length does not match frames")
        return length

    def __len__(self):
        return len(self._windows)

    def _load(self, path: Path):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(path, allow_pickle=False) as shard:
            arrays = {name: shard[name] for name in REQUIRED_ARRAYS}
            if self.self_state_dim:
                arrays["self_state"] = shard["self_state"]
            if self.camera_supervision:
                arrays["camera_target_xy"] = shard["camera_target_xy"]
            if "center_distance_m" in shard:
                arrays["center_distance_m"] = shard["center_distance_m"]
        self._cache[key] = arrays
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        path, start, episode_length, segment_begin = self._windows[index]
        arrays = self._load(path)
        valid_length = min(self.sequence_length, episode_length - start)
        raw_indices = np.arange(start, start + valid_length)

        history_indices = []
        for current in raw_indices:
            first = current - self.history_frames + 1
            history_indices.append([
                max(segment_begin, first + offset)
                for offset in range(self.history_frames)
            ])
        frames = arrays["frames"][np.asarray(history_indices)]

        result = {"frames": frames}
        for name in REQUIRED_ARRAYS[1:]:
            result[name] = arrays[name][raw_indices]
        if self.self_state_dim:
            result["self_state"] = arrays["self_state"][raw_indices]
        if self.camera_supervision:
            result["camera_target_xy"] = arrays["camera_target_xy"][raw_indices]
        if "center_distance_m" in arrays:
            result["center_distance_m"] = arrays["center_distance_m"][raw_indices]
        mask = np.ones(valid_length, dtype=np.float32)

        padding = self.sequence_length - valid_length
        if padding:
            for name, values in tuple(result.items()):
                repeated = np.repeat(values[-1:], padding, axis=0)
                result[name] = np.concatenate([values, repeated], axis=0)
            mask = np.concatenate([
                mask, np.zeros(padding, dtype=np.float32)])
        result["mask"] = mask
        return result
