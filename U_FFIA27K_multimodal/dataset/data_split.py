import os
import sys
import glob
import logging
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set
import numpy as np
from abc import ABC, abstractmethod

# Force stdout/stderr to use UTF-8 encoding to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from config import SplitterConfig


LABEL_TO_CLASS = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
LABEL_ORDER = [0, 1, 2, 3]
VALID_SPLIT_STRATEGIES = {"random_sample", "time_series", "group_random", "stratified"}
VALID_EVALUATION_MODES = {"holdout", "cross_validation"}


def _stable_path_key(path: str) -> str:
    """Normalize paths before sorting so split generation is reproducible."""
    return os.path.normcase(os.path.normpath(str(path)))


def _sample_identity_key(path: str) -> str:
    """Sort by sample identity so video-only and multimodal scans split identically."""
    normalized = str(path).replace('\\', '/').lower()
    stem = Path(normalized).stem.replace("_audio_", "_sample_").replace("_video_", "_sample_")
    parts = list(Path(normalized).parts)
    if len(parts) >= 4:
        return "/".join([parts[-4], parts[-3], parts[-2], stem])
    return stem


class BaseDataSplitter(ABC):
    """
    Abstract base class defining the dataset splitting interface and shared utilities.
    Handles common split export logic (CSV, JSONL, summary JSON) for all splitter implementations.
    """
    def __init__(self, config: SplitterConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset_path
        self.seed = getattr(config, "seed", getattr(config, "random_seed", 42))
        self.test_sample_per_class = getattr(config, "test_sample_per_class", 700)
        self.save_results = getattr(config, "save_results", True)
        self.include_video = getattr(config, "include_video", True)
        raw_strategy = getattr(config, "split_strategy", "random_sample")
        self.split_strategy = "random_sample" if raw_strategy == "stratified" else raw_strategy
        self.evaluation_mode = getattr(config, "evaluation_mode", "holdout")
        self.num_folds = int(getattr(config, "num_folds", 5))
        self.fold_index = getattr(config, "fold_index", None)
        self.cv_val_ratio = float(getattr(config, "cv_val_ratio", 0.2))
        if self.split_strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"Invalid split_strategy='{self.split_strategy}'. "
                f"Expected one of {sorted(VALID_SPLIT_STRATEGIES)}."
            )
        if self.evaluation_mode not in VALID_EVALUATION_MODES:
            raise ValueError(
                f"Invalid evaluation_mode='{self.evaluation_mode}'. "
                f"Expected one of {sorted(VALID_EVALUATION_MODES)}."
            )
        if self.evaluation_mode == "cross_validation":
            if self.fold_index is None:
                raise ValueError("fold_index must be set when evaluation_mode='cross_validation'.")
            if not 0 <= int(self.fold_index) < self.num_folds:
                raise ValueError(f"fold_index={self.fold_index} is outside [0, {self.num_folds}).")

        # Automatically resolve the optional audio directory path.
        audio_dir = os.path.join(self.dataset_path, 'audio')
        if os.path.isdir(audio_dir):
            self.audio_path = audio_dir
        elif Path(self.dataset_path).name.lower() == 'audio':
            self.audio_path = self.dataset_path
        else:
            self.audio_path = None

        # Automatically resolve the video directory path (used as the root for sample splitting).
        video_dir = os.path.join(self.dataset_path, 'video')
        if os.path.isdir(video_dir):
            self.video_path = video_dir
            self.video_exists = True
        elif Path(self.dataset_path).name.lower() == 'video':
            self.video_path = self.dataset_path
            self.video_exists = True
        else:
            self.video_path = None
            self.video_exists = False
            raise FileNotFoundError(
                f"Multimodal dataset splitting requires a 'video' directory inside dataset_path or dataset_path pointing to a video root: '{self.dataset_path}'"
            )

    @abstractmethod
    def get_file_list(self, split_name: str) -> List[str]:
        """
        Scan the directory and return a list of file paths.
        Must be implemented by subclasses.

        Args:
            split_name (str): The feeding intensity class name to scan.
                Example: 'strong', 'medium', 'weak', 'none'

        Returns:
            List[str]: A list of absolute paths of all scanned files.
        """
        pass

    @abstractmethod
    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        """
        Split the file lists into Train, Test, and Validation sets.
        Must be implemented by subclasses.

        Args:
            None (parameters are derived from class attributes).

        Returns:
            Tuple[List[List], List[List], List[List]]: A tuple containing three lists (Train, Test, Val).
        """
        pass

    @abstractmethod
    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """
        Format raw data lists into standardized dictionaries for file export.
        Must be implemented by subclasses.

        Args:
            split_name (str): The dataset split name ('train', 'test', or 'val').
            data_list (List[List]): A list of label-paired data entries in [path, numeric_label] format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing detailed metadata for each sample.
        """
        pass

    @staticmethod
    def _write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
        """Write a list of data samples to a JSONL-formatted file."""
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(samples: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
        """Write a list of data samples to a CSV-formatted file."""
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    @staticmethod
    def _primary_path_from_entry(item: List[Any]) -> str:
        """Return the path used for temporal sorting and sample identity."""
        if len(item) >= 3:
            return str(item[1] or item[0])
        return str(item[0])

    @staticmethod
    def _label_from_entry(item: List[Any]) -> int:
        return int(item[-1])

    @staticmethod
    def _extract_ints(value: str) -> Tuple[int, ...]:
        return tuple(int(number) for number in re.findall(r"\d+", str(value)))

    @classmethod
    def _temporal_path_key(cls, path: str) -> Tuple[Any, ...]:
        """Sort by date, AM/PM session, session number, then sample id."""
        normalized = str(path).replace('\\', '/')
        parts = Path(normalized).parts
        date_part = parts[-4] if len(parts) >= 4 else ""
        session_part = parts[-3] if len(parts) >= 3 else ""
        stem = Path(normalized).stem

        date_numbers = cls._extract_ints(date_part)
        date_key = date_numbers[:3] if len(date_numbers) >= 3 else date_numbers

        session_upper = session_part.upper()
        if session_upper.startswith("AM"):
            day_half = 0
        elif session_upper.startswith("PM"):
            day_half = 1
        else:
            day_half = 2

        session_numbers = cls._extract_ints(session_part)
        session_value = session_numbers[0] if session_numbers else -1
        sample_numbers = cls._extract_ints(stem)
        sample_value = sample_numbers[-1] if sample_numbers else -1

        return (date_key, day_half, session_value, sample_value, stem)

    def _entry_sample_key(self, item: List[Any]) -> str:
        return _sample_identity_key(self._primary_path_from_entry(item))

    def _entry_temporal_key(self, item: List[Any]) -> Tuple[Any, ...]:
        return self._temporal_path_key(self._primary_path_from_entry(item))

    def _entry_group_key(self, item: List[Any]) -> str:
        """Group all samples from one collection session to avoid same-session leakage."""
        normalized = self._primary_path_from_entry(item).replace('\\', '/')
        parts = Path(normalized).parts
        date_part = parts[-4] if len(parts) >= 4 else ""
        session_part = parts[-3] if len(parts) >= 3 else ""
        return f"{date_part.lower()}/{session_part.lower()}"

    def _count_labels(self, data_list: List[List]) -> Counter:
        counts = Counter()
        for item in data_list:
            counts[self._label_from_entry(item)] += 1
        return counts

    def _group_entries(self, entries: List[List]) -> Dict[str, List[List]]:
        groups: Dict[str, List[List]] = {}
        for item in entries:
            groups.setdefault(self._entry_group_key(item), []).append(item)
        return groups

    def _group_label_counts(self, group_entries: Dict[str, List[List]], group_keys: List[str]) -> np.ndarray:
        return np.array(
            [
                [sum(1 for item in group_entries[group_key] if self._label_from_entry(item) == label) for group_key in group_keys]
                for label in LABEL_ORDER
            ],
            dtype=float,
        )

    def _solve_exact_group_holdout(self, group_entries: Dict[str, List[List]]) -> Tuple[Set[str], Set[str]]:
        """
        Select disjoint date_session groups for val/test with exact per-class quotas.
        This keeps every sample from one collection session in the same split.
        """
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError as exc:
            raise ImportError(
                "split_strategy='group_random' requires scipy.optimize.milp. "
                "Install scipy or project requirements before splitting."
            ) from exc

        group_keys = sorted(group_entries)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(group_keys)
        group_count = len(group_keys)
        class_group_counts = self._group_label_counts(group_entries, group_keys)
        target = float(self.test_sample_per_class)

        objective = np.zeros(2 * group_count)

        constraints = []
        lower_bounds = []
        upper_bounds = []

        for row in class_group_counts:
            coeff = np.zeros(2 * group_count)
            coeff[:group_count] = row
            constraints.append(coeff)
            lower_bounds.append(target)
            upper_bounds.append(target)

        for row in class_group_counts:
            coeff = np.zeros(2 * group_count)
            coeff[group_count:] = row
            constraints.append(coeff)
            lower_bounds.append(target)
            upper_bounds.append(target)

        for idx in range(group_count):
            coeff = np.zeros(2 * group_count)
            coeff[idx] = 1
            coeff[group_count + idx] = 1
            constraints.append(coeff)
            lower_bounds.append(0)
            upper_bounds.append(1)

        linear_constraint = LinearConstraint(
            np.vstack(constraints),
            np.array(lower_bounds),
            np.array(upper_bounds),
        )
        result = milp(
            c=objective,
            integrality=np.ones(2 * group_count),
            bounds=Bounds(np.zeros(2 * group_count), np.ones(2 * group_count)),
            constraints=linear_constraint,
            options={"time_limit": 120, "mip_rel_gap": 0},
        )

        if not result.success:
            raise ValueError(
                "Could not find an exact group_random holdout split with "
                f"{self.test_sample_per_class} samples per class for both val and test. "
                f"MILP status={result.status}, message={result.message}"
            )

        selected = np.rint(result.x).astype(int)
        val_groups = {group_keys[idx] for idx in np.where(selected[:group_count] == 1)[0]}
        test_groups = {group_keys[idx] for idx in np.where(selected[group_count:] == 1)[0]}
        return val_groups, test_groups

    def _split_entries_group_holdout(self, all_entries: List[List]) -> Tuple[List[List], List[List], List[List]]:
        group_entries = self._group_entries(all_entries)
        val_groups, test_groups = self._solve_exact_group_holdout(group_entries)

        train_dict = [item for item in all_entries if self._entry_group_key(item) not in val_groups and self._entry_group_key(item) not in test_groups]
        test_dict = [item for item in all_entries if self._entry_group_key(item) in test_groups]
        val_dict = [item for item in all_entries if self._entry_group_key(item) in val_groups]

        logger.info(
            f"group_random holdout selected groups: train={len(set(self._entry_group_key(item) for item in train_dict))}, "
            f"test={len(test_groups)}, val={len(val_groups)}"
        )
        self._validate_split_policy(train_dict, test_dict, val_dict)
        return train_dict, test_dict, val_dict

    def _group_random_cross_validation_expected_splits(self, all_entries: List[List]) -> Tuple[set, set, set]:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError as exc:
            raise ImportError(
                "split_strategy='group_random' with cross_validation requires scikit-learn."
            ) from exc

        ordered_entries = sorted(all_entries, key=self._entry_sample_key)
        labels = np.array([self._label_from_entry(item) for item in ordered_entries])
        groups = np.array([self._entry_group_key(item) for item in ordered_entries])
        indexes = np.arange(len(ordered_entries))

        outer = StratifiedGroupKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        outer_splits = list(outer.split(indexes, labels, groups))
        dev_idx, test_idx = outer_splits[int(self.fold_index)]

        dev_entries = [ordered_entries[int(idx)] for idx in dev_idx]
        dev_labels = np.array([self._label_from_entry(item) for item in dev_entries])
        dev_groups = np.array([self._entry_group_key(item) for item in dev_entries])
        dev_indexes = np.arange(len(dev_entries))

        val_n_splits = max(2, int(round(1.0 / self.cv_val_ratio)))
        val_n_splits = min(val_n_splits, len(set(dev_groups)))
        inner = StratifiedGroupKFold(
            n_splits=val_n_splits,
            shuffle=True,
            random_state=self.seed + 10000 + int(self.fold_index),
        )
        inner_splits = list(inner.split(dev_indexes, dev_labels, dev_groups))
        train_dev_idx, val_dev_idx = inner_splits[int(self.fold_index) % len(inner_splits)]

        train_entries = [dev_entries[int(idx)] for idx in train_dev_idx]
        val_entries = [dev_entries[int(idx)] for idx in val_dev_idx]
        test_entries = [ordered_entries[int(idx)] for idx in test_idx]

        return (
            {self._entry_sample_key(item) for item in train_entries},
            {self._entry_sample_key(item) for item in test_entries},
            {self._entry_sample_key(item) for item in val_entries},
        )

    @staticmethod
    def _sample_key_set(data_list: List[List], sample_key_fn) -> set:
        return {sample_key_fn(item) for item in data_list}

    def _time_series_expected_splits(self, all_entries: List[List]) -> Tuple[set, set, set]:
        """Build expected sample-key sets for the time-series split using val/test quotas."""
        expected_train = set()
        expected_test = set()
        expected_val = set()

        for label in LABEL_ORDER:
            class_entries = [item for item in all_entries if self._label_from_entry(item) == label]
            class_entries = sorted(class_entries, key=self._entry_temporal_key)
            required = 2 * self.test_sample_per_class
            if len(class_entries) < required:
                raise ValueError(
                    f"Class '{LABEL_TO_CLASS[label]}' has only {len(class_entries)} samples, "
                    f"but at least {required} are required for validation and test splits."
                )

            train_entries = class_entries[:-required]
            val_entries = class_entries[-required:-self.test_sample_per_class]
            test_entries = class_entries[-self.test_sample_per_class:]

            expected_train.update(self._entry_sample_key(item) for item in train_entries)
            expected_val.update(self._entry_sample_key(item) for item in val_entries)
            expected_test.update(self._entry_sample_key(item) for item in test_entries)

        return expected_train, expected_test, expected_val

    def _ordered_class_entries(self, entries: List[List], label: int, seed_offset: int = 0) -> List[List]:
        """Return class entries in deterministic order for the configured split strategy."""
        class_entries = [item for item in entries if self._label_from_entry(item) == label]
        if self.split_strategy == "time_series":
            return sorted(class_entries, key=self._entry_temporal_key)

        class_entries = sorted(class_entries, key=self._entry_sample_key)
        rng = np.random.RandomState(self.seed + seed_offset + label)
        rng.shuffle(class_entries)
        return class_entries

    @staticmethod
    def _split_into_folds(items: List[List], num_folds: int) -> List[List[List]]:
        """Split a deterministic list into nearly equal contiguous folds."""
        indexes = np.array_split(np.arange(len(items)), num_folds)
        return [[items[int(idx)] for idx in fold_indexes] for fold_indexes in indexes]

    def _cross_validation_expected_splits(self, all_entries: List[List]) -> Tuple[set, set, set]:
        """Build expected sample-key sets for one stratified outer CV fold."""
        if self.split_strategy == "group_random":
            return self._group_random_cross_validation_expected_splits(all_entries)

        fold_index = int(self.fold_index)
        train_entries: List[List] = []
        test_entries: List[List] = []
        val_entries: List[List] = []

        for label in LABEL_ORDER:
            class_entries = self._ordered_class_entries(entries=all_entries, label=label, seed_offset=0)
            outer_folds = self._split_into_folds(class_entries, self.num_folds)
            test_class_entries = list(outer_folds[fold_index])
            dev_class_entries = [
                item
                for idx, fold_entries in enumerate(outer_folds)
                if idx != fold_index
                for item in fold_entries
            ]

            if self.split_strategy == "time_series":
                dev_class_entries = sorted(dev_class_entries, key=self._entry_temporal_key)
            else:
                dev_class_entries = sorted(dev_class_entries, key=self._entry_sample_key)
                rng = np.random.RandomState(self.seed + 10000 + fold_index + label)
                rng.shuffle(dev_class_entries)

            val_count = int(round(len(dev_class_entries) * self.cv_val_ratio))
            if val_count <= 0 or val_count >= len(dev_class_entries):
                raise ValueError(
                    f"Invalid cv_val_ratio={self.cv_val_ratio} for class '{LABEL_TO_CLASS[label]}' "
                    f"with {len(dev_class_entries)} development samples."
                )

            if self.split_strategy == "time_series":
                train_class_entries = dev_class_entries[:-val_count]
                val_class_entries = dev_class_entries[-val_count:]
            else:
                val_class_entries = dev_class_entries[:val_count]
                train_class_entries = dev_class_entries[val_count:]

            train_entries.extend(train_class_entries)
            val_entries.extend(val_class_entries)
            test_entries.extend(test_class_entries)

        return (
            {self._entry_sample_key(item) for item in train_entries},
            {self._entry_sample_key(item) for item in test_entries},
            {self._entry_sample_key(item) for item in val_entries},
        )

    def _split_entries_cross_validation(self, all_entries: List[List]) -> Tuple[List[List], List[List], List[List]]:
        expected_train, expected_test, expected_val = self._cross_validation_expected_splits(all_entries)
        train_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_train]
        test_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_test]
        val_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_val]

        self._validate_split_policy(train_dict, test_dict, val_dict)
        return train_dict, test_dict, val_dict

    def _validate_split_policy(self, train_dict: List[List], test_dict: List[List], val_dict: List[List]) -> None:
        """Validate split consistency for the selected strategy."""
        splits = {"train": train_dict, "test": test_dict, "val": val_dict}

        seen_samples: Dict[str, str] = {}
        for split_name, data_list in splits.items():
            for item in data_list:
                sample_key = self._entry_sample_key(item)
                if sample_key in seen_samples:
                    raise ValueError(
                        f"Duplicate sample detected across splits: '{sample_key}' "
                        f"appears in both {seen_samples[sample_key]} and {split_name}."
                    )
                seen_samples[sample_key] = split_name

        counts_by_split = {name: self._count_labels(data) for name, data in splits.items()}
        total_counts = Counter()
        for counts in counts_by_split.values():
            total_counts.update(counts)

        if self.split_strategy == "group_random":
            seen_groups: Dict[str, str] = {}
            for split_name, data_list in splits.items():
                for item in data_list:
                    group_key = self._entry_group_key(item)
                    if group_key in seen_groups and seen_groups[group_key] != split_name:
                        raise ValueError(
                            f"Leakage detected: group '{group_key}' appears in both "
                            f"{seen_groups[group_key]} and {split_name}."
                        )
                    seen_groups[group_key] = split_name

        if self.evaluation_mode == "holdout":
            for split_name in ["test", "val"]:
                for label in LABEL_ORDER:
                    actual = counts_by_split[split_name][label]
                    if actual != self.test_sample_per_class:
                        raise ValueError(
                            f"{split_name} split class '{LABEL_TO_CLASS[label]}' has {actual} samples; "
                            f"expected exactly {self.test_sample_per_class}."
                        )

            for label in LABEL_ORDER:
                expected_train = total_counts[label] - (2 * self.test_sample_per_class)
                actual_train = counts_by_split["train"][label]
                if actual_train != expected_train:
                    raise ValueError(
                        f"train split class '{LABEL_TO_CLASS[label]}' has {actual_train} samples; "
                        f"expected exactly {expected_train}."
                    )

        if self.evaluation_mode == "holdout" and self.split_strategy == "time_series":
            expected_train, expected_test, expected_val = self._time_series_expected_splits(
                train_dict + test_dict + val_dict
            )
            actual_train = self._sample_key_set(train_dict, self._entry_sample_key)
            actual_test = self._sample_key_set(test_dict, self._entry_sample_key)
            actual_val = self._sample_key_set(val_dict, self._entry_sample_key)
            if actual_train != expected_train or actual_test != expected_test or actual_val != expected_val:
                raise ValueError("Existing split files do not match the configured time_series split strategy.")

        if self.evaluation_mode == "cross_validation":
            expected_train, expected_test, expected_val = self._cross_validation_expected_splits(
                train_dict + test_dict + val_dict
            )
            actual_train = self._sample_key_set(train_dict, self._entry_sample_key)
            actual_test = self._sample_key_set(test_dict, self._entry_sample_key)
            actual_val = self._sample_key_set(val_dict, self._entry_sample_key)
            if actual_train != expected_train or actual_test != expected_test or actual_val != expected_val:
                raise ValueError("Existing split files do not match the configured cross-validation fold.")

        logger.info(f"Split validation passed for mode='{self.evaluation_mode}', strategy='{self.split_strategy}':")
        for split_name in ["train", "test", "val"]:
            counts = counts_by_split[split_name]
            logger.info(
                f"  - {split_name}: "
                f"none={counts[0]}, strong={counts[1]}, medium={counts[2]}, weak={counts[3]}"
            )

    def _save_splits(self, train_dict: List[List], test_dict: List[List], val_dict: List[List], output_dir: Path) -> None:
        """
        Save dataset splits to CSV, JSONL, and a summary JSON file (shared implementation).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_dict,
            "test": test_dict,
            "val": val_dict
        }

        formatted_splits: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Format data and save CSV/JSONL for each split using the subclass implementation
        for split_name, data_list in splits.items():
            samples = self._format_samples(split_name, data_list)
            formatted_splits[split_name] = samples

            # Proceed to write output files
            fieldnames = ["video_path", "audio_path", "label", "class_name", "date", "session", "sample_id"]
            self._write_jsonl(samples, output_path / f"{split_name}.jsonl")
            self._write_csv(samples, output_path / f"{split_name}.csv", fieldnames=fieldnames)
            logger.info(f"Successfully saved split files to {output_path}")

        # 2. Generate and save the summary.json file
        summary: Dict[str, Any] = {
            "label_map": {
                "none": 0,
                "strong": 1,
                "medium": 2,
                "weak": 3
              },
            "splits": {}
        }
        
        for split_name, samples in formatted_splits.items():
            counts = {"none": 0, "strong": 0, "medium": 0, "weak": 0}
            for s in samples:
                counts[str(s["class_name"])] += 1
            summary["splits"][split_name] = {
                "total": len(samples),
                "by_class": counts
            }

        # Append configuration metadata
        summary.update({
            "dataset_root": self.dataset_path,
            "output_dir": str(output_path),
            "seed": self.seed,
            "test_sample_per_class": self.test_sample_per_class,
            "split_strategy": self.split_strategy,
            "evaluation_mode": self.evaluation_mode,
            "num_folds": self.num_folds,
            "fold_index": self.fold_index,
            "cv_val_ratio": self.cv_val_ratio
        })

        with (output_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved split summary file to {output_path}")


class FishDataSplitter(BaseDataSplitter):
    """
    Concrete implementation for multimodal fish data splitting (Audio + Video).
    Preserves the original file scanning and splitting logic from U-FFIA's fish_audio_dataset.py.
    Automatically links video to audio and applies path-fixing fallback for cross-machine transfers.
    """
    # noinspection DuplicatedCode
    def get_file_list(self, split_name: str) -> List[str]:
        path = self.video_path
        video: List[str] = []
        l1 = sorted(os.listdir(path), key=_stable_path_key)
        for folder_name in l1:
            folder_path = os.path.join(path, folder_name)
            if folder_name == 'splits' or not os.path.isdir(folder_path):
                continue
            l2 = sorted(os.listdir(folder_path), key=_stable_path_key)
            for session_folder in l2:
                session_path = os.path.join(folder_path, session_folder)
                if not os.path.isdir(session_path):
                    continue
                video_dir = os.path.join(session_path, split_name, '*.mp4')
                video.extend(sorted(glob.glob(video_dir), key=_sample_identity_key))
        return video

    def _resolve_audio_path(self, video_path: str) -> str:
        """Automatically resolve the corresponding audio path from a video path when available."""
        normalized_video = str(video_path).replace('\\', '/')
        parts = Path(normalized_video).parts
        parts_list = list(parts)
        if "video" in parts_list:
            idx = parts_list.index("video")
            parts_list[idx] = "audio"

        filename = parts_list[-1]
        if filename.endswith(".mp4"):
            filename = filename[:-4] + ".wav"
        if "_video_" in filename:
            filename = filename.replace("_video_", "_audio_")
        parts_list[-1] = filename

        if parts_list[0].endswith('\\') or parts_list[0].endswith('/'):
            reconstructed_audio = parts_list[0] + os.path.join(*parts_list[1:])
        else:
            reconstructed_audio = os.path.join(*parts_list)

        if os.path.exists(reconstructed_audio):
            return reconstructed_audio
        return ""

    def _resolve_video_path(self, audio_path: str) -> str:
        """Automatically resolve the corresponding video path from an audio path."""
        # Replace backslashes with forward slashes to ensure cross-platform compatibility
        normalized_audio = str(audio_path).replace('\\', '/')
        parts = Path(normalized_audio).parts
        parts_list = list(parts)
        if "audio" in parts_list:
            idx = parts_list.index("audio")
            parts_list[idx] = "video"
        
        filename = parts_list[-1]
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".mp4"
        if "_audio_" in filename:
            filename = filename.replace("_audio_", "_video_")
        parts_list[-1] = filename
        
        if parts_list[0].endswith('\\') or parts_list[0].endswith('/'):
            reconstructed_video = parts_list[0] + os.path.join(*parts_list[1:])
        else:
            reconstructed_video = os.path.join(*parts_list)

        if os.path.exists(reconstructed_video):
            return reconstructed_video
        return ""

    def _make_multimodal_entry(self, video_path: str, label: int) -> List[Any]:
        """Create one [audio_path, video_path, label] entry with optional audio_path."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Missing video file while creating split: '{video_path}'")
        return [self._resolve_audio_path(video_path), video_path, label]

    def _validate_multimodal_entries(self, split_name: str, data_list: List[List]) -> None:
        """Validate that every split entry has a video file and optional matching audio file."""

        for idx, item in enumerate(data_list):
            if len(item) != 3:
                raise ValueError(
                    f"{split_name} split entry #{idx} must be [audio_path, video_path, label], got: {item}"
                )
            audio_path, video_path, _ = item
            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"{split_name} split entry #{idx} has missing video_path: '{video_path}'"
                )
            expected_audio = self._resolve_audio_path(video_path)
            if audio_path and not os.path.exists(audio_path):
                raise FileNotFoundError(
                    f"{split_name} split entry #{idx} has missing audio_path: '{audio_path}'"
                )
            if audio_path and expected_audio and os.path.normpath(audio_path) != os.path.normpath(expected_audio):
                raise ValueError(
                    f"{split_name} split entry #{idx} has mismatched audio_path.\n"
                    f"  video_path:     {video_path}\n"
                    f"  audio_path:     {audio_path}\n"
                    f"  expected_audio: {expected_audio}"
                )

    def _clear_existing_splits(self, splits_root: Path) -> None:
        dataset_path = Path(self.dataset_path).resolve()
        splits_root = Path(splits_root).resolve()
        allowed_roots = {(dataset_path / 'splits').resolve()}
        if dataset_path.name.lower() in ['audio', 'video']:
            allowed_roots.add((dataset_path.parent / 'splits').resolve())

        if splits_root.name != 'splits' or splits_root not in allowed_roots:
            raise ValueError(f"Refusing to delete unexpected split path: '{splits_root}'")

        if splits_root.exists():
            logger.info(f"Removing existing split files before generating a fresh split: '{splits_root}'")
            shutil.rmtree(splits_root)

    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        # Automatically determine the splits directory. When dataset_path points
        # directly to an audio folder, keep splits inside that folder so the
        # audio root remains self-contained on marimo.
        dataset_path = Path(self.dataset_path)
        local_splits_dir = dataset_path / 'splits'
        parent_splits_dir = dataset_path.parent / 'splits'

        if dataset_path.name in ['audio', 'video'] and parent_splits_dir.exists() and not local_splits_dir.exists():
            base_splits_dir = parent_splits_dir
        else:
            base_splits_dir = local_splits_dir

        if self.evaluation_mode == "cross_validation":
            splits_dir = base_splits_dir / "cv" / f"fold_{int(self.fold_index):02d}"
            if int(self.fold_index) == 0:
                self._clear_existing_splits(base_splits_dir)
        else:
            splits_dir = base_splits_dir
            self._clear_existing_splits(base_splits_dir)


        train_csv = splits_dir / 'train.csv'
        test_csv = splits_dir / 'test.csv'
        val_csv = splits_dir / 'val.csv'

        if splits_dir.exists() and train_csv.exists() and test_csv.exists() and val_csv.exists():
            logger.info("==================================================")
            logger.info(f"Fallback mode enabled: found existing split files at '{splits_dir}'.")
            logger.info("Loading existing dataset splits instead of recomputing them...")
            logger.info(f"Base video search path: '{self.video_path}'")
            logger.info("==================================================")

            train_dict = []
            test_dict = []
            val_dict = []
            need_rewrite_files = False

            try:
                # Helper function to load data from CSV and auto-fix file paths with batched logging
                def load_and_fix_csv(csv_path: Path, split_name: str) -> List[List]:
                    nonlocal need_rewrite_files
                    loaded_data = []
                    fixed_count = 0
                    missing_count = 0
                    audio_updated_count = 0
                    
                    # Variables for logging a few example path corrections
                    video_fix_example = None
                    audio_fix_example = None
                    
                    with csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            original_audio_path = row.get("audio_path", "")
                            raw_video_path = row.get("video_path", "")
                            if not raw_video_path and original_audio_path:
                                raw_video_path = self._resolve_video_path(original_audio_path)
                            label = int(row["label"])
                            
                            # If the video path does not exist, attempt automatic path correction
                            if not os.path.exists(raw_video_path):
                                # Replace backslashes with forward slashes to ensure cross-platform compatibility
                                normalized_video = raw_video_path.replace('\\', '/') if raw_video_path else original_audio_path.replace('\\', '/')
                                parts = Path(normalized_video).parts
                                if len(parts) >= 4:
                                    # Generate multiple possible paths due to different server layouts
                                    candidates = [
                                        os.path.join(self.video_path, parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, 'video', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, 'U_FFIA', 'video', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'video', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'U_FFIA', 'video', parts[-4], parts[-3], parts[-2], parts[-1]),
                                    ]
                                    
                                    fixed_path = None
                                    for cand in candidates:
                                        if os.path.exists(cand):
                                            fixed_path = cand
                                            break
                                            
                                    if fixed_path is not None:
                                        if video_fix_example is None:
                                            video_fix_example = (raw_video_path, fixed_path)
                                        raw_video_path = fixed_path
                                        fixed_count += 1
                                        need_rewrite_files = True
                                    else:
                                        if missing_count < 3:
                                            # Display the primary checked path in warning for reference
                                            primary_checked = candidates[0]
                                            logger.warning(f"{split_name} split: Tried path correction candidates, for example '{primary_checked}', but the file still does not exist on this server.")
                                        missing_count += 1

                            if not os.path.exists(raw_video_path):
                                raise FileNotFoundError(
                                    f"{split_name} split has a missing video file that could not be corrected: '{raw_video_path}'"
                                )
                            
                            audio_path_str = self._resolve_audio_path(raw_video_path)
                            if os.path.normpath(original_audio_path) != os.path.normpath(audio_path_str):
                                if audio_fix_example is None:
                                    audio_fix_example = (original_audio_path, audio_path_str)
                                audio_updated_count += 1
                                need_rewrite_files = True
                            loaded_data.append([audio_path_str, raw_video_path, label])
                    
                    # Print detailed logs
                    if fixed_count > 0:
                        if video_fix_example:
                            logger.info(f"{split_name} split: Successfully auto-corrected {fixed_count} invalid video paths.")
                            logger.info(f"   * Video path correction example:\n     - Old: {video_fix_example[0]}\n     - New: {video_fix_example[1]}")
                    if audio_updated_count > 0:
                        logger.info(f"{split_name} split: Successfully auto-filled {audio_updated_count} newly detected audio paths.")
                        if audio_fix_example:
                            logger.info(f"   * Audio path example: {audio_fix_example[1]}")
                    if missing_count > 0:
                        logger.warning(f"{split_name} split: {missing_count} files do not exist on this machine. Please check the dataset.")
                        
                    return loaded_data

                train_dict = load_and_fix_csv(train_csv, "Train")
                test_dict = load_and_fix_csv(test_csv, "Test")
                val_dict = load_and_fix_csv(val_csv, "Validation")

                self._validate_multimodal_entries("Train", train_dict)
                self._validate_multimodal_entries("Test", test_dict)
                self._validate_multimodal_entries("Validation", val_dict)
                self._validate_split_policy(train_dict, test_dict, val_dict)

                logger.info("Successfully loaded dataset splits from files:")
                logger.info(f"- Train samples: {len(train_dict)}")
                logger.info(f"- Test samples:  {len(test_dict)}")
                logger.info(f"- Val samples:   {len(val_dict)}")
                
                # Print first samples for user verification
                if len(train_dict) > 0:
                    logger.info(f"  * First loaded train sample: {train_dict[0]}")
                if len(test_dict) > 0:
                    logger.info(f"  * First loaded test sample:  {test_dict[0]}")
                if len(val_dict) > 0:
                    logger.info(f"  * First loaded val sample:   {val_dict[0]}")
                
                # If any paths were corrected or new video paths added, overwrite split files on disk to synchronize
                if need_rewrite_files:
                    logger.info("Updating split files on disk to synchronize corrected paths...")
                    self._save_splits(train_dict, test_dict, val_dict, splits_dir)
                    logger.info("Finished synchronizing corrected split files on disk.")
                
                logger.info("==================================================")
                return train_dict, test_dict, val_dict
            except Exception as e:
                logger.warning(f"Failed to load or auto-correct existing splits: {e}. Falling back to standard splitting.")

        logger.info("==================================================")
        logger.info("Starting multimodal dataset splitting...")
        logger.info(f"Dataset root directory: '{self.dataset_path}'")
        logger.info(f"Random seed: {self.seed}")
        logger.info(f"Test/Val samples per class: {self.test_sample_per_class}")
        logger.info(f"Evaluation mode: {self.evaluation_mode}")
        logger.info(f"Split strategy: {self.split_strategy}")
        if self.evaluation_mode == "cross_validation":
            logger.info(f"CV folds: {self.num_folds} | Fold index: {self.fold_index} | CV val ratio: {self.cv_val_ratio}")
        logger.info(f"Save split results: {self.save_results}")
        logger.info("Video paths are required; audio paths are filled only when available.")
        logger.info("==================================================")

        # Scan files and log the counts
        logger.info("Scanning video files for each class...")
        strong_list = self.get_file_list(split_name='strong')
        logger.info(f"Class 'strong': Found {len(strong_list)} files.")
        
        medium_list = self.get_file_list(split_name='medium')
        logger.info(f"Class 'medium': Found {len(medium_list)} files.")
        
        weak_list = self.get_file_list(split_name='weak')
        logger.info(f"Class 'weak': Found {len(weak_list)} files.")
        
        none_list = self.get_file_list(split_name='none')
        logger.info(f"Class 'none': Found {len(none_list)} files.")

        def build_entries() -> List[List]:
            return (
                [self._make_multimodal_entry(video, 1) for video in strong_list] +
                [self._make_multimodal_entry(video, 2) for video in medium_list] +
                [self._make_multimodal_entry(video, 3) for video in weak_list] +
                [self._make_multimodal_entry(video, 0) for video in none_list]
            )

        if self.evaluation_mode == "cross_validation":
            logger.info("Building stratified cross-validation fold split...")
            train_dict, test_dict, val_dict = self._split_entries_cross_validation(build_entries())

            logger.info("==================================================")
            logger.info("Cross-validation dataset splitting completed successfully!")
            logger.info(f"- Train samples: {len(train_dict)}")
            logger.info(f"- Test samples:  {len(test_dict)}")
            logger.info(f"- Val samples:   {len(val_dict)}")
            if len(train_dict) > 0:
                logger.info(f"  * First generated train sample: {train_dict[0]}")
            if len(test_dict) > 0:
                logger.info(f"  * First generated test sample:  {test_dict[0]}")
            if len(val_dict) > 0:
                logger.info(f"  * First generated val sample:   {val_dict[0]}")
            logger.info("==================================================")

            if self.save_results:
                self._save_splits(train_dict, test_dict, val_dict, splits_dir)

            return train_dict, test_dict, val_dict

        if self.split_strategy == "group_random":
            logger.info("Building group_random holdout split with group_key=date_session...")
            train_dict, test_dict, val_dict = self._split_entries_group_holdout(build_entries())

            logger.info("==================================================")
            logger.info("group_random holdout dataset splitting completed successfully!")
            logger.info(f"- Train samples: {len(train_dict)}")
            logger.info(f"- Test samples:  {len(test_dict)}")
            logger.info(f"- Val samples:   {len(val_dict)}")
            if len(train_dict) > 0:
                logger.info(f"  * First generated train sample: {train_dict[0]}")
            if len(test_dict) > 0:
                logger.info(f"  * First generated test sample:  {test_dict[0]}")
            if len(val_dict) > 0:
                logger.info(f"  * First generated val sample:   {val_dict[0]}")
            logger.info("==================================================")

            if self.save_results:
                self._save_splits(train_dict, test_dict, val_dict, splits_dir)

            return train_dict, test_dict, val_dict

        random_state = np.random.RandomState(self.seed)
        if self.split_strategy == "time_series":
            logger.info("Sorting each class chronologically for time_series split...")
            strong_list = sorted(strong_list, key=self._temporal_path_key)
            medium_list = sorted(medium_list, key=self._temporal_path_key)
            weak_list = sorted(weak_list, key=self._temporal_path_key)
            none_list = sorted(none_list, key=self._temporal_path_key)
        else:
            logger.info(f"Shuffling each class list independently with seed={self.seed}...")
            random_state.shuffle(strong_list)
            random_state.shuffle(medium_list)
            random_state.shuffle(weak_list)
            random_state.shuffle(none_list)

        # Perform dataset splitting
        logger.info("Slicing class lists into train, test, and val splits...")
        if self.split_strategy == "time_series":
            holdout_size = 2 * self.test_sample_per_class
            strong_train = strong_list[:-holdout_size]
            medium_train = medium_list[:-holdout_size]
            weak_train = weak_list[:-holdout_size]
            none_train = none_list[:-holdout_size]

            strong_val = strong_list[-holdout_size:-self.test_sample_per_class]
            medium_val = medium_list[-holdout_size:-self.test_sample_per_class]
            weak_val = weak_list[-holdout_size:-self.test_sample_per_class]
            none_val = none_list[-holdout_size:-self.test_sample_per_class]

            strong_test = strong_list[-self.test_sample_per_class:]
            medium_test = medium_list[-self.test_sample_per_class:]
            weak_test = weak_list[-self.test_sample_per_class:]
            none_test = none_list[-self.test_sample_per_class:]
        else:
            strong_test = strong_list[:self.test_sample_per_class]
            medium_test = medium_list[:self.test_sample_per_class]
            weak_test = weak_list[:self.test_sample_per_class]
            none_test = none_list[:self.test_sample_per_class]

            strong_val = strong_list[self.test_sample_per_class:2*self.test_sample_per_class]
            medium_val = medium_list[self.test_sample_per_class:2*self.test_sample_per_class]
            weak_val = weak_list[self.test_sample_per_class:2*self.test_sample_per_class]
            none_val = none_list[self.test_sample_per_class:2*self.test_sample_per_class]

            strong_train = strong_list[2*self.test_sample_per_class:]
            medium_train = medium_list[2*self.test_sample_per_class:]
            weak_train = weak_list[2*self.test_sample_per_class:]
            none_train = none_list[2*self.test_sample_per_class:]

        # Warn if there are insufficient samples for splitting
        for class_name, size in [('strong', len(strong_list)), ('medium', len(medium_list)), ('weak', len(weak_list)), ('none', len(none_list))]:
            req_samples = 2 * self.test_sample_per_class
            if size < req_samples:
                logger.warning(f"Class '{class_name}' has only {size} files, but at least {req_samples} samples are required for Test and Val splits.")

        # Log detailed split sizes per class
        logger.info("Per-class split details:")
        logger.info(f"       - class 'strong': Train={len(strong_train)}, Test={len(strong_test)}, Val={len(strong_val)}")
        logger.info(f"       - class 'medium': Train={len(medium_train)}, Test={len(medium_test)}, Val={len(medium_val)}")
        logger.info(f"       - class 'weak':   Train={len(weak_train)}, Test={len(weak_test)}, Val={len(weak_val)}")
        logger.info(f"       - class 'none':   Train={len(none_train)}, Test={len(none_test)}, Val={len(none_val)}")

        # Map integer labels and merge lists using list comprehension
        logger.info("Mapping integer labels and building dataset lists...")
        train_dict = (
            [self._make_multimodal_entry(video, 1) for video in strong_train] +
            [self._make_multimodal_entry(video, 2) for video in medium_train] +
            [self._make_multimodal_entry(video, 3) for video in weak_train] +
            [self._make_multimodal_entry(video, 0) for video in none_train]
        )
        test_dict = (
            [self._make_multimodal_entry(video, 1) for video in strong_test] +
            [self._make_multimodal_entry(video, 2) for video in medium_test] +
            [self._make_multimodal_entry(video, 3) for video in weak_test] +
            [self._make_multimodal_entry(video, 0) for video in none_test]
        )
        val_dict = (
            [self._make_multimodal_entry(video, 1) for video in strong_val] +
            [self._make_multimodal_entry(video, 2) for video in medium_val] +
            [self._make_multimodal_entry(video, 3) for video in weak_val] +
            [self._make_multimodal_entry(video, 0) for video in none_val]
        )

        if self.split_strategy == "random_sample":
            logger.info("Applying final shuffle to the train split...")
            random_state.shuffle(train_dict)

        self._validate_split_policy(train_dict, test_dict, val_dict)

        logger.info("==================================================")
        logger.info("Dataset splitting completed successfully!")
        logger.info(f"- Train samples: {len(train_dict)}")
        logger.info(f"- Test samples:  {len(test_dict)}")
        logger.info(f"- Val samples:   {len(val_dict)}")
        
        # Print first samples for user verification
        if len(train_dict) > 0:
            logger.info(f"  * First generated train sample: {train_dict[0]}")
        if len(test_dict) > 0:
            logger.info(f"  * First generated test sample:  {test_dict[0]}")
        if len(val_dict) > 0:
            logger.info(f"  * First generated val sample:   {val_dict[0]}")
        logger.info("==================================================")

        # Automatically save results to the splits directory alongside dataset_path if save_results is True
        if self.save_results:
            self._save_splits(train_dict, test_dict, val_dict, splits_dir)

        return train_dict, test_dict, val_dict

    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """Parse sample information into standardized dictionaries for file export."""
        label_to_class = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
        samples = []
        for item in data_list:
            # Support [audio_path, video_path, label] and video-only [video_path, label].
            if len(item) == 3:
                audio_path_str = item[0]
                video_path_str = item[1]
                label = item[2]
            else:
                video_path_str = item[0]
                label = item[1]
                audio_path_str = self._resolve_audio_path(video_path_str)

            # Extract date, session, and sample_id from the absolute path
            primary_path = audio_path_str or video_path_str
            parts = Path(primary_path).parts
            date_part = parts[-4] if len(parts) >= 4 else ""
            session_part = parts[-3] if len(parts) >= 3 else ""
            stem = Path(primary_path).stem
            if "_audio_" in stem:
                sample_id = stem.split("_audio_")[-1]
            elif "_video_" in stem:
                sample_id = stem.split("_video_")[-1]
            else:
                sample_id = stem

            samples.append({
                "video_path": video_path_str,
                "audio_path": str(audio_path_str),
                "label": int(label),
                "class_name": label_to_class.get(label, ""),
                "date": date_part,
                "session": session_part,
                "sample_id": sample_id
            })
        return samples
