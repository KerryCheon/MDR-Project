"""Deterministic forward-time and station-time fold construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import stable_json_hash


@dataclass(frozen=True)
class FoldTask:
    family: str
    origin: int
    fold_id: str
    partition_seed: int | None
    learner_seed: int
    held_stations: tuple[str, ...]
    train_index: tuple[int, ...]
    validation_index: tuple[int, ...]

    @property
    def repeat_id(self) -> str:
        partition = "none" if self.partition_seed is None else self.partition_seed
        return f"p{partition}_l{self.learner_seed}"


def station_repeat_pairs(config: Mapping[str, object]) -> list[tuple[int, int]]:
    folds = dict(config["folds"])
    partition_seeds = [int(value) for value in folds["partition_seeds"]]
    learner_seeds = [int(value) for value in folds["station_time_learner_seeds"]]
    base_partition = partition_seeds[0]
    base_learner = learner_seeds[0]
    pairs = [(seed, base_learner) for seed in partition_seeds]
    pairs.extend((base_partition, seed) for seed in learner_seeds)
    return list(dict.fromkeys(pairs))


def balanced_station_partition(
    station_counts: Mapping[str, int],
    *,
    n_partitions: int,
    seed: int,
) -> dict[str, int]:
    """Create a seeded, row-balanced assignment with every group represented."""
    counts = {str(station): int(count) for station, count in station_counts.items()}
    if n_partitions < 2 or len(counts) < n_partitions:
        raise ValueError("station partitions require at least one station per group")
    if any(count <= 0 for count in counts.values()):
        raise ValueError("station counts must all be positive")

    rng = np.random.default_rng(int(seed))
    stations = list(counts)
    random_priority = {station: float(rng.random()) for station in stations}
    ordered = sorted(
        stations,
        key=lambda station: (-counts[station], random_priority[station]),
    )
    group_priority = list(rng.permutation(n_partitions))
    groups: list[list[str]] = [[] for _ in range(n_partitions)]
    loads = [0] * n_partitions
    for station in ordered:
        eligible = sorted(
            range(n_partitions),
            key=lambda group: (
                loads[group],
                len(groups[group]),
                group_priority.index(group),
            ),
        )
        chosen = eligible[0]
        groups[chosen].append(station)
        loads[chosen] += counts[station]
    mapping = {
        station: group for group, members in enumerate(groups) for station in members
    }
    if set(mapping) != set(counts) or set(mapping.values()) != set(range(n_partitions)):
        raise AssertionError("invalid station partition assignment")
    if max(loads) - min(loads) > max(counts.values()):
        raise AssertionError("greedy station partition exceeded its balance bound")
    return mapping


def _station_counts(frame: pd.DataFrame, station_col: str) -> dict[str, int]:
    return {
        str(station): int(count)
        for station, count in frame[station_col].astype(str).value_counts().items()
    }


def build_outer_tasks(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[list[FoldTask], pd.DataFrame, dict[int, dict[str, int]]]:
    data = dict(config["data"])
    folds = dict(config["folds"])
    station_col = str(data["station_col"])
    origins = [int(value) for value in folds["outer_origins"]]
    n_partitions = int(folds["station_partitions"])
    minimum_train = int(folds["minimum_train_rows"])
    minimum_validation = int(folds["minimum_validation_rows"])
    year_values = frame["_year"].to_numpy(dtype=int)
    station_values = frame[station_col].astype(str).to_numpy()
    all_stations = sorted(np.unique(station_values).tolist())

    mappings = {
        int(seed): balanced_station_partition(
            _station_counts(frame, station_col),
            n_partitions=n_partitions,
            seed=int(seed),
        )
        for seed in folds["partition_seeds"]
    }
    tasks: list[FoldTask] = []
    coverage_rows = []
    for origin in origins:
        train_all = np.flatnonzero(year_values < origin)
        validate_all = np.flatnonzero(year_values == origin)
        if len(train_all) < minimum_train or len(validate_all) < minimum_validation:
            raise ValueError(f"origin {origin} has insufficient forward-time rows")
        for learner_seed in folds["forward_time_learner_seeds"]:
            tasks.append(
                FoldTask(
                    family="forward_time",
                    origin=origin,
                    fold_id=f"forward_{origin}_l{int(learner_seed)}",
                    partition_seed=None,
                    learner_seed=int(learner_seed),
                    held_stations=(),
                    train_index=tuple(int(value) for value in train_all),
                    validation_index=tuple(int(value) for value in validate_all),
                )
            )
        for partition_seed, learner_seed in station_repeat_pairs(config):
            mapping = mappings[partition_seed]
            for group in range(n_partitions):
                assigned = tuple(
                    sorted(station for station, value in mapping.items() if value == group)
                )
                held = np.isin(station_values, np.asarray(assigned, dtype=object))
                train_index = np.flatnonzero((year_values < origin) & ~held)
                validation_index = np.flatnonzero((year_values == origin) & held)
                observed = tuple(sorted(np.unique(station_values[validation_index]).tolist()))
                coverage_rows.append(
                    {
                        "origin": origin,
                        "partition_seed": partition_seed,
                        "learner_seed": learner_seed,
                        "group": group,
                        "assigned_stations": "|".join(assigned),
                        "observed_stations": "|".join(observed),
                        "assigned_station_count": len(assigned),
                        "observed_station_count": len(observed),
                        "train_rows": len(train_index),
                        "validation_rows": len(validation_index),
                    }
                )
                if not observed or set(observed) != set(assigned):
                    raise ValueError(
                        f"zero-observation station-year fold at origin {origin}, "
                        f"partition {partition_seed}, group {group}: "
                        f"assigned={assigned}, observed={observed}"
                    )
                if len(train_index) < minimum_train or len(validation_index) < minimum_validation:
                    raise ValueError(
                        f"insufficient station-time rows for origin {origin}, group {group}"
                    )
                if set(station_values[train_index]).intersection(assigned):
                    raise AssertionError("held stations leaked into outer training")
                tasks.append(
                    FoldTask(
                        family="station_time",
                        origin=origin,
                        fold_id=(
                            f"station_{origin}_g{group}_p{partition_seed}_l{learner_seed}"
                        ),
                        partition_seed=partition_seed,
                        learner_seed=learner_seed,
                        held_stations=assigned,
                        train_index=tuple(int(value) for value in train_index),
                        validation_index=tuple(int(value) for value in validation_index),
                    )
                )
    coverage = pd.DataFrame(coverage_rows)
    covered = set(
        station
        for value in coverage["observed_stations"]
        for station in str(value).split("|")
        if station
    )
    if covered != set(all_stations):
        raise ValueError(f"incomplete station coverage: {covered} != {set(all_stations)}")
    return tasks, coverage, mappings


def build_inner_folds(
    outer_training: pd.DataFrame,
    config: Mapping[str, object],
    *,
    family: str,
    partition_seed: int,
) -> list[FoldTask]:
    """Build up to the last two causal ranking years inside outer training."""
    data = dict(config["data"])
    folds = dict(config["folds"])
    station_col = str(data["station_col"])
    years = outer_training["_year"].to_numpy(dtype=int)
    stations = outer_training[station_col].astype(str).to_numpy()
    unique_years = sorted(np.unique(years).tolist())
    minimum_prior = int(folds["minimum_prior_years"])
    minimum_train = int(folds["minimum_train_rows"])
    minimum_validation = int(folds["minimum_validation_rows"])
    eligible = unique_years[minimum_prior:]
    eligible = eligible[-int(folds["inner_max_validation_years"]):]
    if not eligible:
        raise ValueError("outer task has no eligible causal inner ranking year")
    tasks: list[FoldTask] = []
    if family == "forward_time":
        for year in eligible:
            train = np.flatnonzero(years < year)
            validation = np.flatnonzero(years == year)
            if len(train) < minimum_train or len(validation) < minimum_validation:
                raise ValueError(
                    f"insufficient inner forward-time rows for year {year}: "
                    f"train={len(train)}, validation={len(validation)}"
                )
            tasks.append(
                FoldTask(
                    family=family,
                    origin=int(year),
                    fold_id=f"inner_forward_{year}",
                    partition_seed=None,
                    learner_seed=42,
                    held_stations=(),
                    train_index=tuple(int(value) for value in train),
                    validation_index=tuple(int(value) for value in validation),
                )
            )
    elif family == "station_time":
        mapping = balanced_station_partition(
            _station_counts(outer_training, station_col),
            n_partitions=int(folds["station_partitions"]),
            seed=int(partition_seed),
        )
        for year in eligible:
            for group in range(int(folds["station_partitions"])):
                held_stations = tuple(
                    sorted(station for station, value in mapping.items() if value == group)
                )
                held = np.isin(stations, np.asarray(held_stations, dtype=object))
                train = np.flatnonzero((years < year) & ~held)
                validation = np.flatnonzero((years == year) & held)
                if not len(validation):
                    raise ValueError(
                        f"zero-observation inner fold for {year}, {held_stations}"
                    )
                if len(train) < minimum_train or len(validation) < minimum_validation:
                    raise ValueError(
                        f"insufficient inner station-time rows for year {year}, "
                        f"group {group}: train={len(train)}, "
                        f"validation={len(validation)}"
                    )
                if set(stations[train]).intersection(held_stations):
                    raise AssertionError("held station leaked into candidate generation")
                tasks.append(
                    FoldTask(
                        family=family,
                        origin=int(year),
                        fold_id=f"inner_station_{year}_g{group}_p{partition_seed}",
                        partition_seed=int(partition_seed),
                        learner_seed=42,
                        held_stations=held_stations,
                        train_index=tuple(int(value) for value in train),
                        validation_index=tuple(int(value) for value in validation),
                    )
                )
    else:
        raise ValueError(f"unknown inner fold family: {family}")
    return tasks


def task_manifest_rows(frame: pd.DataFrame, tasks: Iterable[FoldTask]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        train_keys = frame.iloc[list(task.train_index)]["_row_key"].tolist()
        validation_keys = frame.iloc[list(task.validation_index)]["_row_key"].tolist()
        rows.append(
            {
                "fold_family": task.family,
                "outer_origin": task.origin,
                "fold_id": task.fold_id,
                "station_partition_seed": task.partition_seed,
                "learner_seed": task.learner_seed,
                "held_stations": "|".join(task.held_stations),
                "train_rows": len(train_keys),
                "validation_rows": len(validation_keys),
                "train_row_keys_sha256": stable_json_hash(train_keys),
                "validation_row_keys_sha256": stable_json_hash(validation_keys),
                "latest_training_year": int(
                    frame.iloc[list(task.train_index)]["_year"].max()
                ),
                "validation_year_min": int(
                    frame.iloc[list(task.validation_index)]["_year"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def assert_train_before_origin(frame: pd.DataFrame, task: FoldTask) -> None:
    train_years = frame.iloc[list(task.train_index)]["_year"]
    validation_years = frame.iloc[list(task.validation_index)]["_year"]
    if not (train_years < task.origin).all():
        raise AssertionError(f"training leakage in {task.fold_id}")
    if not (validation_years == task.origin).all():
        raise AssertionError(f"validation-year mismatch in {task.fold_id}")
