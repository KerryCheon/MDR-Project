"""Report helpers for deduplicated exact-search artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CONFIG_COLUMNS = ["global_features", "cluster_0_additions", "cluster_1_additions"]


@dataclass(frozen=True)
class ExactLeaderboard:
    """Unique exact configurations plus their logical candidate aliases."""

    leaderboard: pd.DataFrame
    alias_groups: pd.DataFrame
    winner_aliases: tuple[str, ...]


def _configuration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV empty cells so equivalent empty delta lists compare equally."""
    copied = frame.copy()
    for column in CONFIG_COLUMNS:
        copied[column] = copied[column].fillna("").astype(str)
    return copied


def build_exact_leaderboard(
    results: pd.DataFrame,
    aliases: pd.DataFrame,
    winner_candidate_id: str,
) -> ExactLeaderboard:
    """Consolidate exact-fit aliases without requiring an alias group for the winner.

    ``search_results.csv`` contains only model fits.  ``candidate_aliases.csv``
    records logical candidates that deliberately reused an equivalent fit.  Keeping
    the two files separate makes fit counts honest while still showing why names such
    as the global backbone and the 0/0 delta have the same metrics.
    """
    required_results = {
        "candidate_id",
        "model_kind",
        "completion_status",
        "pooled_r2",
        "pooled_rmse",
        "global_feature_count",
        "cluster_0_feature_count",
        "cluster_1_feature_count",
        "year_2023_r2",
        "year_2024_r2",
        "year_2025_r2",
        *CONFIG_COLUMNS,
    }
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(
            "search_results.csv is missing the deduplicated-search fields: "
            + ", ".join(missing_results)
        )

    exact = results.loc[
        (results["model_kind"] == "exact")
        & (results["completion_status"] == "on_time")
    ].copy()
    if exact.empty:
        raise ValueError("No on-time exact results are available for the report.")
    exact = _configuration_frame(exact)
    exact["winner_first"] = exact["candidate_id"].eq(winner_candidate_id)
    ranked = exact.sort_values(
        ["winner_first", "pooled_r2", "pooled_rmse", "global_feature_count", "candidate_id"],
        ascending=[False, False, True, True, True],
    )
    unique_ranked = ranked.drop_duplicates(CONFIG_COLUMNS, keep="first").copy()

    required_aliases = {"candidate_id", "model_kind", *CONFIG_COLUMNS}
    missing_aliases = sorted(required_aliases - set(aliases.columns))
    if missing_aliases:
        raise ValueError(
            "candidate_aliases.csv is missing required fields: " + ", ".join(missing_aliases)
        )
    exact_aliases = _configuration_frame(
        aliases.loc[aliases["model_kind"] == "exact"].copy()
    )
    logical_candidates = pd.concat(
        [
            ranked.loc[:, ["candidate_id", *CONFIG_COLUMNS]],
            exact_aliases.loc[:, ["candidate_id", *CONFIG_COLUMNS]],
        ],
        ignore_index=True,
    ).drop_duplicates("candidate_id", keep="first")
    grouped = (
        logical_candidates.groupby(CONFIG_COLUMNS, dropna=False, sort=False)["candidate_id"]
        .agg(list)
        .rename("alias_candidates")
        .reset_index()
    )
    grouped["equivalent_evaluations"] = grouped["alias_candidates"].map("; ".join)
    grouped["alias_count"] = grouped["alias_candidates"].map(len)

    annotated = unique_ranked.merge(
        grouped.loc[:, [*CONFIG_COLUMNS, "equivalent_evaluations", "alias_count"]],
        on=CONFIG_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    alias_groups = annotated.loc[
        annotated["alias_count"].fillna(0).astype(int) > 1,
        ["candidate_id", "equivalent_evaluations", "alias_count"],
    ].copy()

    winner_rows = ranked.loc[ranked["candidate_id"].eq(winner_candidate_id), CONFIG_COLUMNS]
    if winner_rows.empty:
        raise ValueError(
            f"Winner {winner_candidate_id!r} is not an on-time exact result."
        )
    winner_key = winner_rows.iloc[0]
    winner_group = grouped.loc[
        (grouped[CONFIG_COLUMNS] == winner_key).all(axis=1), "alias_candidates"
    ]
    if winner_group.empty:
        winner_aliases = (winner_candidate_id,)
    else:
        winner_aliases = tuple(winner_group.iloc[0])

    return ExactLeaderboard(
        leaderboard=annotated.reset_index(drop=True),
        alias_groups=alias_groups.reset_index(drop=True),
        winner_aliases=winner_aliases,
    )
