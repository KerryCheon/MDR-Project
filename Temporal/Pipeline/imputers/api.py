# Jakob Balkovec
# Nov 16th 2025
# voting.py

from MDR.Temporal.Pipeline.imputers.voting import VotingImputer
from MDR.Temporal.Pipeline.imputers.interpolation import LinearInterpolationImputer
from MDR.Temporal.Pipeline.imputers.fbfill import ForwardBackwardImputer
from MDR.Temporal.Pipeline.imputers.smoothing import RollingMeanImputer
from MDR.Temporal.Pipeline.imputers.climatology import ClimatologyImputer
from MDR.Temporal.Pipeline.imputers.linear_model import LinearModelImputer
from MDR.Temporal.Pipeline.imputers.xgb_model import XGBImputer

from MDR.Temporal.Pipeline.utils.config import load_config
from MDR.Temporal.Pipeline.utils.logger import get_logger

from MDR.Temporal.Pipeline.records.daily_records import DailyRecordBuilder

from MDR.Temporal.Pipeline.utils.imputer_utils import (
    _run_diagnostics,
    _validate_inputs,
    _prepare_dataframe,
    _apply_ndvi_lockout,
    _should_skip_interpolation,
    _run_ensemble,
    _apply_postprocessing,
    _apply_gap_confidence,
    attach_gap_metadata,
    compute_all_gap_lengths,
    bucket_gap_statistics,
    compute_confidence_vs_gap,
)

from MDR.Temporal.Pipeline.validation.validator import (
    ValidationRunner,
    attach_gap_metadata,
    compute_all_gap_lengths,
    bucket_gap_statistics,
    compute_confidence_vs_gap,
)


def transform_with_ensemble(df, col, return_diag=False, diag_path=None, auto_validate=False):
    # pre:  df has 'date' and col
    # post: returns df with imputed column and metadata; optionally diagnostics
    # desc: main function to run ensemble imputation on a dataframe column
    # note: adds columns ->
    #       col + "_interp"      : imputed values
    #       col + "_conf"        : confidence scores
    #       col + "_gap_length"  : gap lengths in days
    #       col + "_gap_norm"    : normalized gap lengths

    logger = get_logger().getChild(f"imputer.ensemble.{col}")
    logger.debug(f"starting ensemble transform for column '{col}'")

    cfg = load_config()

    _validate_inputs(df, col, logger)

    df = _prepare_dataframe(df, logger)
    dates = df["date"]
    values = df[col]

    df_original = df.copy()

    df = _apply_ndvi_lockout(df, col, cfg, logger)
    ndvi_lock = df["_ndvi_lock"].values

    if _should_skip_interpolation(col, cfg, logger):
        df[col + "_interp"] = df[col].values
        df[col + "_conf"] = (1.0 * (~df[col].isna())).astype(float).values
        df[col + "_gap_length"] = 0
        df[col + "_gap_norm"] = 0
        logger.debug("high-frequency feature bypass complete")
        return df

    filled, conf, voter = _run_ensemble(df, col, dates, values, logger)

    builder = DailyRecordBuilder(df, col)
    records, gap_lengths = builder.make_records(
        dates, filled, conf, excluded_mask=ndvi_lock
    )
    logger.debug(f"DailyRecordBuilder created {len(records)} records")

    # POSTPROCESSING
    df = _apply_postprocessing(
        df, col, filled, conf, ndvi_lock, gap_lengths, logger
    )

    tau_gap = cfg.get("imputer", {}).get("tau_gap", 10)

    df[col + "_conf"] = _apply_gap_confidence(
        df[col + "_conf"], gap_lengths, logger, tau_gap
    )
    df[col + "_conf_norm"] = df[col + "_conf"] / df[col + "_conf"].max()
    # POSTPROCESSING

    max_gap = gap_lengths.max() if gap_lengths.max() > 0 else 1.0
    df[col + "_gap_norm"] = gap_lengths / max_gap

    logger.debug(
        f"completed ensemble transform for '{col}' with coverage={filled.notna().mean():.3f}"
    )

    if return_diag or diag_path:
        df = attach_gap_metadata(df, col)
        dataset_gaps = compute_all_gap_lengths(df, col)
        gap_stats = bucket_gap_statistics(dataset_gaps)
        conf_vs_gap = compute_confidence_vs_gap(df, col)

        diag = _run_diagnostics(voter, dates, values, df, diag_path, logger)
        diag["dataset_gap_stats"] = gap_stats
        diag["confidence_vs_gap"] = conf_vs_gap
        return df, diag

    if auto_validate:
        try:
            runner = ValidationRunner()

            def ensemble_fn(d, c): # this is dangerous...the stack is gonna grow...
                return transform_with_ensemble(
                    d, c,
                    return_diag=False,
                    diag_path=None,
                    auto_validate=False
                )

            val = runner.evaluate(df_original, col, ensemble_fn)

        except Exception as e:
            logger.error(f"auto-validation failed: {e}")
            val = {"valid": False, "error": str(e)}

        if return_diag or diag_path:
            diag = diag or {}
            diag["validation"] = val
            return df, diag

        return df, {"validation": val}

    return df
