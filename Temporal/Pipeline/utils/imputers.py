# Jakob Balkovec
# Nov 16th 2025
# imputers.py

# Implements the imputer models and the VotingImputer used for filling missing satellite data.

from utils.logger import get_logger
from utils.config import load_config

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

class BaseImputer:
    # desc: Base class for all imputers.

    def __init__(self, name: str):
        # load config once per instance
        self.cfg = load_config()
        self.imputer_cfg = self.cfg.get("imputer", {})
        self.log_cfg = self.cfg.get("logging", {})

        self.name = name
        root_logger = get_logger().getChild("imputer")
        self.logger = root_logger.getChild(self.name)

        self.active = True # for graceful exits

    def fit(self, dates: pd.Series, values: pd.Series, aux_df: Optional[pd.DataFrame] = None):
        # pre:  accepts raw date/value series
        # post: stores model state if needed

        return self

    def impute(self, dates, values, aux_df=None):
        # pre:  imputers must return (filled_series, confidence_series)
        # post: must align with original index
        # note: NOP

        raise NotImplementedError

class LinearInterpolationImputer(BaseImputer):
    # desc: Time-based linear interpolation for short gaps.

    def __init__(self):
        super().__init__("linear_interp")
        cfg = self.imputer_cfg.get("linear_interp", {})
        tau = cfg.get("tau_days", 7.0)
        self.tau_days = float(tau)
        self.logger.debug(f"initialized with tau_days={self.tau_days}")

    def impute(self, dates, values, aux_df=None):
        # pre:  expects date-aligned series
        # post: returns interpolated series with confidence

        self.logger.debug(f"starting interpolation for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        s_interp = s.astype(float).interpolate(method="time")

        mask_missing = s.isna()
        self.logger.debug(f"missing mask computed: {mask_missing.sum()} missing points")

        valid_dates = pd.Series(s.index, index=s.index)
        valid_dates[mask_missing] = pd.NaT

        prev_valid = valid_dates.ffill()
        next_valid = valid_dates.bfill()

        gap_span = (next_valid - prev_valid).dt.days.astype(float)
        gap_span.replace(0, 0.1, inplace=True)

        self.logger.debug(f"computed gap spans (sample): {gap_span.head().tolist()}")

        conf = np.exp(-gap_span / self.tau_days)
        conf[~mask_missing] = 1.0

        self.logger.debug("Interpolation complete, generating confidence scores")

        return s_interp, conf.clip(0.0, 1.0)

class ForwardBackwardImputer(BaseImputer):
    # desc: Fills very short gaps and edges using ffill + bfill.

    def __init__(self):
        super().__init__("ffill_bfill")
        cfg = self.imputer_cfg.get("ffill_bfill", {})
        tau = cfg.get("tau_days", 3.0)
        self.tau_days = float(tau)
        self.logger.debug(f"initialized with tau_days={self.tau_days}")

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting ffill/bfill for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        filled = s.ffill().bfill()

        mask_missing = s.isna()
        self.logger.debug(f"missing mask has {mask_missing.sum()} entries")

        valid_dates = pd.Series(s.index, index=s.index)
        valid_dates[mask_missing] = pd.NaT

        prev_valid = valid_dates.ffill()
        next_valid = valid_dates.bfill()

        dist_prev = (s.index - prev_valid).dt.days.astype(float)
        dist_next = (next_valid - s.index).dt.days.astype(float)

        nearest = np.minimum(dist_prev, dist_next)
        nearest.replace(np.inf, np.nan, inplace=True)
        self.logger.debug("computed nearest distances for confidence weighting")

        conf = np.exp(-nearest / self.tau_days)
        conf[~mask_missing] = 1.0

        self.logger.debug("ffill/bfill imputation complete")
        return filled, conf.fillna(0.0).clip(0.0, 1.0)


class RollingMeanImputer(BaseImputer):
    # desc: Rolling mean for smoothing and medium-size gaps.

    def __init__(self):
        super().__init__("rolling_mean")
        cfg = self.imputer_cfg.get("rolling_mean", {})
        window = cfg.get("window", 7)
        self.window = int(window)
        self.logger.debug(f"initialized with window={self.window}")

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting rolling mean for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        base = s.ffill().bfill()
        rolled = base.rolling(self.window, min_periods=1, center=True).mean()

        self.logger.debug("computed rolling mean values")

        counts = s.rolling(self.window, min_periods=1, center=True).count()
        conf = (counts / float(self.window)).clip(0.0, 1.0)
        conf[~s.isna()] = 1.0

        self.logger.debug("rolling mean imputation complete")
        return rolled, conf


class ClimatologyImputer(BaseImputer):
    # desc: Seasonal DOY filling. Strong for long gaps.

    def __init__(self):
        super().__init__("climatology")
        cfg = self.imputer_cfg.get("climatology", {})
        m = cfg.get("min_count_for_high_conf", 10)
        self.min_count_for_high_conf = int(m)
        self.clim_mean = None
        self.clim_count = None
        self.logger.debug(f"initialized with min_count_for_high_conf={self.min_count_for_high_conf}")

    def fit(self, dates, values, aux_df=None):
        s = pd.Series(values.values, index=pd.to_datetime(dates))
        df = pd.DataFrame({"doy": s.index.dayofyear, "val": s.values})

        g = df.dropna(subset=["val"]).groupby("doy")["val"]
        self.clim_mean = g.mean()
        self.clim_count = g.count()

        self.logger.debug(
            f"fit complete using {len(df.dropna(subset=['val']))} valid samples; "
            f"{self.clim_count.sum()} total climatology counts"
        )

        return self

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting climatology imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.clim_mean is None:
            self.logger.debug("climatology not fit; returning fallback confidences")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        doy = s.index.dayofyear
        clim_vals = self.clim_mean.reindex(doy).values
        clim_counts = self.clim_count.reindex(doy).fillna(0.0).values

        out = s.copy()
        mask_missing = s.isna()
        out[mask_missing] = clim_vals[mask_missing]

        raw_conf = clim_counts / float(self.min_count_for_high_conf)
        raw_conf = np.clip(raw_conf, 0.0, 1.0)

        self.logger.debug("assigned climatology values and computed raw confidences")

        conf = pd.Series(0.0, index=s.index)
        conf[mask_missing] = raw_conf[mask_missing]
        conf[~mask_missing] = 1.0

        self.logger.debug("climatology imputation complete")

        return out, conf


class LinearModelImputer(BaseImputer):
    # desc: Linear regression imputer using temporal encodings and optional cross-feature predictors.

    def __init__(self):
        super().__init__("linear_model")
        cfg = self.imputer_cfg.get("linear_model", {})
        self.min_known = int(cfg.get("min_known", 15))
        self.use_cross = cfg.get("use_cross_features", True)
        self.model = None
        self.features = []
        self.logger.debug(
            f"initialized with min_known={self.min_known}, use_cross_features={self.use_cross}"
        )

    def fit(self, dates, values, aux_df=None):
        if aux_df is None or "date" not in aux_df.columns:
            self.logger.debug("aux_df missing or has no 'date' column; skipping fit")
            return self

        self.logger.debug(f"starting fit for feature '{values.name}'")

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # temporal features
        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        col = values.name
        feats = ["DOY_sin", "DOY_cos", "year"]

        if self.use_cross:
            feats += [c for c in ["LST", "NDVI", "Rain_sat"] if c != col and c in df.columns]

        self.features = feats
        self.logger.debug(f"using features: {self.features}")

        df[col] = values.values
        known = df.dropna(subset=[col])

        if len(known) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough samples ({len(known)}) to train imputer...DISABLING LM"
            )
            self.model = None
            return self

        X_train = known[feats].fillna(method="ffill").fillna(method="bfill")
        y_train = known[col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        self.model = model

        self.logger.debug(f"trained linear model with {len(known)} samples")
        return self

    def impute(self, dates, values, aux_df=None):
        if not self.active:
            self.logger.debug("imputer inactive; returning None")
            return None, None

        self.logger.debug(f"starting imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.model is None or aux_df is None:
            self.logger.debug("model not trained or aux_df missing; falling back to original values")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df[values.name] = s.reindex(df["date"].values).values

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        X_pred = df[self.features].fillna(method="ffill").fillna(method="bfill")
        preds = self.model.predict(X_pred)

        self.logger.debug("generated predictions from linear model")

        s_pred = pd.Series(preds, index=df["date"]).reindex(s.index)

        raw_conf = min(1.0, len(df.dropna(subset=[values.name])) / float(self.min_known))
        conf = pd.Series(raw_conf, index=s.index)
        conf[~s.isna()] = 1.0

        self.logger.debug("linear regression imputation complete")
        return s_pred, conf.clip(0.0, 1.0)


class XGBImputer(BaseImputer):
    # desc: Context-based imputer using temporal encodings + cross-feature signals.

    def __init__(self):
        super().__init__("xgboost")
        cfg = self.imputer_cfg.get("xgboost", {})
        min_known = cfg.get("min_known", 30)
        self.min_known = int(min_known)
        self.model = None
        self.features = []
        self.logger.debug(f"initialized with min_known={self.min_known}")

    def fit(self, dates, values, aux_df=None):
        if aux_df is None or "date" not in aux_df.columns:
            self.logger.debug("aux_df missing or has no 'date' column; skipping xgb fit")
            return self

        self.logger.debug(f"starting xgb fit for feature '{values.name}'")

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        col = values.name
        feats = ["DOY_sin", "DOY_cos", "year"]
        feats += [c for c in ["LST", "NDVI", "Rain_sat"] if c != col and c in df.columns]
        self.features = feats

        self.logger.debug(f"xgb features: {self.features}")

        df[col] = values.values
        known = df.dropna(subset=[col])

        if len(known) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough samples ({len(known)}) to train imputer -- DISABLING XGB"
            )
            self.model = None
            return self

        X = known[self.features].fillna(method="ffill").fillna(method="bfill")
        y = known[col]

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X, y)
        self.model = model

        self.logger.debug(f"trained xgb model with {len(known)} samples")
        return self

    def impute(self, dates, values, aux_df=None):
        if not self.active:
            self.logger.debug("imputer inactive; returning None")
            return None, None

        self.logger.debug(f"starting xgb imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.model is None or aux_df is None:
            self.logger.debug("xgb model not trained or aux_df missing; falling back to original values")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"]).sort_values()

        df[values.name] = s.reindex(df["date"].values).values

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        X_pred = df[self.features].fillna(method="ffill").fillna(method="bfill")
        preds = self.model.predict(X_pred)

        self.logger.debug("generated xgb predictions")

        s_pred = pd.Series(preds, index=df["date"]).reindex(s.index)

        conf = pd.Series(0.0, index=s.index)
        conf[s.isna()] = 0.7
        conf[~s.isna()] = 1.0

        self.logger.debug("xgb imputation complete")

        return s_pred, conf


class VotingImputer:
    # desc: Combines multiple imputers into a single voted result.

    def __init__(self, imputers: List[BaseImputer]):
        cfg = load_config().get("imputer", {})          # <-- FIXED
        self.imputers = imputers

        bw = cfg.get("base_weights", {})
        total = sum(bw.values()) if bw else 1.0
        self.base_weights = {k: v / total for k, v in bw.items()}

        self.outlier_factor = cfg.get("outlier_factor", 3.0)

        self.logger = get_logger().getChild("imputer.voter")
        self.logger.debug(
            f"initialized voting imputer with outlier_factor={self.outlier_factor} "
            f"and base_weights={self.base_weights}"
        )

    def fit(self, dates, values, aux_df=None):
        # pre: calls fit on each imputer
        # post: stores trained state

        self.logger.debug(f"starting fit for feature '{values.name}' using {len(self.imputers)} imputers")

        for imp in self.imputers:
            self.logger.debug(f"fitting imputer '{imp.name}'")
            try:
                imp.fit(dates, values, aux_df)
                if not getattr(imp, "active", True):
                    self.logger.warning(f"imputer '{imp.name}' disabled during fit")
            except Exception as e:
                imp.active = False
                self.logger.error(f"imputer '{imp.name}' failed during fit: {e}")

        self.logger.debug("fit complete")
        return self

    def impute(self, dates, values, aux_df=None):
        # pre: runs all imputers
        # post: combines weighted predictions, produces final and confidence series

        self.logger.debug(f"starting ensemble imputation for feature '{values.name}'")

        index = pd.to_datetime(dates)
        original = pd.Series(values.values, index=index)
        mask_missing = original.isna()

        cand_vals = {}
        cand_conf = {}

        for imp in self.imputers:
            if not getattr(imp, "active", True):
                self.logger.debug(f"skipping inactive imputer '{imp.name}'")
                continue

            self.logger.debug(f"running imputer '{imp.name}'")

            try:
                filled, conf = imp.impute(dates, values, aux_df)
            except Exception as e:
                self.logger.error(f"imputer '{imp.name}' failed during impute: {e}")
                imp.active = False
                continue

            if filled is None or conf is None:
                self.logger.warning(f"imputer '{imp.name}' returned no output; skipping")
                imp.active = False
                continue

            cand_vals[imp.name] = filled.reindex(index)
            cand_conf[imp.name] = conf.reindex(index).fillna(0.0)

        final = original.copy()
        final_conf = pd.Series(1.0, index=index)

        for t in index[mask_missing.values]: # mask_missing -> mask_missing.values
            vals = []
            weights = []
            confs = []

            for imp in self.imputers:
                if not getattr(imp, "active", True):
                    continue
                if imp.name not in cand_vals:
                    continue

                v = cand_vals[imp.name].loc[t]
                c = float(cand_conf[imp.name].loc[t])
                bw = self.base_weights.get(imp.name, 0.0)
                w = bw * c

                if not np.isnan(v) and w > 0:
                    vals.append(v)
                    weights.append(w)
                    confs.append(c)

            if not vals:
                self.logger.debug(f"no valid imputer predictions for timestamp {t}, assigning nan")
                final.loc[t] = np.nan
                final_conf.loc[t] = 0.0
                continue

            vals = np.array(vals)
            weights = np.array(weights)

            median = np.median(vals)
            mad = np.median(np.abs(vals - median)) + 1e-6

            for i in range(len(vals)):
                if abs(vals[i] - median) > self.outlier_factor * mad:
                    self.logger.debug(
                        f"outlier detected at time {t} for prediction value={vals[i]:.4f}, "
                        f"median={median:.4f}, mad={mad:.4f}"
                    )
                    weights[i] *= 0.1

            if weights.sum() == 0:
                weights[:] = 1.0

            weights /= weights.sum()

            final_val = np.sum(vals * weights)
            final_conf_val = np.sum(np.array(confs) * weights)

            final.loc[t] = final_val
            final_conf.loc[t] = final_conf_val

            self.logger.debug(
                f"timestamp {t}: ensemble_value={final_val:.4f}, confidence={final_conf_val:.3f}"
            )

        self.logger.debug("ensemble imputation complete")
        return final, final_conf.clip(0.0, 1.0)


    def diagnostics(self, dates, values, aux_df=None, path: str = None):
        # desc: returns a dictionary summarizing imputer performance and participation.
        #       if path is provided, writes the summary to a json file.

        index = pd.to_datetime(dates)
        original = pd.Series(values.values, index=index)
        mask_missing = original.isna()

        summary = {
            "feature": values.name,
            "n_missing": int(mask_missing.sum()),
            "imputers": {}
        }

        for imp in self.imputers:
            name = imp.name
            active = getattr(imp, "active", True)

            entry = {
                "active": active,
                "contributions": 0,
                "avg_conf": 0.0,
                "skipped": False,
                "fail_reason": None,
            }

            if not active:
                entry["skipped"] = True
                entry["fail_reason"] = "inactive"
                summary["imputers"][name] = entry
                continue

            try:
                filled, conf = imp.impute(dates, values, aux_df)
            except Exception as e:
                entry["active"] = False
                entry["skipped"] = True
                entry["fail_reason"] = str(e)
                summary["imputers"][name] = entry
                continue

            if filled is None or conf is None:
                entry["active"] = False
                entry["skipped"] = True
                entry["fail_reason"] = "returned None"
                summary["imputers"][name] = entry
                continue

            filled = filled.reindex(index)
            conf = conf.reindex(index).fillna(0.0)

            missing_idx = index[mask_missing.values]

            contrib_count = 0
            conf_values = []

            for t in missing_idx:
                v = filled.loc[t]
                c = conf.loc[t]
                if not np.isnan(v) and c > 0:
                    contrib_count += 1
                    conf_values.append(float(c))

            entry["contributions"] = contrib_count
            entry["avg_conf"] = float(np.mean(conf_values)) if conf_values else 0.0

            summary["imputers"][name] = entry

        if path is not None:
            try:
                import json
                from pathlib import Path

                out_path = Path(path)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                self.logger.debug(f"wrote diagnostics to '{path}'")

            except Exception as e:
                self.logger.error(f"failed to write diagnostics to '{path}': {e}")

        return summary


def transform_with_ensemble(df, col, return_diag = False, diag_path = None
                            ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    # desc: Convenience function for running full voting ensemble for one column.

    logger = get_logger().getChild(f"imputer.ensemble.{col}")
    logger.debug(f"starting ensemble transform for column '{col}'")

    if "date" not in df.columns:
        logger.error("missing required 'date' column")
        raise ValueError("run_ensemble requires a 'date' column")

    df = df.copy().sort_values("date")
    logger.debug(f"sorted dataframe with {len(df)} rows")

    dates = pd.to_datetime(df["date"])
    values = df[col]

    imputers = [
        LinearInterpolationImputer(),
        ForwardBackwardImputer(),
        RollingMeanImputer(),
        ClimatologyImputer(),
        LinearModelImputer(),
        XGBImputer(),
    ]

    logger.debug(f"initialized {len(imputers)} imputers for column '{col}'")

    voter = VotingImputer(imputers)

    logger.debug("fitting ensemble")
    voter.fit(dates, values, aux_df=df)

    logger.debug("performing ensemble imputation")
    filled, conf = voter.impute(dates, values, aux_df=df)

    df[col + "_interp"] = filled.values
    df[col + "_conf"] = conf.values

    logger.debug(
        f"completed ensemble transform for '{col}' "
        f"with coverage={filled.notna().mean():.3f}"
    )

    # optionally run diagnostics
    if return_diag or diag_path:
        logger.debug("running diagnostics")
        diag = voter.diagnostics(
            dates,
            values,
            aux_df=df,
            path=diag_path
        )
        return df, diag

    return df
