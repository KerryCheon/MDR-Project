# Jakob Balkovec
# Nov 16th 2025
# voting.py

import pandas as pd
import numpy as np
from typing import List

from pipeline.Pipeline.imputers.base import BaseImputer
from pipeline.Pipeline.utils.config import load_config
from pipeline.Pipeline.utils.logger import get_logger

class VotingImputer:
    # desc: Combines multiple imputers into a single voted result.

    def __init__(self, imputers: List[BaseImputer], config=None):
        cfg = (config or load_config()).get("imputer", {})
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
        real_mask = ~values.isna()

        if real_mask.any():
            first_real = index[real_mask].min()
            last_real  = index[real_mask].max()

            # clip to observed bounds
            clipped = values.copy()
            clipped[index < first_real] = values[real_mask].iloc[0]   # pre-fill
            clipped[index > last_real]  = values[real_mask].iloc[-1]  # post-fill
            values = clipped
        else:
            raise RuntimeError("No observed values available for imputation")

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
