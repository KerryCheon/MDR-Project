# Jakob Balkovec
# Now 8th 2025
# impute_models.py

# This module defines imputation models for handling missing values in satellite data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def run_linear(df, col):
    # pre: the df has a datetime index and a column 'col' with missing values
    # post: the df with an additional column 'col_interp' with imputed values
    # desc: Imputes missing values in 'col' using linear regression based on the index position.

    known = df.dropna(subset=[col])
    X = np.arange(len(known)).reshape(-1, 1)
    y = known[col].values
    model = LinearRegression().fit(X, y)
    full_pred = model.predict(np.arange(len(df)).reshape(-1, 1))
    df[col + "_interp"] = full_pred
    return df

def run_rolling(df, col, window=7):
    # pre: the df has a datetime index and a column 'col' with missing values
    # post: the df with an additional column 'col_interp' with imputed values
    # desc: Imputes missing values in 'col' using rolling mean with forward fill.

    df[col + "_interp"] = df[col].ffill().rolling(window, min_periods=1).mean()
    return df

def run_xgboost(df, col):
    # pre: df has a 'date' column and numeric features including the target col
    # post: adds col_interp (model predictions) to df
    # desc: XGBoost-based imputation leveraging temporal and cross-satellite context.

    df = df.copy().sort_values("date").reset_index(drop=True)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # temporal encoding (sin/cos improves continuity across year boundaries)
    df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # select predictors — temporal + cross-satellite context if available
    features = ["DOY_sin", "DOY_cos", "year"]
    aux = [c for c in ["LST", "NDVI", "Rain_sat"] if c != col and c in df.columns]
    features += aux

    # fallback if too few known values
    known = df.dropna(subset=[col])
    if len(known) < 5:
        df[col + "_interp"] = df[col]
        return df

    X_train = known[features].fillna(method="ffill").fillna(method="bfill")
    y_train = known[col]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_pred = df[features].fillna(method="ffill").fillna(method="bfill")
    df[col + "_interp"] = model.predict(X_pred)

    return df


def bridge_test(df, col, model_fn, model_type=None, start_date=None, end_date=None, gap=None, verbose=True):
    # pre: df has a 'date' column and a column 'col' with known values at start_date and end_date
    # post: returns a dict with imputation results and metrics
    # desc: Simulates a missing value bridge test by masking values between start_date and end

    df = df.copy().sort_values("date").reset_index(drop=True)
    df = df[["date", col]].drop_duplicates(subset="date")
    df = df.set_index("date")

    if start_date is None or end_date is None:
        valid_dates = df[col].dropna().index.to_list()
        found = False
        for i in range(len(valid_dates) - 1):
            diff = (valid_dates[i + 1] - valid_dates[i]).days
            if diff >= (gap or 5):
                start_date, end_date = valid_dates[i], valid_dates[i + 1]
                found = True
                break
        if not found:
            raise ValueError("No suitable start/end pair found with required gap.")

    if start_date not in df.index or end_date not in df.index:
        raise ValueError("Start or end date not found in dataset with known values.")

    true_val = df.loc[end_date, col]

    bridge_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    bridge_df = pd.DataFrame({"date": bridge_dates})
    bridge_df = bridge_df.merge(df.reset_index(), on="date", how="left")

    mask = (bridge_df["date"] > start_date) & (bridge_df["date"] < end_date)
    bridge_df.loc[mask, col] = np.nan

    bridge_df_interp = model_fn(bridge_df.copy(), col)

    pred_val = bridge_df_interp.loc[
        bridge_df_interp["date"] == end_date, col + "_interp"
    ].values[0]
    error = pred_val - true_val
    rmse = np.sqrt(mean_squared_error([true_val], [pred_val]))
    mae = mean_absolute_error([true_val], [pred_val])

    if verbose:
        model_name = model_type or model_fn.__name__
        print("=" * 70)
        print(f"Imputation Bridge Test for: {col} | Model: {model_name}")
        print("-" * 70)
        print(f" Start Date       : {start_date.date()}")
        print(f" End Date         : {end_date.date()}")
        print(f" Days Spanned     : {(end_date - start_date).days}")
        print(f" True End Value   : {true_val:10.4f}")
        print(f" Predicted Value  : {pred_val:10.4f}")
        print(f" Error (signed)   : {error:10.4f}")
        print(f" Absolute Error   : {abs(error):10.4f}")
        print(f" RMSE             : {rmse:10.4f}")
        print(f" MAE              : {mae:10.4f}")
        print(f" Model Type       : {model_name}")
        print("-" * 70)
        print(" Intermediate Values (Simulated Daily Interpolation):\n")
        print(bridge_df_interp.to_string(index=False))
        print("=" * 70)

    return {
        "feature": col,
        "start_date": start_date,
        "end_date": end_date,
        "gap_days": (end_date - start_date).days,
        "true_val": true_val,
        "pred_val": pred_val,
        "error": error,
        "abs_error": abs(error),
        "RMSE": rmse,
        "MAE": mae,
        "interp_segment": bridge_df_interp,
        "model": model_type or model_fn.__name__
    }
