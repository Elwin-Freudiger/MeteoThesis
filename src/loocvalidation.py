import os
import numpy as np
import pandas as pd
import gstools as gs

from tqdm.contrib.concurrent import process_map
from krigin_imputation import ked_interpolation_gstools


# ============================================================
# Utility helpers
# ============================================================

def _prepare_df(df, time_col="time"):
    df = df.copy()
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], format="%Y%m%d%H%M")
    return df


def _safe_error(true_value, pred, rounding_digits=1):
    if pred is None or pd.isna(pred):
        return None

    pred_eval = round(float(pred), rounding_digits) if rounding_digits is not None else float(pred)
    return float(true_value) - pred_eval


def _metrics_from_records(records):
    if not records:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    errors = np.array([r["error"] for r in records], dtype=float)

    return {
        "n": len(errors),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
    }


def _estimate_len_scale(coords):
    """
    Simple robust default for ordinary kriging.
    Uses median pairwise distance between stations.
    """
    coords = np.asarray(coords, dtype=float)

    if len(coords) < 2:
        return 1.0

    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    positive = dists[dists > 0]

    if len(positive) == 0:
        return 1.0

    return float(np.median(positive))

# ============================================================
# Prediction functions
# ============================================================

def predict_ked(train_set, test_point, var, drift="altitude"):
    """
    Kriging with external drift using your existing function.

    Expected known_points columns:
    east, north, drift, var

    Expected unknown_points columns:
    east, north, drift
    """
    known_points = train_set[["east", "north", drift, var]].values
    unknown_points = test_point[["east", "north", drift]].values.reshape(1, -1)

    pred = ked_interpolation_gstools(known_points, unknown_points)[0]
    return float(pred)

def predict_ordinary_kriging(
    train_set,
    test_point,
    var,
    model_cls=gs.Exponential,
    len_scale=None,
):
    """
    Ordinary kriging without external drift.

    Uses only:
    east, north, var
    """
    coords = train_set[["east", "north"]].to_numpy(dtype=float)
    values = train_set[var].to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan

    value_var = float(np.nanvar(values))
    if value_var <= 0 or np.isnan(value_var):
        # If all training values are identical, kriging is unnecessary.
        return float(np.nanmean(values))

    if len_scale is None:
        len_scale = _estimate_len_scale(coords)

    model = model_cls(
        dim=2,
        var=value_var,
        len_scale=len_scale,
    )

    ok = gs.krige.Ordinary(
        model,
        cond_pos=[coords[:, 0], coords[:, 1]],
        cond_val=values,
    )

    x0 = np.array([float(test_point["east"])])
    y0 = np.array([float(test_point["north"])])

    pred = ok((x0, y0), return_var=False)

    # GSTools can return slightly different shapes depending on version.
    pred = np.asarray(pred).ravel()[0]
    return float(pred)

def predict_closest_station(train_set, test_point, var):
    """
    Closest station interpolation for one timestamp.

    Uses horizontal distance in east/north coordinates.
    """
    coords = train_set[["east", "north"]].to_numpy(dtype=float)
    values = train_set[var].to_numpy(dtype=float)

    x0 = float(test_point["east"])
    y0 = float(test_point["north"])

    dists = np.sqrt((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2)
    idx = int(np.argmin(dists))

    return float(values[idx])


def predict_forward_fill(df_all, test_point, var):
    """
    Forward fill prediction for the held-out station and timestamp.

    Uses the previous available value from the same station.
    Does not use the current hidden value.
    """
    station = test_point["station"]
    timestamp = test_point["time"]

    hist = df_all[
        (df_all["station"] == station)
        & (df_all["time"] < timestamp)
        & (~df_all[var].isna())
    ].sort_values("time")

    if hist.empty:
        return np.nan

    return float(hist.iloc[-1][var])


def predict_backward_fill(df_all, test_point, var):
    """
    Backward fill prediction for the held-out station and timestamp.

    Uses the next available value from the same station.
    Does not use the current hidden value.
    """
    station = test_point["station"]
    timestamp = test_point["time"]

    future = df_all[
        (df_all["station"] == station)
        & (df_all["time"] > timestamp)
        & (~df_all[var].isna())
    ].sort_values("time")

    if future.empty:
        return np.nan

    return float(future.iloc[0][var])


# ============================================================
# LOOCV worker
# ============================================================

def _process_timestamp_all_methods(args):
    timestamp, df, var, drift, rounding_digits = args

    records = {
        "ked": [],
        "ordinary_kriging": [],
        "forward_fill": [],
        "backward_fill": [],
        "closest_station": [],
    }

    subset = df[df["time"] == timestamp].copy()
    subset = subset[~subset[var].isna()]

    if len(subset) < 2:
        return records

    for i in range(len(subset)):
        test_point = subset.iloc[i]
        train_set = subset.drop(subset.index[i])

        true_value = test_point[var]
        station = test_point.get("station", None)

        # ----------------------------
        # 1. Kriging with external drift
        # ----------------------------
        try:
            if drift in train_set.columns and drift in test_point.index:
                train_ked = train_set.dropna(subset=["east", "north", drift, var])
                if len(train_ked) >= 2 and not pd.isna(test_point[drift]):
                    pred = predict_ked(train_ked, test_point, var, drift=drift)
                    err = _safe_error(true_value, pred, rounding_digits)
                    if err is not None:
                        records["ked"].append({
                            "time": timestamp,
                            "station": station,
                            "true": float(true_value),
                            "pred": float(pred),
                            "error": err,
                        })
        except Exception:
            pass

        # ----------------------------
        # 2. Ordinary kriging, no drift
        # ----------------------------
        try:
            train_ok = train_set.dropna(subset=["east", "north", var])
            if len(train_ok) >= 2:
                pred = predict_ordinary_kriging(train_ok, test_point, var)
                err = _safe_error(true_value, pred, rounding_digits)
                if err is not None:
                    records["ordinary_kriging"].append({
                        "time": timestamp,
                        "station": station,
                        "true": float(true_value),
                        "pred": float(pred),
                        "error": err,
                    })
        except Exception:
            pass

        # ----------------------------
        # 3. Closest station
        # ----------------------------
        try:
            train_closest = train_set.dropna(subset=["east", "north", var])
            if len(train_closest) >= 1:
                pred = predict_closest_station(train_closest, test_point, var)
                err = _safe_error(true_value, pred, rounding_digits)
                if err is not None:
                    records["closest_station"].append({
                        "time": timestamp,
                        "station": station,
                        "true": float(true_value),
                        "pred": float(pred),
                        "error": err,
                    })
        except Exception:
            pass

        # ----------------------------
        # 4. Forward fill
        # ----------------------------
        try:
            pred = predict_forward_fill(df, test_point, var)
            err = _safe_error(true_value, pred, rounding_digits)
            if err is not None:
                records["forward_fill"].append({
                    "time": timestamp,
                    "station": station,
                    "true": float(true_value),
                    "pred": float(pred),
                    "error": err,
                })
        except Exception:
            pass

        # ----------------------------
        # 5. Backward fill
        # ----------------------------
        try:
            pred = predict_backward_fill(df, test_point, var)
            err = _safe_error(true_value, pred, rounding_digits)
            if err is not None:
                records["backward_fill"].append({
                    "time": timestamp,
                    "station": station,
                    "true": float(true_value),
                    "pred": float(pred),
                    "error": err,
                })
        except Exception:
            pass

    return records


# ============================================================
# Main comparison function
# ============================================================

def compare_interpolation_methods(
    df,
    var,
    drift="altitude",
    n=100,
    workers=None,
    random_state=42,
    rounding_digits=1,
):
    """
    Runs LOOCV comparison for:

    - kriging with external drift
    - ordinary kriging without drift
    - forward fill
    - backward fill
    - closest station

    Returns:
    metrics_df, records_df
    """
    df = _prepare_df(df)

    required = {"time", "station", "east", "north", var}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if drift not in df.columns:
        raise ValueError(f"Drift column '{drift}' not found in dataframe.")

    all_timestamps = df["time"].dropna().unique()

    if len(all_timestamps) < n:
        raise ValueError(
            f"Not enough timestamps: requested {n}, available {len(all_timestamps)}"
        )

    rng = np.random.default_rng(random_state)
    selected_timestamps = rng.choice(all_timestamps, size=n, replace=False)

    args_list = [
        (ts, df, var, drift, rounding_digits)
        for ts in selected_timestamps
    ]

    if workers is None:
        workers = os.cpu_count() or 1

    results = process_map(
        _process_timestamp_all_methods,
        args_list,
        max_workers=workers,
        desc=f"Comparing interpolation methods for {var}",
    )

    # Flatten records
    all_records = []
    by_method = {
        "ked": [],
        "ordinary_kriging": [],
        "forward_fill": [],
        "backward_fill": [],
        "closest_station": [],
    }

    for result in results:
        for method, method_records in result.items():
            by_method[method].extend(method_records)

            for r in method_records:
                row = r.copy()
                row["method"] = method
                all_records.append(row)

    metrics_rows = []

    for method, method_records in by_method.items():
        metrics = _metrics_from_records(method_records)
        metrics_rows.append({
            "method": method,
            "n": metrics["n"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "bias": metrics["bias"],
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae")
    records_df = pd.DataFrame(all_records)

    return metrics_df, records_df

def main():

    vars = {'precip':'altitude',
            'moisture':'altitude',
            'pression':'altitude',
            'temperature':'altitude',
            'North':'wind_speed',
            'East':'wind_speed'}
    
    stations_info = pd.read_csv("../data/clean/stations_with_wind_speed.csv")
    stations_info = stations_info[['station', 'wind_speed']]
    df = pd.read_csv("../data/filtered/merged_valais.csv")
    df = df.merge(right=stations_info, how='left', on='station')

    for var in vars:
        metrics_df, records_df = compare_interpolation_methods(
            df=df,
            var=var,
            drift=vars[var],
            n=50,
            workers=None,
            random_state=42,
            rounding_digits=1,
        )

        print(f"\n===== COMPARISON RESULTS: {var} =====")
        print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()