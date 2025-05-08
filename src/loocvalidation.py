import pandas as pd
import numpy as np
import os
import gstools as gs
from tqdm.contrib.concurrent import process_map
from krigin_imputation import ked_interpolation_gstools

def _process_timestamp(args):
    # Unpack arguments
    timestamp, df, var = args
    errors = []
    subset = df[df['time'] == timestamp]
    subset = subset[~subset[var].isna()]
    if len(subset) < 2:
        return errors
    
    # Leave-one-out for this timestamp
    for i in range(len(subset)):
        test_point = subset.iloc[i]
        train_set = subset.drop(subset.index[i])
        known_points = train_set[['east', 'north', 'altitude', var]].values
        unknown_points = test_point[['east', 'north', 'altitude']].values.reshape(1, -1)
        try:
            pred = ked_interpolation_gstools(known_points, unknown_points)[0]
            true_value = test_point[var]
            errors.append(true_value - round(pred, 1))
        except Exception as e:
            # Skip errors for this point
            continue
    return errors

def leave_one_out_kriging(df, var, n=100, workers=None):
    """
    Leave-One-Out Cross-Validation (LOOCV) for Kriging with External Drift using parallel processing.
    """
    # Parse and sample timestamps
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    all_timestamps = df['time'].unique()
    if len(all_timestamps) < n:
        raise ValueError(f"Not enough timestamps: requested {n}, available {len(all_timestamps)}")
    selected_timestamps = np.random.choice(all_timestamps, size=n, replace=False)

    # Prepare arguments for parallel processing
    args_list = [(ts, df, var) for ts in selected_timestamps]
    # Determine worker count
    if workers is None:
        workers = os.cpu_count() or 1

    # Run LOOCV in parallel per timestamp
    results = process_map(
        _process_timestamp,
        args_list,
        max_workers=workers,
        desc="Processing timestamps"
    )

    # Flatten errors
    errors = [err for sublist in results for err in sublist]

    # Compute metrics
    mae = np.mean(np.abs(errors)) if errors else np.nan
    rmse = np.sqrt(np.mean(np.square(errors))) if errors else np.nan
    return mae, rmse

def _process_closest(args):
    timestamp, df, var = args
    errors = []
    subset = df[df['time'] == timestamp]
    subset = subset[~subset[var].isna()]
    if len(subset) < 2:
        return errors
    
    # Interpolate with closest k station for this timestamp
    for i in range(len(subset)):
        test_point = subset.iloc[i]
        train_set = subset.drop(subset.index[i])
        known_points = train_set[['east', 'north', 'altitude', var]].values
        unknown_points = test_point[['east', 'north', 'altitude']].values.reshape(1, -1)
        try:
            
            pred = ked_interpolation_gstools(known_points, unknown_points)[0]
            true_value = test_point[var]
            errors.append(true_value - round(pred, 1))
        except Exception as e:
            # Skip errors for this point
            continue
    return errors


def leave_one_out_k_closest(df, var, n=100, workers=None):
    None

def main():
    var = 'East'
    df = pd.read_csv('data/filtered/merged_valais.csv')
    mae, rmse = leave_one_out_kriging(df, var=var, n=10)
    print(f"For {var}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main()