import pandas as pd
import numpy as np
import os
import gstools as gs
from tqdm.contrib.concurrent import process_map
from krigin_imputation import ked_interpolation_gstools

def _process_timestamp(args):
    timestamp, df, var, drift = args
    errors = []
    preds = []
    trues = []
    subset = df[df['time'] == timestamp]
    subset = subset[~subset[var].isna()]
    if len(subset) < 2:
        return errors, preds, trues

    for i in range(len(subset)):
        test_point = subset.iloc[i]
        train_set = subset.drop(subset.index[i])
        known_points = train_set[['east', 'north', drift, var]].values
        unknown_points = test_point[['east', 'north', drift]].values.reshape(1, -1)
        try:
            pred = ked_interpolation_gstools(known_points, unknown_points)[0]
            true_value = test_point[var]
            errors.append(true_value - round(pred, 1))
            preds.append(pred)
            trues.append(true_value)
        except Exception:
            continue
    return errors, preds, trues

def leave_one_out_kriging(df, var, drift='altitude', n=100, workers=None):
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    all_timestamps = df['time'].unique()
    if len(all_timestamps) < n:
        raise ValueError(f"Not enough timestamps: requested {n}, available {len(all_timestamps)}")
    selected_timestamps = np.random.choice(all_timestamps, size=n, replace=False)

    args_list = [(ts, df, var, drift) for ts in selected_timestamps]

    if workers is None:
        workers = os.cpu_count() or 1

    results = process_map(
        _process_timestamp,
        args_list,
        max_workers=workers,
        desc=f"Processing {var}"
    )

    # Flatten results
    errors = [e for r in results for e in r[0]]
    preds = [p for r in results for p in r[1]]
    trues = [t for r in results for t in r[2]]

    mae = np.mean(np.abs(errors)) if errors else np.nan
    rmse = np.sqrt(np.mean(np.square(errors))) if errors else np.nan

    return mae, rmse, preds, trues

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

def main_wind():
    drift = 'average_wind'
    df = pd.read_csv('data/filtered/merged_valais.csv')
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')

    # 1. LOOCV for speed directly
    print("==> LOOCV for speed (magnitude)")
    df['speed'] = np.sqrt(df['East']**2 + df['North']**2)
    mae_speed, rmse_speed, _, _ = leave_one_out_kriging(df, var='speed', drift=drift, n=10)

    # 2. LOOCV for East and North separately
    print("==> LOOCV for East component")
    mae_east, rmse_east, pred_east, true_east = leave_one_out_kriging(df, var='East', drift=drift, n=10)

    print("==> LOOCV for North component")
    mae_north, rmse_north, pred_north, true_north = leave_one_out_kriging(df, var='North', drift=drift, n=10)

    # 3. Reconstructed speed
    if pred_east and pred_north:
        reconstructed_speed = np.sqrt(np.array(pred_east)**2 + np.array(pred_north)**2)
        true_speed = np.sqrt(np.array(true_east)**2 + np.array(true_north)**2)
        speed_errors = true_speed - reconstructed_speed
        mae_vec = np.mean(np.abs(speed_errors))
        rmse_vec = np.sqrt(np.mean(speed_errors**2))
    else:
        mae_vec = rmse_vec = np.nan

    # 4. Print comparison
    print("\n===== RESULTS =====")
    print(f"Speed (direct):      MAE = {mae_speed:.3f}, RMSE = {rmse_speed:.3f}")
    print(f"East  (component):   MAE = {mae_east:.3f}, RMSE = {rmse_east:.3f}")
    print(f"North (component):   MAE = {mae_north:.3f}, RMSE = {rmse_north:.3f}")
    print(f"Speed (recombined):  MAE = {mae_vec:.3f}, RMSE = {rmse_vec:.3f}")

def main():
    var = 'North'
    drift = 'altitude'
    df = pd.read_csv('data/filtered/merged_valais.csv')

    mae, rmse, _, _ = leave_one_out_kriging(df, var=var, drift=drift, n=5)
    print(f"For {var} using drift '{drift}', MAE: {mae:.3f}, RMSE: {rmse:.3f}")


def main_humidity():
    var = 'East'
    drift = 'temp_drift'
    df = pd.read_csv('data/filtered/merged_valais.csv')
    df_temp = pd.read_csv('data/clean/valais_clean.csv')
    df_temp['temp_drift'] = df_temp['temperature']
    df = df.merge(df_temp[['station', 'time', 'temp_drift']], on=['station', 'time'], how='left')

    mae, rmse, _, _ = leave_one_out_kriging(df, var=var, drift=drift, n=5)
    print(f"For {var} using drift '{drift}', MAE: {mae:.3f}, RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main()