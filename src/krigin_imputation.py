import pandas as pd
import numpy as np
import gstools as gs
import h5py
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def prepare_data(df_var, target_time, var):
    """
    Extract known and unknown points for a given time and variable.
    """

    time_data = df_var[df_var['time'] == target_time]
    known = time_data[~time_data[var].isna()]
    unknown = time_data[time_data[var].isna()]

    if len(known) == 0 or len(unknown) == 0:
        return None, None, None

    known_points = known[['east', 'north', 'altitude', var]].values
    unknown_points = unknown[['east', 'north', 'altitude']].values
    unknown_stations = unknown['station'].values

    return known_points, unknown_points, unknown_stations

def ked_interpolation_gstools(known_points, unknown_points):
    """
    Perform kriging with a custom variogram model fit for each run.
    """
    x_known, y_known, z_known = known_points[:, 0], known_points[:, 1], known_points[:, 2]
    values = known_points[:, 3]
    x_unknown, y_unknown, z_unknown = unknown_points[:, 0], unknown_points[:, 1], unknown_points[:, 2]

    if np.all(values == 0):
        return np.zeros(len(unknown_points))

    bin_center, gamma = gs.vario_estimate(
        (x_known, y_known),
        values,
        bin_edges=np.linspace(0, 100000, 15)
    )

    model = gs.Spherical(dim=2)
    model.fit_variogram(bin_center, gamma)

    ked = gs.krige.ExtDrift(
        model=model,
        cond_pos=(x_known, y_known),
        cond_val=values,
        ext_drift=z_known,
    )

    predictions, _ = ked(
        (x_unknown, y_unknown),
        ext_drift=z_unknown,
        return_var=True
    )

    return predictions

def ked_interpolation_gstools_fixed(known_points, unknown_points):
    """
    Kriging with External Drift using a fixed variogram model.
    """
    x_known = known_points[:, 0]
    y_known = known_points[:, 1]
    z_known = known_points[:, 2]
    values = known_points[:, 3]

    x_unknown = unknown_points[:, 0]
    y_unknown = unknown_points[:, 1]
    z_unknown = unknown_points[:, 2]

    if np.all(values == 0):
        return np.zeros(len(unknown_points))

    model = gs.Spherical(dim=2, var=1.0, len_scale=10000, nugget=0.1)

    ked = gs.krige.ExtDrift(
        model=model,
        cond_pos=(x_known, y_known),
        cond_val=values,
        ext_drift=z_known,
    )

    predictions, _ = ked(
        (x_unknown, y_unknown),
        ext_drift=z_unknown,
        return_var=True
    )

    return predictions

def interpolate_time(df_var, var, time):
    results = []
    known_points, unknown_points, unknown_stations = prepare_data(df_var, time, var)

    if known_points is None or len(known_points) < 5:
        return results

    try:
        interpolated = ked_interpolation_gstools(known_points, unknown_points)

        for station, value in zip(unknown_stations, interpolated):
            # Apply constraints
            if var == 'precip':
                value = max(value, 0)
            elif var in ['North', 'East']:
                value = max(value, 0)
            elif var == 'moisture':
                value = min(value, 100)
                value = max(0, value)

            results.append({
                'time': time,
                'station': station,
                var: round(value, 1),
                f'{var}_interpolated': True
            })

    except Exception as e:
        print(f"[{var}] Error interpolating {time}: {e}")

    return results

def interpolate_variable(df, var, output_dir, position=0):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df_var = df[['time', 'station', 'east', 'north', 'altitude', var]].copy()

    missing_times = df_var.loc[df_var[var].isna(), 'time'].drop_duplicates()
    results = []

    for t in tqdm(missing_times, desc=f'Interpolating {var}', position=position, leave=True):
        known_points, unknown_points, unknown_stations = prepare_data(df_var, t, var)

        if known_points is None or len(known_points) < 5:
            continue

        try:
            interpolated = ked_interpolation_gstools(known_points, unknown_points)
            for station, value in zip(unknown_stations, interpolated):
                if var == 'precip':
                    value = max(value, 0)
                elif var in ['North', 'East']:
                    value = max(value, 0)
                elif var == 'moisture':
                    value = min(value, 100)

                results.append({
                    'time': t,
                    'station': station,
                    var: round(value, 1),
                    f'{var}_interpolated': True
                })
        except Exception as e:
            print(f"Error interpolating {var} at {t}: {e}")

    if results:
        df_known = df_var.dropna(subset=[var]).copy()
        df_known[f"{var}_interpolated"] = False

        df_results = pd.DataFrame(results)
        df_results['time'] = pd.to_datetime(df_results['time'])

        df_combined = pd.concat([df_var.dropna(subset=[var]), df_results], ignore_index=True)
        df_combined = df_combined.sort_values(by=['station', 'time'])

        output_path = os.path.join(output_dir, f"{var}_interpolated.csv")
        df_combined.to_csv(output_path, index=False)
        print(f"Saved {var} interpolation to {output_path}")
    else:
        print(f"No results for variable {var}")

def call_interpolate_variable(args):
    return interpolate_variable(*args)


def run_interpolation_pipeline(timeseries_file, output_dir, variables):
    stations_info = pd.read_csv("data/clean/valais_stations.csv")
    station_filter = stations_info[['station', 'east', 'north', 'altitude']]
    df_missing = pd.read_csv(timeseries_file, parse_dates=['time'])
    df = df_missing.merge(station_filter, how='left', on='station')
    os.makedirs(output_dir, exist_ok=True)

    args = [(df, var, output_dir, i) for i, var in enumerate(variables)]
    process_map(call_interpolate_variable, args, max_workers=os.cpu_count(), desc="Variables")


if __name__ == "__main__":
    timeseries_file = 'data/interpolated/merged_interpol.csv'
    output_dir = 'data/interpol'
    variables = ['temperature', 'precip', 'pression', 'moisture', 'North', 'East']

    run_interpolation_pipeline(timeseries_file, output_dir, variables)

    
    """
    
    timeseries_file = 'data/filtered/merged_valais.csv'
    df = pd.read_csv(timeseries_file)
    mae, rmse = leave_one_out(df, 'precip', end_date=pd.to_datetime('2019-01-31 23:50'))
    print(f'Mean Absolute Error is: {mae}')
    print(f'Root Mean Square Error is: {rmse}')
    """  