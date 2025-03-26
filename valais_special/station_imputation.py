import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def main():
    # Load dataset
    merged_valais = pd.read_csv('data/filtered/merged_valais.csv')
    station_list = pd.read_csv('data/processed/stations.csv')
    stations_withcanton = pd.read_csv('valais_special/data/station_list.csv', delimiter=';', encoding='Windows 1252')

    # Merge station list with canton information
    best_station_list = pd.merge(station_list, stations_withcanton[['Canton', 'Abr.']], 
                                 how='left', left_on='station', right_on='Abr.')

    # Filter for Valais stations
    valais_stations = best_station_list[best_station_list['Canton'] == 'VS'].copy()

    # Compute pairwise distances between stations
    coords = valais_stations[['east', 'north']].values
    station_ids = valais_stations['station'].values
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # Find the 4 closest stations for each station
    closest_stations = {station: station_ids[np.argsort(dist_matrix[i])[1:5]]  # Take up to 4 closest stations
                        for i, station in enumerate(station_ids)}

    # Extract up to 4 closest stations into separate columns
    closest_df = pd.DataFrame.from_dict(closest_stations, orient='index', columns=['closest_station_1', 'closest_station_2', 'closest_station_3', 'closest_station_4'])
    closest_df.reset_index(inplace=True)
    closest_df.rename(columns={'index': 'station'}, inplace=True)

    # Merge closest station information with dataset
    merged_valais = merged_valais.merge(closest_df, on='station', how='left')

    # Identify columns to impute (excluding precipitation)
    columns_imputable = ['moisture', 'pression', 'temperature', 'North', 'East']

    # Iteratively fill missing values using up to 4 closest stations
    for col in tqdm(columns_imputable, desc="Imputing missing values"):
        for i in range(1, 5):  # Use 1st to 4th closest stations
            closest_col = f"{col}_closest_{i}"
            if f'closest_station_{i}' not in merged_valais.columns:
                continue  # Skip if there are fewer than 4 closest stations

            merged_closest = merged_valais[['time', 'station', col]].rename(
                columns={'station': f'closest_station_{i}', col: closest_col}
            )

            # Merge with the corresponding closest station's data
            merged_valais = merged_valais.merge(merged_closest, on=['time', f'closest_station_{i}'], how='left')

            # Fill missing values using the closest station's data
            merged_valais[col] = merged_valais[col].fillna(merged_valais[closest_col])
            merged_valais.drop(columns=[closest_col], inplace=True)

    # Final cleanup
    merged_valais.drop(columns=[f'closest_station_{i}' for i in range(1, 5)], inplace=True, errors='ignore')
    merged_valais['North'] = merged_valais['North'].round(2)
    merged_valais['East'] = merged_valais['East'].round(2)

    # Save the dataset
    merged_valais.to_csv('data/clean/valais_imputed.csv', index=False)
    print('Imputation complete. Saved to data/clean/valais_imputed.csv.')

if __name__ == "__main__":
    main()
