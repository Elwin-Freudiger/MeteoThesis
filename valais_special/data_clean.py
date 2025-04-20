import pandas as pd
import numpy as np

def merge():
    stations = pd.read_csv("valais_special/data/station_list.csv", encoding='Windows 1252', delimiter=';')
    valais_stations = stations[stations['Canton']=="VS"]

    # Load datasets
    precipitation = pd.read_csv("data/processed/precipitation.csv")
    moisture = pd.read_csv("data/processed/moisture.csv")
    pression = pd.read_csv("data/processed/pression.csv")
    temperature = pd.read_csv("data/processed/temperature.csv")
    wind = pd.read_csv("data/processed/wind_vectors.csv")

    # Only select stations in Valais
    valais_stations_list = valais_stations["Abr."].tolist()
    precipitation_valais = precipitation[precipitation['station'].isin(valais_stations_list)]
    moisture_valais = moisture[moisture['station'].isin(valais_stations_list)]
    pression_valais = pression[pression['station'].isin(valais_stations_list)]
    temperature_valais = temperature[temperature['station'].isin(valais_stations_list)]
    wind_valais = wind[wind['station'].isin(valais_stations_list)]

    # Filter out stations with too many missing values
    stations_delete = ['VSBRI', 'VSBAS', 'VSTSN', 'VSTRI', 'VSCHY', 'GSB']
    precipitation_valais = precipitation_valais[~precipitation_valais['station'].isin(stations_delete)]


    # Merge datasets
    merged_dataset = precipitation_valais.copy()  # Start with precipitation

    merged_dataset = pd.merge(merged_dataset, 
                            moisture_valais,
                            how='left', on=['time', 'station'])
    merged_dataset = pd.merge(merged_dataset, 
                            pression_valais,
                            how='left', on=['time', 'station'])
    merged_dataset = pd.merge(merged_dataset, 
                            temperature_valais,
                            how='left', on=['time', 'station'])
    merged_dataset = pd.merge(merged_dataset, 
                            wind_valais,
                            how='left', on=['time', 'station'])

    # Save to CSV
    merged_dataset.to_csv('data/filtered/merged_valais.csv', index=False)


def what_var():
    varlist=['precip', 'temperature', 'pression', 'moisture', 'North', 'East']
    termlist = ['Precipitation', 'Temperature', 'Pression', 'Humidite', 'Vent', 'Vent']

    stations = pd.read_csv("valais_special/data/station_list.csv", delimiter=';')
    stations_clean = pd.read_csv('data/clean/valais_stations.csv')
    valais_stations = stations[stations['Canton']=="VS"]
    valais_stations = valais_stations[['Abr.','Mesures']]

    for var, term in zip(varlist, termlist):
        valais_stations[var] = valais_stations['Mesures'].str.contains(term)
    
    df = valais_stations.drop('Mesures', axis=1).rename(columns={'Abr.':'station'})
    merged = stations_clean.merge(df, how='left', on='station')

    merged.to_csv('data/clean/valais_stations_2.csv', index=False)



if __name__ == "__main__":
    what_var()