import pandas as pd
import numpy as np


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