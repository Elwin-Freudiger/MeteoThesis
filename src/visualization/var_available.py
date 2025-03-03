import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# List of variable names
variables = ['moisture', 'precipitation', 'pression', 'stations', 'temperature', 'wind_vectors']

# Path pattern for CSV files
path_pattern = 'data/processed/{var}.csv'

# Dictionary to hold unique stations per variable
stations_per_var = {}

# Load CSVs and extract unique stations
for var in variables:
    file_path = path_pattern.format(var=var)
    try:
        df = pd.read_csv(file_path)
        unique_stations = df['station'].unique()
        stations_per_var[var] = set(unique_stations)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError:
        print(f"'station' column missing in {file_path}")

# Combine all stations and count appearances
all_stations = [station for stations in stations_per_var.values() for station in stations]
station_counts = Counter(all_stations)

# Number of variables each station appears in
stations_appearance = {station: sum(station in stations for stations in stations_per_var.values()) for station in station_counts}

# Sorting stations by their appearances
sorted_stations = sorted(stations_appearance.items(), key=lambda x: x[1], reverse=True)

# Total count of unique stations
total_stations = len(station_counts)
print(f"Total number of unique stations: {total_stations}")

# Plot station availability
plt.figure(figsize=(10, 6))
plt.barh([s[0] for s in sorted_stations], [s[1] for s in sorted_stations], color='skyblue')
plt.xlabel('Number of Variables Station Appears In')
plt.ylabel('Weather Station')
plt.title('Weather Station Availability Across Variables')
plt.savefig('tests/station_availability.png')

# Insights
total_vars = len(variables)
always_available = [s[0] for s in sorted_stations if s[1] == total_vars]
rarely_available = [s[0] for s in sorted_stations if s[1] == 1]

print(f"Stations available for all variables: {always_available}")
print(f"Stations available for only one variable: {rarely_available}")

# Identify missing variables for stations appearing less than the max
max_appearance = total_vars
for station, count in sorted_stations:
    if count < max_appearance:
        missing_vars = [var for var, stations in stations_per_var.items() if station not in stations]
        print(f"For station {station}, missing variables: {missing_vars}")