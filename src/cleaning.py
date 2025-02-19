import glob
import pandas as pd
import os
import re

#return a list of files matching the pattern
def get_text_files(pattern):
    return glob.glob(pattern)

#get the name and code for each station
def extract_station_data(file_path):
    station_data = []
    with open(file_path, 'r', encoding="windows-1252") as file:
        lines = file.readlines()
    
    station_section = False
    for line in lines:
        if line.strip().startswith("Stations"):  # Identify start of station section
            station_section = True
            continue
        
        if station_section:
            match = re.match(r"^(\S+)\s+(.+?)\s+rre150z0.+?(\d+)\.\d+'/\d+\.\d+'\s+(\d+)/(\d+)\s+(\d+)", line)

            if match:
                station_code = match.group(1).strip()
                station_name = match.group(2).strip()
                east_location = match.group(3).strip()
                north_location = match.group(4).strip()
                altitude = match.group(5).strip()
                station_data.append((station_code, station_name, east_location, north_location, altitude))
    
    return station_data

#aggregate every station name
def process_legend_files(pattern="data/raw/weather_stations/*_legend.txt", output_file="data/clean/stations.csv"):
    files = get_text_files(pattern)
    all_data = []
    
    for file_path in files:
        data = extract_station_data(file_path)
        all_data.extend(data)
    
    df = pd.DataFrame(all_data, columns=["code", "name", "east", "north", "altitude"])
    df.to_csv(output_file, index=False)


#read the csv of one file
def get_df(filepath):
    df = pd.read_csv(filepath,
                    delimiter=';',
                    skiprows=2,
                    header=0,
                    dtype=str,
                    encoding='utf-8',
                    na_values=['-'])
    #remove unuseful headers
    df = df[df['stn'] != 'stn']

    #convert to correct type
    df['stn'] = df['stn'].astype(str)
    df['time'] = df['time'].astype(str)
    df['rre150z0'] = pd.to_numeric(df['rre150z0'], errors='coerce')

    df = df.rename(columns={'stn':'station', 'rre150z0': 'precip'})

    return df


def process_data_files(pattern='data/raw/weather_stations/*_data.txt', output_file = "data/clean/station_precipitation.csv"):
    files = get_text_files(pattern)
    all_data = pd.DataFrame()

    for file_path in files:
        all_data = pd.concat([all_data, get_df(file_path)], ignore_index=False)
    
    all_data.to_csv(output_file, index=False)


def main():
    process_legend_files()
    process_data_files()



if __name__ == "__main__":
    main()