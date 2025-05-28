import pandas as pd
from itertools import repeat
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def take_away(var, lame_stations, station_list):
    df = pd.read_csv(f"data/processed/{var}.csv")    
    #remove stations available for only one var
    df = df[~df['station'].isin(lame_stations)]
    #remove stations that don't have precipitation
    df = df[df['station'].isin(station_list['station'])]
    #Thats all thankx for watching
    df.to_csv(f"data/filtered/{var}_filter.csv", index=False)


def main():
    other_vars = ["moisture", "precipitation", "pression", "temperature"]
    varlist = ["wind_vectors"]
    lame_stations = ['DLALH', 'KSHIG', 'DLALM', 'PRE', 'AEG'] #stations only present in one variable
    station_list = pd.read_csv("data/clean/valais_stations.csv") #list of precipitation stations

    with ProcessPoolExecutor() as executor: #mutithreading 
        list(tqdm(executor.map(take_away, varlist, repeat(lame_stations), repeat(station_list)),
                   total=len(varlist),
                   desc= "Processing variables"))    


if __name__ == "__main__":
    main()