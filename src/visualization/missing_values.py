import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def find_missing_chunks(df, var):
    # Create a column to mark where the data is missing (True if NaN)
    df['missing'] = df[var].isna()
    
    # Find the chunks of missing data
    missing_chunks = []
    for station in tqdm(df['station'].unique(), desc=f"Processing stations in {var}"):
        station_data = df[df['station'] == station]
        
        counter = 0
        for i, row in station_data.iterrows():
            if row['missing']:
                counter += 1
            elif not row['missing'] and counter>0:
                missing_chunks.append((station, counter))
                counter = 0
        if counter > 0:
            missing_chunks.append((station, counter))

    return missing_chunks


def viz(var):
    # Read the data
    df = pd.read_csv(f"data/processed/{var}.csv")

    # Get the missing data chunks
    missing_chunks = find_missing_chunks(df, var)

    # Plot the missing chunks if there are any
    print(f"{var} has {len(missing_chunks)} missing chunks")

    plt.figure(figsize=(10, 6))
    for station, counter in missing_chunks:
        plt.scatter(station, counter*10, alpha=0.5)
    
    plt.title(f"Missing Data Chunks for {var}")
    plt.xlabel("Station")
    plt.ylabel("Duration of missing data (minutes)")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"report/figures/missing_chunks_{var}.png")
    plt.close()

def main():
    varlist = ["moisture", "precipitation", "pression", "temperature", "wind_vectors"]

    with ProcessPoolExecutor() as executor:
        executor.map(viz, varlist)

if __name__ == "__main__":
    main()