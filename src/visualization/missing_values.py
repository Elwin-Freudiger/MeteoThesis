import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import KBinsDiscretizer

def find_missing_chunks(var):
    #open dataframe
    df = pd.read_csv(f"data/filtered/{var}_filter.csv")

    if var == "wind_vectors":
        missing_chunks = []
        for component in ['North', 'East']:
            df['missing'] = df[component].isna()

            for station, station_data in tqdm(df.groupby('station'), desc=f"Processing stations in {var}"):
                # Identify where missing status changes
                change_points = station_data['missing'].ne(station_data['missing'].shift())
                groups = station_data.groupby(change_points.cumsum())
                
                for _, group in groups:
                    if group['missing'].iloc[0]:  # Only take the missing chunks
                        missing_chunks.append((station, len(group), component))

            return missing_chunks

    # Mark where data is missing
    df['missing'] = df[var].isna()
    
    # Group by station and find lengths of missing data chunks
    missing_chunks = []
    for station, station_data in tqdm(df.groupby('station'), desc=f"Processing stations in {var}"):
        # Identify where missing status changes
        change_points = station_data['missing'].ne(station_data['missing'].shift())
        groups = station_data.groupby(change_points.cumsum())
        
        for _, group in groups:
            if group['missing'].iloc[0]:  # Only take the missing chunks
                missing_chunks.append((station, len(group), var))

    return missing_chunks

def para_lol():
    varlist = ["moisture", "precipitation", "pression", "temperature", "wind_vectors"]
    all_missing_chunks = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(find_missing_chunks, varlist), total=len(varlist), desc="Visualizing variables"))

    # Combine results from all processes
    for chunks in results:
        all_missing_chunks.extend(chunks)
    # Save combined missing data info to a single CSV
    combined_df = pd.DataFrame(all_missing_chunks, columns=['station', 'missing_duration', 'variable'])
    combined_df.to_csv("data/combined_missing_data.csv", index=False)

def duration_hist():
    miss_df = pd.read_csv('data/combined_missing_data.csv')

    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(miss_df['missing_duration']/6,
            bins = 60,
            color = 'darkblue',
            edgecolor='black')

    #titles
    ax.set_title('Distribution of missing periods by duration')
    ax.set_xlabel('Duration (Hours)')
    ax.set_ylabel('Frequency')
    
    #layout and save
    plt.tight_layout()
    plt.savefig('report/figures/missing/missing_histogram.pdf')
    plt.show()

def categorical_hist():
    miss_df = pd.read_csv('data/combined_missing_data.csv')
    missing_durations = miss_df['missing_duration']

    bins = [0, 1, 6, 12, 24, 168, float("inf")]
    labels = ["<1h", "1-6h", "6-12h", "12-24h", "1-7d", ">7d"]

    durations_reshaped = pd.DataFrame(missing_durations).values.reshape(-1, 1)

    discretizer = KBinsDiscretizer(n_bins=len(bins)-1, encode='ordinal', strategy='uniform')
    binned = discretizer.fit_transform(durations_reshaped).flatten()

    binned_labels = pd.cut(missing_durations, bins=bins, labels=labels)

    category_counts = pd.Series(binned_labels).value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.bar(category_counts.index, category_counts.values,
           color='darkblue',
           edgecolor='black')
    
    ax.set_title('Distribution of missing periods')
    ax.set_xlabel('Duration (hours)')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('report/figures/missing/categorical_missing_duration.pdf')
    plt.show()

def missing_count():
    miss_df = pd.read_csv('data/combined_missing_data.csv')
    station_list = pd.read_csv('data/filtered/stations.csv')

    station_count = miss_df[['station']].value_counts(dropna=False).reset_index()
    station_count.columns = ['station', 'missing_count']

    merged_df = station_list.merge(station_count, how='left', on='station')
    merged_df['missing_count'] = merged_df['missing_count'].fillna(0)
    #create plot
    fig, ax = plt.subplots(figsize=(8,6))

    ax.hist(merged_df['missing_count'],
            bins=10,
            color='darkblue',
            edgecolor='black')
    
    ax.set_title('Distribution of missing period count')
    ax.set_xlabel('Number of missing periods')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('report/figures/missing/missing_count_hist.pdf')
    plt.show()

def missing_count_table():
    miss_df = pd.read_csv('data/combined_missing_data.csv')
    station_list = pd.read_csv('data/filtered/stations.csv')

    station_count = miss_df[['station']].value_counts(dropna=False).reset_index()
    station_count.columns = ['station', 'missing_count']

    merged_df = station_list.merge(station_count, how='left', on='station')
    merged_df['missing_count'] = merged_df['missing_count'].fillna(0)   

    bins = [-1, 0, 1, 5, 10, 100, 1000, float("inf")]
    labels = ["0", "1", "2-5", "6-10", "11-100", "101-1000", ">1000"]

    merged_df['missing_bin'] = pd.cut(merged_df['missing_count'],
                                      bins=bins,
                                      labels=labels,
                                      right=True)

    summary_table = merged_df['missing_bin'].value_counts().sort_index().reset_index()
    summary_table.columns = ['Missing duration', 'Number of stations']

    latex_table = summary_table.to_latex(index=False, caption="Distribution of Missing Periods by Duration", label="tab:missing_periods", column_format='|l|r|', escape=False)

    print(latex_table)

def missing


def main():
    #para_lol()
    #duration_hist()
    #categorical_hist()
    #missing_count()
    #missing_count_table()

if __name__ == "__main__":
    main()
