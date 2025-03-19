#Plotting timeseries for the whole duration


#import packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from functools import reduce
from scipy.spatial.distance import pdist, squareform


precip = pd.read_csv("data/filtered/precipitation_filter.csv")
#moisture = pd.read_csv("data/filtered/moisture_filter.csv")
#pression = pd.read_csv("data/filtered/pression_filter.csv")
#temperature = pd.read_csv("data/filtered/temperature_filter.csv")
#wind = pd.read_csv("data/filtered/wind_vectors_filter.csv")

def all_ts(station='PUY'):
    # Filter data for the given station
    precip_filtered = precip[precip['station'] == station]
    moisture_filtered = moisture[moisture['station'] == station]
    pression_filtered = pression[pression['station'] == station]
    temperature_filtered = temperature[temperature['station'] == station]
    wind_filtered = wind[wind['station'] == station]

    # Compute wind speed
    wind_filtered['speed'] = np.sqrt(wind_filtered['North']**2 + wind_filtered['East']**2)

    # Create figure with shared x-axis
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Define variable names and datasets
    datasets = [precip_filtered, moisture_filtered, pression_filtered, temperature_filtered, wind_filtered]
    labels = ["precipitation", "moisture", "pression", "temperature", "speed"]

    time = pd.date_range(start='2019-01-01 00:00', end='2023-12-31 23:50', freq='10min')

    for i, (data, label) in enumerate(zip(datasets, labels)):
        axes[i].plot(time, data.loc[:, label], label=label)  # Assuming the third column is the relevant variable
        axes[i].set_ylabel(label)
        axes[i].legend()

    axes[-1].set_xlabel("Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('report/figures/all_vars_ts.pdf')

#plotting the temperature for each station
def all_stations_ts():

    vaud_stations = ["LSN","PUY", "VEV", "COS", "AVA"]
    station_name = ["Lausanne", "Pully", "Vevey", "Cossonay", "Les Avants"]

    precip_filter = precip[precip['station'].isin(vaud_stations)]
    precip_wide = precip_filter.pivot(index='time', columns='station', values='precipitation')
    
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')

    y_max = precip_wide.max().max()

    fig, axes = plt.subplots(len(vaud_stations), 1, figsize=(12, 6), sharex=True)

    # Iterate correctly using zip
    for i, (station, name) in enumerate(zip(vaud_stations, station_name)):
        axes[i].plot(precip_wide.index, precip_wide[station], label=name, color='darkblue')
        axes[i].set_ylim(0, y_max) 
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Set common labels
    axes[2].set_ylabel("Precipitation (mm)")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Precipitation Time Series for Selected Stations Around Lausanne")

    plt.tight_layout()
    plt.savefig("report/figures/lausanne_stations_ts.pdf")
    plt.show()

    
def monthly_vd_ts():
    vaud_stations = ["LSN","PUY", "VEV", "COS", "AVA"]
    station_name = ["Lausanne", "Pully", "Vevey", "Cossonay", "Les Avants"]

    precip_filter = precip[precip['station'].isin(vaud_stations)]
    precip_wide = precip_filter.pivot(index='time', columns='station', values='precipitation')
    
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')

    monthly_precip = precip_wide.resample('W').sum()

    fig, ax = plt.subplots(figsize=(12,6))

    for station, name in zip(vaud_stations, station_name):
        ax.plot(monthly_precip.index, monthly_precip.loc[:, station], label=name, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("Monthly precipitation total for selected stations")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("report/figures/monthly_lausanne_ts.pdf")
    plt.show()

    
def cumsum_vd_ts():
    vaud_stations = ["LSN","PUY", "VEV", "COS", "AVA"]
    station_name = ["Lausanne", "Pully", "Vevey", "Cossonay", "Les Avants"]

    precip_filter = precip[precip['station'].isin(vaud_stations)]
    precip_wide = precip_filter.pivot(index='time', columns='station', values='precipitation')
    
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')

    monthly_precip = precip_wide.cumsum()

    fig, ax = plt.subplots(figsize=(12,6))

    for station, name in zip(vaud_stations, station_name):
        ax.plot(monthly_precip.index, monthly_precip.loc[:, station], label=name)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("Cumulative precipitation sum for Selected Stations around Lausanne")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("report/figures/cumsum_vd_ts.pdf")
    plt.show()


def cumsum_periods(station='PUY'):
    precip_filter = precip[precip['station']==station]
    
    precip_filter.index = pd.to_datetime(precip_filter['time'], format='%Y%m%d%H%M')

    precip_filter_period = precip_filter.groupby(precip_filter.index.to_period('Q'))['precipitation'].cumsum()

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(precip_filter_period.index, precip_filter_period, label=station, color='darkblue')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("Quarterly cumulative precipitation over Pully")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("report/figures/cumsum_quarter_pully.pdf")
    plt.show()

def overlap_cumsum(station='PUY'):
    precip_filter = precip[precip['station']==station]
    
    precip_filter.index = pd.to_datetime(precip_filter['time'], format='%Y%m%d%H%M')

    years = precip_filter.index.year.unique()
    fig, ax = plt.subplots(figsize=(12,6))

    for i, year in enumerate(years):
        precip_year = precip_filter[precip_filter.index.year == year]
        cumsum = precip_year['precipitation'].cumsum
        ax.plot(cumsum.index, cumsum, color=cmap(i))
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("Yearly Cumulative precipitation over Pully")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.savefig("report/figures/overlap_cumsum.pdf")

def temp_overlap(station='PUY'):
    # Filter data for the selected station
    temp_filter = temperature[temperature['station'] == station]

    # Convert 'time' column to datetime
    temp_filter['time'] = pd.to_datetime(temp_filter['time'], format='%Y%m%d%H%M')

    # Extract the year and reset the index to maintain order of data points
    temp_filter['year'] = temp_filter['time'].dt.year
    temp_filter['day_of_year'] = temp_filter['time'].dt.dayofyear  # Day of the year (1 to 365/366)

    # Get the list of unique years in the dataset
    years = temp_filter['year'].unique()

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each year's data, each as a separate line
    for year in years:
        # Filter data for each year
        temp_year = temp_filter[temp_filter['year'] == year]
        
        # Plot the temperature values, using 'day_of_year' on the x-axis
        ax.plot(temp_year['day_of_year'], temp_year['temperature'], label=str(year))

    # Set labels and title
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Temperature Comparison for {station} by Year")

    # Adding legend and grid
    ax.legend(loc="upper left", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Tight layout and show plot
    plt.tight_layout()
    plt.show()
    
    # Save the plot to a file
    #plt.savefig(f"report/figures/monthly_temp_pully.pdf")

#Correlation graph with every variable
def correlation(station='PUY'):
    precip = pd.read_csv("data/filtered/precipitation_filter.csv")
    moisture = pd.read_csv("data/filtered/moisture_filter.csv")
    pression = pd.read_csv("data/filtered/pression_filter.csv")
    temperature = pd.read_csv("data/filtered/temperature_filter.csv")
    wind = pd.read_csv("data/filtered/wind_vectors_filter.csv")

    temp_data = temperature[temperature['station'] == station]
    precip_data = precip[precip['station'] == station]
    wind_data = wind[wind['station'] == station]
    moisture_data = moisture[moisture['station'] == station]
    pression_data = pression[pression['station'] == station]   

    temp_data.index = pd.to_datetime(temp_data['time'], format='%Y%m%d%H%M')
    precip_data.index = pd.to_datetime(precip_data['time'], format='%Y%m%d%H%M')
    wind_data.index = pd.to_datetime(wind_data['time'], format='%Y%m%d%H%M')
    moisture_data.index = pd.to_datetime(moisture_data['time'], format='%Y%m%d%H%M')
    pression_data.index = pd.to_datetime(pression_data['time'], format='%Y%m%d%H%M')

    merged_data = precip_data[['precipitation']].merge(temp_data[['temperature']], left_index=True, right_index=True, how='left')
    merged_data = merged_data.merge(wind_data[['North', 'East']],  left_index=True, right_index=True, how='left')
    merged_data = merged_data.merge(moisture_data[['moisture']],  left_index=True, right_index=True, how='left')
    merged_data = merged_data.merge(pression_data[['pression']],  left_index=True, right_index=True, how='left')

    correlation_matrix = merged_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Correlation Matrix of Variables for Pully")
    plt.savefig('report/figures/correlation_matrix_pully.pdf')
    plt.show()

def station_correlation():
    station_list = ['LSN', 'GVE', 'BER', 'SMA', 'SIO', 'LUG', 'BAS']
    names = ['Lausanne', 'Geneva', 'Bern', 'Züri', 'Sion', 'Lugano', 'Basel']

    precip_copy = precip.copy()  
    precip_copy.index = pd.to_datetime(precip_copy['time'], format='%Y%m%d%H%M')

    precip_filter = precip_copy[precip_copy['station'].isin(station_list)]

    precip_wide = precip_filter.pivot(columns='station', values='precipitation')


    correlation_matrix = precip_wide.corr()

    rename_dict = dict(zip(station_list, names))
    correlation_matrix = correlation_matrix.rename(index=rename_dict, columns=rename_dict)

    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Correlation matrix of precipitation for stations in Switzerland")
    plt.savefig('report/figures/correlation_matrix_precip_CH.pdf')
    plt.show()

def correlation_n_distance():
    station_list = ['AIG', 'BEX', 'BIE', 'FRE', 'CHD', 'CDM', 'COS', 'AUB', 'DOL', 'LSN', 'AVA', 'CHB', 'DIA', 'LON', 'MAH', 'CGI', 'ORO', 'PAY', 'PUY', 'PRE', 'VEV', 'VIT']

    stations = pd.read_csv("data/filtered/stations.csv")
    stations_filter = stations[stations['station'].isin(station_list)]
    stations_np = stations_filter[['east', 'north']].to_numpy()

    distances = squareform(pdist(stations_np, metric='euclidean'))
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances.flatten()

    precip_copy = precip.copy()  
    precip_copy.index = pd.to_datetime(precip_copy['time'], format='%Y%m%d%H%M')

    precip_filter = precip_copy[precip_copy['station'].isin(station_list)]

    precip_wide = precip_filter.pivot(columns='station', values='precipitation')

    correlation_matrix = precip_wide.corr()
    corr_np = np.array(correlation_matrix)
    corr_np = corr_np[np.triu_indices_from(corr_np, k=1)]
    corr_np = corr_np.flatten()

    #plot the scatter
    fig, ax = plt.subplots(figsize=(10,8))

    ax.scatter(distances, corr_np)

    # Set labels and title
    ax.set_xlabel("Distance between stations (m)")
    ax.set_ylabel("Precipitation correlation")
    ax.set_title("Correlation between the distance and the precipitation correlation")

    # Tight layout and show plot
    plt.tight_layout()
    plt.savefig("report/figures/scatter_distance_corr.pdf")
    plt.show()

def altitude_precip(): 
    precip_copy = precip.copy().drop(columns=['time'], errors='ignore')

    mean_precip = precip_copy.groupby('station').sum().reset_index()
    

    stations = pd.read_csv("data/filtered/stations.csv")

    merged_df = stations[['station', 'altitude']].merge(mean_precip, on='station', how='inner')
    
        #plot the scatter
    fig, ax = plt.subplots(figsize=(10,8))

    ax.scatter(data=merged_df, x='precipitation', y='altitude')

    # Set labels and title
    ax.set_xlabel("Total Precipitations (mm)")
    ax.set_ylabel("Altitude")
    ax.set_title("Precipitation totals depending on the altitude")

    # Tight layout and show plot
    plt.tight_layout()
    plt.savefig("report/figures/sum_precip_altitude.pdf")
    plt.show()

    

def main():
    station_correlation()


if __name__ == '__main__':
    main()