#Plotting timeseries for the whole duration


#import packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from functools import reduce
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.seasonal import STL

#precip = pd.read_csv("data/filtered/precipitation_filter.csv")
#moisture = pd.read_csv("data/filtered/moisture_filter.csv")
#pression = pd.read_csv("data/filtered/pression_filter.csv")
#temperature = pd.read_csv("data/filtered/temperature_filter.csv")
#wind = pd.read_csv("data/filtered/wind_vectors_filter.csv")

everything = pd.read_csv('data/filtered/merged_valais.csv')

def all_ts(station='SIO'):
    # Filter data for the given station
    sion_data = everything[everything['station'] == station].copy()

    # Compute wind speed
    sion_data['speed'] = np.sqrt(sion_data['North']**2 + sion_data['East']**2)
    sion_data['time'] = pd.to_datetime(sion_data['time'], format='%Y%m%d%H%M')
    sion_data = sion_data.set_index('time')

    # Compute monthly averages
    monthly_avg = sion_data[['precip', 'speed', 'moisture', 'pression', 'temperature']].resample('ME').mean()

    # Create figure with shared x-axis
    fig, axes = plt.subplots(5, 1, figsize=(8, 6), sharex=True)

    vars=['precip', 'speed', 'moisture', 'pression', 'temperature']
    labels=['Precipitation', 'Wind Speed', 'Moisture', 'Pressure', 'Temperature']

    for i, (ax, var, label) in enumerate(zip(axes, vars, labels)):
        ax.plot(sion_data.index, sion_data[var], label=label, alpha=0.5)
        ax.plot(monthly_avg.index, monthly_avg[var], label="monthly average", color='red')
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time", fontsize=13)
    plt.suptitle('Timeseries of available variables with monthly averages', fontsize=13)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('report/figures/all_vars_sion.pdf')
    plt.show()

def all_stations_ts():
    valais_stations = ["SIO","VSSIE", "ZER", "VSDER", "VIS"]
    station_name = ["Sion", "Sierre", "Zermatt", "Derborence", "Visp"]

    precip_filter = everything[everything['station'].isin(valais_stations)]
    precip_wide = precip_filter.pivot(index='time', columns='station', values='precip')
    
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')

    y_max = precip_wide.max().max()

    fig, axes = plt.subplots(len(valais_stations), 1, figsize=(12, 6), sharex=True)

    for ax, station, name in zip(axes, valais_stations, station_name):
        ax.plot(precip_wide.index, precip_wide[station], label=name, color='darkblue')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7))) 
        ax.set_ylim(0, y_max) 
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.grid(True, which='both', linestyle="--", alpha=0.5)

    # Set common labels
    axes[2].set_ylabel("Precipitation (mm)", fontsize=13)
    axes[-1].set_xlabel("Time", fontsize=13)
    fig.suptitle("Precipitation Time Series for Selected Stations in Valais", fontsize=13)

    plt.tight_layout()  
    plt.savefig("report/figures/valais_stations_ts.pdf", dpi=100)
    plt.show()
  
def monthly_vd_ts():
    valais_stations = ["SIO","VSSIE", "ZER", "VSDER", "VIS"]
    station_name = ["Sion", "Sierre", "Zermatt", "Derborence", "Visp"]

    precip_filter = everything[everything['station'].isin(valais_stations)]
    precip_wide = precip_filter.pivot(index='time', columns='station', values='precip')
    
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')

    monthly_precip = precip_wide.resample('ME').sum()
    bar_width = 0.19
    x = np.arange(len(monthly_precip.index))

    fig, ax = plt.subplots(figsize=(12,6))

    for i, (station, name) in enumerate(zip(valais_stations, station_name)):
        ax.bar(x + i* bar_width, monthly_precip[station], width=bar_width, label=name)

    ax.set_xticks(x + (bar_width * (len(valais_stations) / 2)))
    ax.set_xticklabels(monthly_precip.index.strftime('%Y-%m'), rotation=45)    
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Precipitation (mm)", fontsize=13)
    ax.set_title("Monthly precipitation total for selected stations", fontsize=13)
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("report/figures/monthly_valais_bar.pdf")
    plt.show()
   
def cumsum_vd_ts():
    valais_stations = ["SIO","VSSIE", "ZER", "VSDER", "VIS", 'median']
    station_name = ["Sion", "Sierre", "Zermatt", "Derborence", "Visp", 'Median value']

    precip_wide = everything.pivot(index='time', columns='station', values='precip')
    precip_wide.index = pd.to_datetime(precip_wide.index, format='%Y%m%d%H%M')
    monthly_precip = precip_wide.cumsum()

    monthly_precip['median'] = monthly_precip.median(axis=1)

    fig, ax = plt.subplots(figsize=(12,6))

    for station, name in zip(valais_stations, station_name):
        if station =='median':
            ax.plot(monthly_precip.index, monthly_precip.loc[:, station], label=name, linewidth=2, color='darkblue')
        else:
            ax.plot(monthly_precip.index, monthly_precip.loc[:, station], label=name, alpha=0.8)
    
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Precipitation (mm)", fontsize=13)
    ax.set_title("Cumulative precipitation sum for selected stations with median Valais value", fontsize=13)
    ax.legend(loc="upper left", ncol=1, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("report/figures/cumsum_valais_ts.pdf")
    plt.show()

def cumsum_periods(station='SIO'):
    precip_filter = everything[everything['station'] == station].copy()
    precip_filter.index = pd.to_datetime(precip_filter['time'], format='%Y%m%d%H%M')

    # Create quarter and year columns
    precip_filter['quarter'] = precip_filter.index.to_period('Q')
    precip_filter['year'] = precip_filter.index.to_period('Y')

    # Compute cumulative precipitation within each quarter
    precip_filter['cumsum_precip'] = precip_filter.groupby(['year', 'quarter'])['precip'].cumsum()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    months = pd.date_range(start='1/1/2018', periods=12, freq='ME')
    # Iterate through quarters (1 to 4)
    for i, q in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        quarter_data = precip_filter[precip_filter.index.quarter == q]

        # Plot cumulative precipitation for each year within the quarter
        for year in quarter_data['year'].unique():
            yearly_data = quarter_data[quarter_data['year'] == year]
            yearly_data = yearly_data.reset_index(drop=True)
            ax.plot(yearly_data.index, yearly_data['cumsum_precip'], label=str(year), alpha=0.7)
            


            # Formatting each subplot
            ax.set_xticklabels([])
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_title(f"Q{i + 1}")


        # Set x-axis label only on bottom row
    for ax in [axes[2], axes[3]]:
        ax.set_xlabel("Time")

    # Set y-axis label only on left column
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel("Cumulative Precipitation (mm)")

    # Main figure title and axis labels
    fig.suptitle(f"Quarterly Cumulative Precipitation - Sion", fontsize=15)

    plt.tight_layout()
    plt.savefig(f"report/figures/cumsum_quarter_sion.pdf")
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

def correlation(station='SIO'):
    sion_data = everything[everything['station'] == station].drop(columns=['station', 'time'])

    correlation_matrix = sion_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Variables - Sion", fontsize=13)
    plt.savefig('report/figures/correlation_matrix_sion.pdf')
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

def valais_precip_corr(): 
    everything.index = pd.to_datetime(everything['time'], format='%Y%m%d%H%M')
    #only take stations that have more than precipitation
    stations = ['GRC', 'BLA', 'JUN', 'MVE', 'ZER',
                'ULR', 'SIO', 'GSB', 'VIS', 'EVO',
                'BIN', 'MAR', 'BOU', 'MTE', 'MOB',
                'SIM', 'ATT', 'EVI', 'GOR', 'EGH']
    filtered = everything[everything['station'].isin(stations)]
    precip_wide = filtered.pivot(index='time', columns='station', values='precip')    

    correlation_matrix = precip_wide.corr()
    plt.figure(figsize=(13,10))
    sns.heatmap(correlation_matrix, annot=True, fmt=':^.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.xlabel("Stations", fontsize=13)
    plt.ylabel("Stations", fontsize=13)
    plt.title("Correlation matrix of precipitation for stations in Valais", fontsize=13)
    plt.savefig('report/figures/correlation_matrix_valais.pdf')
    plt.show()

def correlation_n_distance():
    stations = pd.read_csv("data/clean/valais_stations.csv")
    stations = stations[stations['station'].isin(everything['station'].unique())] #very ugly code

    stations_np = stations[['east', 'north']].to_numpy()

    distances = squareform(pdist(stations_np, metric='euclidean'))
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances.flatten()

    precip_copy = everything[['time', 'station', 'precip']].copy()  
    precip_copy.index = pd.to_datetime(precip_copy['time'], format='%Y%m%d%H%M')
    precip_wide = precip_copy.pivot(columns='station', values='precip')

    correlation_matrix = precip_wide.corr()
    corr_np = np.array(correlation_matrix)
    corr_np = corr_np[np.triu_indices_from(corr_np, k=1)]
    corr_np = corr_np.flatten()

    #plot the scatter
    fig, ax = plt.subplots(figsize=(10,8))

    ax.scatter(distances, corr_np, facecolors='none', edgecolors='black', s=50)

    # Set labels and title
    ax.set_xlabel("Distance between stations (m)", fontsize=13)
    ax.set_ylabel("Precipitation correlation", fontsize=13)
    ax.set_title("Correlation between the distance and the precipitation correlation", fontsize=13)

    # Tight layout and show plot
    plt.tight_layout()
    plt.savefig("report/figures/scatter_distance_corr_valais.pdf")
    plt.show()

def altitude_precip(): 
    precip_copy = everything.copy().drop(columns=['time'], errors='ignore')

    mean_precip = precip_copy.groupby('station')['precip'].sum().reset_index()
    stations = pd.read_csv("data/filtered/stations.csv")
    merged_df = stations[['station', 'altitude']].merge(mean_precip, on='station', how='inner')
    
    #plot the scatter
    fig, ax = plt.subplots(figsize=(10,8))
 
    ax.scatter(data=merged_df, x='precip', y='altitude', facecolors='none', edgecolors='black', s=50)
    print()
    # Set labels and title
    ax.set_xlabel("Total Precipitations (mm)", fontsize=13)
    ax.set_ylabel("Altitude", fontsize=13)
    ax.set_title("Precipitation totals depending on the altitude", fontsize=13)

    # Tight layout and show plot
    plt.tight_layout()
    plt.savefig("report/figures/sum_precip_altitude_vs.pdf")
    plt.show()

def main():
    altitude_precip()


if __name__ == '__main__':
    main()