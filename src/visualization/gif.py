import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import xyzservices.providers as xyz
from PIL import Image
import io

# Read stations file
file = 'data/filtered/stations.csv'
df_stations = pd.read_csv(file)

# Read temperature file
df_temp = pd.read_csv('data/filtered/temperature_filter.csv')

# Convert time and filter data
df_temp['time'] = pd.to_datetime(df_temp['time'], format='%Y%m%d%H%M')
df_temp = df_temp[(df_temp['time'] >= '2021-04-22') & (df_temp['time'] < '2021-04-23')]

# Define CRS and temperature range
crs_2056 = "EPSG:2056"
min_val = df_temp['temperature'].min()
max_val = df_temp['temperature'].max()

# Prepare frames for GIF
frames = []

for timestamp in df_temp['time'].unique():
    filtered_df = df_temp[df_temp['time'] == timestamp]

    # Merge stations with temperature data
    merged_df = df_stations.merge(filtered_df, on='station', how='inner')

    # Create geometry
    geometry = [Point(xy) for xy in zip(merged_df['east'], merged_df['north'])]
    gdf_points = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=crs_2056)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_points.plot(
        ax=ax,
        column='temperature',
        cmap='coolwarm',
        vmin=min_val,
        vmax=max_val,
        edgecolor='black',
        markersize=90,
        legend=True,
        missing_kwds={
            'color': 'gray',
            'label': 'No data',
            'alpha':0.7
        }
    )

    # Add basemap
    swiss_basemap = xyz.SwissFederalGeoportal.NationalMapGrey
    ctx.add_basemap(ax, crs=crs_2056, source=swiss_basemap)
    plt.axis('off')

    # Format title with a cleaner timestamp
    plt.title(f"Temperature in Switzerland on {timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))

    plt.close(fig)

# Save all frames as a GIF
frames[0].save('report/figures/gifs/temperature_animation_nomiss.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)