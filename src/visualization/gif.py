import geopandas as gpd
import pandas as pd
import numpy as np
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
df_temp = pd.read_csv('data/filtered/wind_vectors_filter.csv')

# Convert time and filter data
df_temp['time'] = pd.to_datetime(df_temp['time'], format='%Y%m%d%H%M')
df_temp = df_temp[(df_temp['time'] >= '2020-12-24') & (df_temp['time'] < '2020-12-25')]

# Define CRS and temperature range
crs_2056 = "EPSG:2056"

# Prepare frames for GIF
frames = []

for timestamp in df_temp['time'].unique():
    filtered_df = df_temp[df_temp['time'] == timestamp]

    # Merge stations with temperature data
    merged_df = df_stations.merge(filtered_df, on='station', how='inner')

    # Create geometry
    geometry = [Point(xy) for xy in zip(merged_df['east'], merged_df['north'])]
    gdf_points = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=crs_2056)


    gdf_points['speed'] = abs(np.sqrt(gdf_points['North']**2 + gdf_points['East']**2))
    gdf_points['normed_East'] = gdf_points['East']/gdf_points['speed']
    gdf_points['normed_North'] = gdf_points['North']/gdf_points['speed']


    big_speed = gdf_points[gdf_points['speed'] > 1.5]
    small_speed = gdf_points[gdf_points['speed'] <= 1.5]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    small_speed.plot(
        ax=ax,
        markersize=90,
        color='white',
        edgecolor='black'
    )

    ax.quiver(
        big_speed['east'],
        big_speed['north'],
        big_speed['normed_East'],
        big_speed['normed_North'],
        big_speed['speed'],
        cmap='RdPu',
        pivot='middle'
    )

    # Add basemap
    swiss_basemap = xyz.SwissFederalGeoportal.NationalMapGrey
    ctx.add_basemap(ax, crs=crs_2056, source=swiss_basemap)
    plt.axis('off')

    # Format title with a cleaner timestamp
    plt.title(f"Wind in Switzerland on {timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))

    plt.close(fig)

# Save all frames as a GIF
frames[0].save('report/figures/gifs/wind_animation.gif', save_all=True, append_images=frames[1:], duration=150, loop=0)