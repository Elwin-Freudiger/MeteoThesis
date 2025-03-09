import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import xyzservices.providers as xyz

# Read stations file
file = 'data/filtered/stations.csv'
df = pd.read_csv(file)

#newfile
df_temp = pd.read_csv('data/filtered/temperature_filter.csv')

df_temp['time'] = pd.to_datetime(df_temp['time'], format='%Y%m%d%H%M')
filtered_df = df_temp[df_temp['time'] == '2019-01-15']

#merge both
merged_df = df.merge(filtered_df, on='station', how='inner')


# Define coordinate system
crs_2056 = "EPSG:2056"

geometry = [Point(xy) for xy in zip(merged_df['east'], merged_df['north'])]

gdf_points = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=crs_2056)

fig, ax = plt.subplots(figsize=(10, 10))
gdf_points.plot(ax=ax,
                column='temperature',
                cmap='coolwarm',
                edgecolor='black',
                markersize=90,
                legend=True)

swiss_basemap = xyz.SwissFederalGeoportal.NationalMapColor
ctx.add_basemap(ax, crs=crs_2056, source=swiss_basemap)
plt.axis('off')
plt.show()