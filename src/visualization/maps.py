import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import xyzservices.providers as xyz

# Read stations file
file = 'data/clean/stations.csv'
df = pd.read_csv(file)

df['east'] = df['east'] + 2000000  
df['north'] = df['north'] + 1000000 

# Define coordinate system
crs_2056 = "EPSG:2056"

geometry = [Point(xy) for xy in zip(df['east'], df['north'])]

gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_2056)

fig, ax = plt.subplots(figsize=(10, 10))
gdf_points.plot(ax=ax, color='white', edgecolor='black', alpha=0.8, markersize=90)

swiss_basemap = xyz.SwissFederalGeoportal.NationalMapColor
ctx.add_basemap(ax, crs=crs_2056, source=swiss_basemap)
plt.axis('off')
plt.savefig('report/figures/stations_map.png', dpi=300)