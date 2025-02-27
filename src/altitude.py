import rasterio
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import h5py

# Load the elevation data from an .asc file
asc_file = "data/raw/altitude/DHM200.asc"

with rasterio.open(asc_file) as src:
    elevation_array = src.read(1) 
    transform = src.transform
    nodata = src.nodata

elevation_array = np.where(elevation_array == nodata, np.nan, elevation_array)

xelev = xr.DataArray(elevation_array,
                     dims=("N", "E"),
                     coords={
                         "N": np.linspace(transform.f, transform.f + transform.e * elevation_array.shape[0], elevation_array.shape[0]),
                         "E": np.linspace(transform.c, transform.c + transform.a * elevation_array.shape[1], elevation_array.shape[1])
                         })

coarsened = xelev.coarsen(E=5, N=5, boundary = "trim").mean()

coarsen_np = coarsened.values

#save it into a hdf5 file
hdf5_file = "data/processed/elevation_grid.h5"

with h5py.File(hdf5_file, "w") as f:
    f.create_dataset("elevation", data=coarsen_np)
    f.create_dataset("E", data=coarsened["E"].values)
    f.create_dataset("N", data=coarsened["N"].values)


plt.figure(figsize=(10, 8))
plt.imshow(coarsened, cmap="terrain", origin="upper")
plt.colorbar(label="Elevation (m)")
plt.title("Elevation Map")
plt.savefig("tests/small_elevation.png")
