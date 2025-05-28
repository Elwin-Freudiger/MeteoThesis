library(tidyverse)
library(rayshader)
library(terra)
library(bfsMaps)
library(sf)

# Load raster
raster <- rast("data/raw/altitude/DHM200.asc")
crs(raster) <- "EPSG:21781"

raster_95 <-  project(raster, "EPSG:2056")


# Set bfsMaps base path and load national boundary
options(bfsMaps.base ="data/swiss_boundaries/mb-x-00.07.01.01-25/2025_GEOM_TK")
national_grenz <- GetMap("ch.map")
national_polygon <- national_grenz$geometry


crs(raster)

national_proj <- st_transform(national_polygon, crs(raster_95))

# Convert to terra vector
national_vect <- vect(national_proj)

# Crop and mask
trimmed_map <- raster_95 %>%
  crop(national_vect) %>%
  mask(national_vect)

plot(trimmed_map)

# Convert to matrix for rayshader
elmat <- matrix(
  values(trimmed_map),
  nrow = nrow(trimmed_map),
  ncol = ncol(trimmed_map),
  byrow = TRUE
)

elmat_rotated <- t(elmat)[, nrow(elmat):1]
elmat_rotated <- elmat_rotated[, ncol(elmat_rotated):1]

# Rayshader plot
elmat_rotated %>%
  sphere_shade(texture = "imhof2") %>%
  plot_map()

render_snapshot()

  
#add water
lakes_sf <- st_read("data/swiss_boundaries/mb-x-00.07.01.01-25/2025_GEOM_TK/00_TOPO/K4_seenyyyymmdd")
water_proj <- st_transform(water_sf, crs(trimmed_map))

blank_raster <- trimmed_map
values(blank_raster) <- NA  # Set all to NA

# Rasterize water shapes onto the blank raster
water_raster <- rasterize(vect(water_proj), blank_raster, field=1, background=NA)

# Convert water_raster to matrix
water_mat <- matrix(values(water_raster),
                    nrow = nrow(water_raster),
                    ncol = ncol(water_raster),
                    byrow = TRUE)



rotate_90_clockwise <- function(mat) {
  tmat <- t(mat)                   # Step 1: transpose
  tmat[, ncol(tmat):1]            # Step 2: flip columns (Y-axis mirror)
}



# Convert to logical matrix (TRUE = water)
water_mat_rotated <- rotate_90_clockwise(water_mat)
water_mat_rotated <- water_mat_rotated[nrow(water_mat_rotated):1, ncol(water_mat_rotated):1]
water_mat_logical <- !is.na(water_mat_rotated)


elmat_rotated %>%
  sphere_shade(texture = "imhof2") %>% 
  add_water(water_mat_logical, color = "lightblue") %>% 
  plot_3d(elmat, zscale = 10, fov = 0, theta = 45, zoom = 1, phi = 45, windowsize = c(1000, 800))
Sys.sleep(0.2)
render_snapshot()




#only plot Valais
