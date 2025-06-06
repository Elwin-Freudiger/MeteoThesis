library(tidyverse)
library(rayshader)
library(terra)
library(bfsMaps)
library(sf)
library(grid)
library(png)

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

wind_vectors_filter <- read_csv("data/filtered/merged_valais.csv")
# --- Load and process elevation raster ---
raster <- rast("data/raw/altitude/DHM200.asc")
crs(raster) <- "EPSG:21781"  # LV03

# Reproject to LV95
raster_95 <- project(raster, "EPSG:2056")

# Load Valais canton border
options(bfsMaps.base = "data/swiss_boundaries/mb-x-00.07.01.01-25/2025_GEOM_TK")
valais_grenz <- GetMap("kant.map")
valais_polygon <- valais_grenz %>% filter(name == "Valais")

# Project polygon to raster CRS and crop/mask
valais_proj <- st_transform(valais_polygon$geometry, crs(raster_95))
valais_vect <- vect(valais_proj)

valais_map <- raster_95 %>%
  crop(valais_vect) %>%
  mask(valais_vect)

# Convert elevation raster to matrix
elmat <- matrix(values(valais_map),
                nrow = nrow(valais_map),
                ncol = ncol(valais_map),
                byrow = TRUE)

elmat <- t(elmat)
# flip columns (mirror on vertical axis)


hillshade <- sphere_shade(elmat, texture = "desert", sunangle = 0)

plot_map(hillshade)  # or use in 3D with plot_3d



# Create shaded relief image
hillshade <- sphere_shade(elmat, texture = "desert", sunangle = 0)

# Get extent for ggplot overlay
ext <- ext(valais_map)
xmin <- ext[1]; xmax <- ext[2]
ymin <- ext[3]; ymax <- ext[4]

# --- Load and process wind vector data ---

# Compute average wind per station and convert to LV95 points
wind_summary <- wind_vectors_filter %>%
  select(station, East, North, east, north) %>%
  mutate(
    easting = east + 2000000,
    northing = north + 1000000
  ) %>%
  drop_na(North, East, easting, northing) %>%
  group_by(station, easting, northing) %>%
  summarise(
    North = mean(North, na.rm = TRUE),
    East = mean(East, na.rm = TRUE)
  ) %>%
  mutate(
    angle = atan2(East, North),
    magnitude = sqrt(North^2 + East^2)
  ) %>%
  ungroup()

arrow_length <-  5000
wind_summary <- wind_summary %>%
  mutate(
    xend = easting + arrow_length * sin(angle),  # East is X axis
    yend = northing + arrow_length * cos(angle)  # North is Y axis
  )

ggplot() +
  annotation_raster(
    raster = hillshade,
    xmin = xmin, xmax = xmax,
    ymin = ymin, ymax = ymax
  ) +
  # Arrows: white fill
  geom_segment(
    data = wind_summary,
    aes(x = easting, y = northing, xend = xend, yend = yend),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "black", linewidth = 1.2
  ) +
  # Station labels, non-overlapping
  geom_text_repel(
    data = wind_summary,
    aes(x = easting, y = northing, label = station),
    size = 3,
    color = "black",
    box.padding = 0.3,
    point.padding = 0.5,
    segment.color = "black"
  ) +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) +
  theme_void()




