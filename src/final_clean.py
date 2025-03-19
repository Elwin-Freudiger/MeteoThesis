#let's start by removing the VSBRI from our observations

import pandas as pd
from functools import reduce


precip = pd.read_csv("data/filtered/precipitation_filter.csv")
moisture = pd.read_csv("data/filtered/moisture_filter.csv")
pression = pd.read_csv("data/filtered/pression_filter.csv")
temperature = pd.read_csv("data/filtered/temperature_filter.csv")
wind = pd.read_csv("data/filtered/wind_vectors_filter.csv")
vars = [precip, moisture, pression, temperature, wind]