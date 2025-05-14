import os
import shutil
import pandas as pd
import logging
import tempfile

df_start = pd.read_csv('../data/clean/valais_clean.csv')

df_start['ds'] = pd.to_datetime(df_start['time'], format='%Y%m%d%H%M')
df_start = df_start.rename(columns={
    'station':      'unique_id',
    'precip':       'y',
    'East':         'east_wind',
    'North':        'north_wind'
})

df_start = df_start[['unique_id', 'ds', 'y',
         'temperature', 'pressure', 'moisture',
         'east_wind', 'north_wind']]


#first, we lower the granularity of our observations. We will go from 10 minute totals to 1 hour totals
df_hourly = (
    df_start
    .groupby('unique_id')
    .resample('h', on='ds')
    .agg({'y': 'sum',
          'east_wind': 'mean',
          'north_wind': 'mean',
          'moisture': 'mean',
          'pressure': 'mean',
          'temperature': 'mean'
          })
    .reset_index()
)

df_hourly.to_csv('../data/clean/hourly_data.csv', index=False)