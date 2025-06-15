import os
import shutil
import pandas as pd
import numpy as np
import logging
import tempfile

df_hourly = pd.read_csv('data/clean/hourly_data.csv')

index_6 = pd.api.indexers.FixedForwardWindowIndexer(window_size=6)
index_12 = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
index_24 = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
index_48 = pd.api.indexers.FixedForwardWindowIndexer(window_size=48)

df_hourly['sum_6'] = df_hourly['y'].rolling(window=index_6, min_periods=1).sum()
df_hourly['sum_12'] = df_hourly['y'].rolling(window=index_12, min_periods=1).sum()
df_hourly['sum_24'] = df_hourly['y'].rolling(window=index_24, min_periods=1).sum()
df_hourly['sum_48'] = df_hourly['y'].rolling(window=index_48, min_periods=1).sum()

df_hourly['bool_6'] = np.where(df_hourly['sum_6'] == 0, 0, 1)
df_hourly['bool_12'] = np.where(df_hourly['sum_12'] == 0, 0, 1)
df_hourly['bool_24'] = np.where(df_hourly['sum_24'] == 0, 0, 1)
df_hourly['bool_48'] = np.where(df_hourly['sum_48'] == 0, 0, 1)

df_hourly.to_csv('data/clean/rolling_sums.csv', index=False)