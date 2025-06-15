import os
import shutil
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import joblib

# limit PyTorch CUDA split size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import DistributionLoss, MAE
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape

import torch
import torch.nn.functional as F
from neuralforecast.models.kan import KAN

# 1) Logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import torch
import torch.nn as nn
from neuralforecast.losses.pytorch import DistributionLoss

# 2) Load data
df_hourly = pd.read_csv('../data/clean/hourly_data.csv')
df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])
df_hourly = df_hourly.rename(columns={'y': 'precip', 'temperature':'y'})

static_df = (
    pd.read_csv('../data/clean/valais_stations.csv')
      .rename(columns={'station': 'unique_id'})
      [['unique_id', 'altitude', 'east', 'north']]
)

# 3) Train / validation split
cutoff    = pd.to_datetime('2022-12-31 23:00')
train_df  = df_hourly[df_hourly['ds'] <= cutoff].copy()
valid_df  = df_hourly[df_hourly['ds'] >  cutoff].copy()

# 4) Scale dynamic features
feature_cols = ['precip', 'pressure','moisture','east_wind','north_wind']
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
valid_df[feature_cols] = scaler.transform(valid_df[feature_cols])

# persist scaler
joblib.dump(scaler, 'valais_scaler.pkl')

# 5) Partition train set into parquet by unique_id
parquet_dir = '../data/valais_train_parquet'
if os.path.exists(parquet_dir):
    shutil.rmtree(parquet_dir)

train_df.to_parquet(
    parquet_dir,
    partition_cols=['unique_id'],
    index=False
)

train_paths = [
    os.path.join(parquet_dir, d)
    for d in os.listdir(parquet_dir)
    if d.startswith('unique_id=')
]

# 6) Model definition
horizon    = 6
input_size = 48

models = [
    KAN(
        h                = horizon,
        input_size       = input_size,
        loss             = MAE(),
        hist_exog_list   = feature_cols,
        stat_exog_list   = ['east','north','altitude'],
        windows_batch_size= 32,
        grid_size = 20,
        max_steps        = 100
    )
]

nf = NeuralForecast(
    models = models,
    freq   = 'h',
    local_scaler_type='standard'
)

# 7) Fit on parquet partitions
nf.fit(
    df         = train_paths,
    static_df  = static_df,
    id_col     = 'unique_id'
)

cv_df = nf.cross_validation(
    df = valid_df,
    static_df = static_df,
    id_col = 'unique_id',
    n_windows = 200,
    step_size = horizon,
    verbose = True
)
print(cv_df)

evaluation_df = evaluate(
    cv_df.drop(columns=['cutoff']),
    metrics = [mae, rmse, smape]
)
print(evaluation_df)