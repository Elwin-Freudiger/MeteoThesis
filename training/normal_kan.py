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
from neuralforecast.losses.pytorch import DistributionLoss
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape

# 1) Logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# 2) Load data
df_hourly = pd.read_csv('data/clean/hourly_data.csv')
df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])

static_df = (
    pd.read_csv('data/clean/valais_stations.csv')
      .rename(columns={'station': 'unique_id'})
      [['unique_id', 'altitude', 'east', 'north']]
)

# 3) Train / validation split
cutoff    = pd.to_datetime('2022-12-31 23:00')
train_df  = df_hourly[df_hourly['ds'] <= cutoff].copy()
valid_df  = df_hourly[df_hourly['ds'] >  cutoff].copy()

# 4) Scale dynamic features
feature_cols = ['temperature','pressure','moisture','east_wind','north_wind']
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
valid_df[feature_cols] = scaler.transform(valid_df[feature_cols])

# persist scaler
joblib.dump(scaler, 'valais_scaler.pkl')

# 5) Partition train set into parquet by unique_id
parquet_dir = 'data/valais_train_parquet'
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
input_size = 24

models = [
    KAN(
        h                = horizon,
        input_size       = input_size,
        loss             = DistributionLoss("Poisson"),
        hist_exog_list   = feature_cols,
        stat_exog_list   = ['east','north','altitude'],
        max_steps        = 1000
    )
]

nf = NeuralForecast(
    models = models,
    freq   = '10min'
)

# 7) Fit on parquet partitions
nf.fit(
    df         = train_paths,
    static_df  = static_df,
    id_col     = 'unique_id'
)

nf.save(path='../checkpoints/hourly',
        model_index=None, 
        overwrite=True,
        save_dataset=False)

# 8) Single‐window cross‐validation (n_windows=1)
cv_df = nf.cross_validation(
    df         = train_df,
    static_df  = static_df,
    id_col     = 'unique_id',
    n_windows  = 1,
    step_size  = horizon,
    verbose    = False
)

# 9) Compute MAE, RMSE, sMAPE over that test window
#    drop 'cutoff' column before evaluation
evaluation_df = evaluate(
    cv_df.drop(columns=['cutoff']),
    metrics = [mae, rmse, smape]
)

# 10) Print results
print("\nCross‐validation metrics (single window):\n")
print(evaluation_df.to_string(index=False))

# Optional: pivot for a compact overview
summary = evaluation_df.pivot(index='metric', columns='unique_id', values='KAN')
print("\nAggregated by metric:\n", summary)
