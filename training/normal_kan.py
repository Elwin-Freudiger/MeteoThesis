import os
import shutil
import pandas as pd
import logging
import tempfile

#scaling
from sklearn.preprocessing import StandardScaler
import joblib

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from neuralforecast import NeuralForecast
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import DistributionLoss
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape

#get loader:
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

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
    .agg({'precip': 'sum',
          'East': 'mean',
          'North': 'mean',
          'moisture': 'mean',
          'pressure': 'mean',
          'temperature': 'mean'
          })
    .reset_index()
)

df_hourly = (
    df_hourly
    .drop(columns=['ds'])        
    .rename(columns={'ds_first':'ds'})  
    .set_index(['station','ds'])
)

print(df_hourly)

static_df = (
    pd.read_csv('../data/clean/valais_stations.csv')
      .rename(columns={'station': 'unique_id'})
      [['unique_id', 'altitude', 'east', 'north']]
)

#split training and validation
cutoff = pd.to_datetime('2022-12-31 23:00')
train_df = df_hourly[df_hourly['ds'] <= cutoff]
valid_df = df_hourly[df_hourly['ds'] >  cutoff]

feature_cols = ['y','temperature','pressure','moisture','east_wind','north_wind']
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
valid_df[feature_cols] = scaler.transform(valid_df[feature_cols])

joblib.dump(scaler, 'valais_scaler.pkl')

#split the data into several files with parquet
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

horizon = 12
input = 24
models = [KAN(
    h                = horizon*6,   
    input_size       = input*6,  
    loss             = DistributionLoss("Poisson"),
    hist_exog_list   = ['temperature','pressure','moisture','east_wind','north_wind'],
    stat_exog_list   = ['east','north','altitude']
)]

nf = NeuralForecast(
    models   = models,
    freq     = '10min'
)

#fit the model
nf.fit(
    df        = train_paths,
    static_df = static_df,
    id_col    = 'unique_id'
)

uid = 'BIN'
ts = (
    valid_df[valid_df['unique_id'] == uid]
      .sort_values('ds')
)

pred_df = ts.tail(models.input_size).copy()

futr_df = (
    ts.tail(models.input_size + models.h)
      .head(models.h)
      .drop(columns=['y'])
      .copy()
)

predictions = nf.predict(
    df         = pred_df,
    futr_df    = futr_df,
    static_df  = static_df
)

print(predictions)