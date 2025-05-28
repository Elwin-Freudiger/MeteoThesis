import os
import shutil
import pandas as pd
import logging
import tempfile
import matplotlib.pyplot as plt

#scaling
from sklearn.preprocessing import StandardScaler
import joblib

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from neuralforecast import NeuralForecast
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import DistributionLoss
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape
from utilsforecast.plotting import plot_series

#get loader:
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

df_hourly = pd.read_csv('../data/clean/hourly_data.csv')
df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])
static_df = (
    pd.read_csv('../data/clean/valais_stations.csv')
      .rename(columns={'station': 'unique_id'})
      [['unique_id', 'altitude', 'east', 'north']]
)

#split training and validation
cutoff = pd.to_datetime('2022-12-31 23:00')
train_df = df_hourly[df_hourly['ds'] <= cutoff]
valid_df = df_hourly[df_hourly['ds'] >  cutoff]

feature_cols = ['temperature','pressure','moisture','east_wind','north_wind']
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

horizon = 6
input = 24
models = [KAN(
    h                = horizon,   
    input_size       = input,  
    loss             = DistributionLoss("Poisson"),
    hist_exog_list   = ['temperature','pressure','moisture','east_wind','north_wind'],
    stat_exog_list   = ['east','north','altitude'],
    max_steps = 20
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

# pick your station
uid = 'BIN'
h = 6
input = 24
ts  = valid_df[valid_df.unique_id == uid].sort_values('ds')

#make input window
window = ts[:input]

predictions = nf.predict(
    df         = window,
    static_df  = static_df
)

print(predictions)
fig = plot_series(window, predictions)
fig.show()
"""
# get the predictions
predictions = nf.predict(
    df        = pred_df,
    futr_df   = futr_df,
    static_df = static_df
)

# merge with actuals for the forecast period
actual = ts[['ds','y']].tail(h).copy()
df_plot = actual.merge(predictions, on='ds')

# identify the forecast column (everything except ds & y)
pred_col = [c for c in df_plot.columns if c not in ('ds','y')][0]

# now plot
plt.figure()
plt.plot(df_plot['ds'], df_plot['y'],          label='Actual')
plt.plot(df_plot['ds'], df_plot[pred_col],     label='Predicted')
plt.xlabel('Timestamp')
plt.ylabel('Precipitation (mm)')
plt.title(f'Actual vs Predicted Rainfall for {uid}')
plt.legend()
plt.tight_layout()
plt.show()
"""