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
df_hourly = pd.read_csv('../data/clean/hourly_data.csv')
df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])

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
feature_cols = ['temperature','pressure','moisture','east_wind','north_wind']
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
input_size = 24

models = [
    KAN(
        h                = horizon,
        input_size       = input_size,
        loss             = DistributionLoss("Poisson"),
        hist_exog_list   = feature_cols,
        stat_exog_list   = ['east','north','altitude'],
        max_steps        = 20
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


# --- 1) metric definitions in pandas/numpy ---
def mae_pandas(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse_pandas(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def smape_pandas(y_true, y_pred):
    # avoid division by zero
    denom = np.abs(y_true) + np.abs(y_pred)
    safe_denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(2 * np.abs(y_pred - y_true) / safe_denom) * 100

# --- 2) rolling‐window forecast on validation set for one station ---
uid         = 'BIN'         # change to your station
input_size  = 24            # same as model input_size
horizon     = 6             # same as model h
ts_full     = valid_df[valid_df.unique_id == uid] \
                  .sort_values('ds') \
                  .reset_index(drop=True)

predictions = []
for start in range(0, len(ts_full) - input_size, horizon):
    # history window
    window = ts_full.iloc[start : start + input_size].copy()
    # forecast horizon
    pred = nf.predict(
        df        = window,
        static_df = static_df
    )
    predictions.append(pred)

# concatenate and drop any duplicates
pred_df = pd.concat(predictions, ignore_index=True)
pred_df = pred_df.drop_duplicates(subset=['ds'])

# --- 3) merge and compute metrics ---
eval_df = ts_full.merge(
    pred_df[['ds','KAN']],  # model name as column
    on='ds', how='inner'
)

metrics = {
    'MAE':  mae_pandas(eval_df['y'],  eval_df['KAN']),
    'RMSE': rmse_pandas(eval_df['y'], eval_df['KAN']),
    'sMAPE': smape_pandas(eval_df['y'], eval_df['KAN']),
}

print(f"Validation metrics for station {uid}:")
for name, val in metrics.items():
    print(f"  {name}: {val:.4f}")