import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from neuralforecast import NeuralForecast

# --- Config ---
DATA_PATH = '../data/clean/hourly_data.csv'
STATIC_PATH = '../data/clean/valais_stations.csv'
MODEL_PATH = '../checkpoints/hourly/'
OUTPUT_CSV = '../model_testing/eval_data/hourly_eval_all_stations.csv'
CUTOFF = pd.to_datetime('2022-12-31 23:00')
INPUT_SIZE = 24
HORIZON = 6
N_JOBS = -1  # Use all available cores

# --- Load data ---
df_hourly = pd.read_csv(DATA_PATH)
df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])

static_df = (
    pd.read_csv(STATIC_PATH)
    .rename(columns={'station': 'unique_id'})
    [['unique_id', 'altitude', 'east', 'north']]
)

train_df = df_hourly[df_hourly['ds'] <= CUTOFF].copy()
valid_df = df_hourly[df_hourly['ds'] > CUTOFF].copy()

# --- Scale features ---
feature_cols = ['temperature', 'pressure', 'moisture', 'east_wind', 'north_wind']
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
valid_df[feature_cols] = scaler.transform(valid_df[feature_cols])

# --- Load model ---
nf = NeuralForecast.load(path=MODEL_PATH)

# --- Define forecasting function ---
def forecast_station(uid):
    ts_full = valid_df[valid_df['unique_id'] == uid].sort_values('ds').reset_index(drop=True)
    predictions = []

    for start in range(0, len(ts_full) - INPUT_SIZE, HORIZON):
        window = ts_full.iloc[start : start + INPUT_SIZE].copy()
        pred = nf.predict(df=window, static_df=static_df)
        predictions.append(pred)

    if not predictions:
        return pd.DataFrame()

    pred_df = pd.concat(predictions, ignore_index=True).drop_duplicates(subset=['ds'])
    eval_df = ts_full.merge(pred_df[['ds', 'KAN']], on='ds', how='inner')
    eval_df['station'] = uid
    return eval_df

# --- Run in parallel ---
uids = valid_df['unique_id'].unique()
results = Parallel(n_jobs=N_JOBS)(delayed(forecast_station)(uid) for uid in uids)

# --- Save results ---
final_df = pd.concat(results, ignore_index=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved evaluation results to: {OUTPUT_CSV}")

## CONFUSION MATRIX
# Well add a separation for the metrics when it said there would be rain and when it said there would not be rain.

#add a column for actual rain:
eval_df['actual_rain'] = np.where(
    eval_df['y'] == 0, 0, 1)

eval_df['rain_predicted'] = np.where(
    eval_df['KAN'] == 0, 0, 1)

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(eval_df['actual_rain'], eval_df['rain_predicted'])
print(conf)