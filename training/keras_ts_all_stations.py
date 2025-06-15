import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_efficient_kan import KANLinear
from tqdm import tqdm

# ── CONFIG ────────────────────────────────
WEATHER_CSV = "../data/clean/valais_clean.csv"
STATIONS_CSV = "../data/clean/valais_stations.csv"

HIST_LEN = 36     # 6 hours history
HORIZON = 6       # 1 hour ahead
BATCH_SIZE = 256
EPOCHS = 10
SPLIT_FRACTION = 0.8
LEARNING_RATE = 0.0001

# ── LOAD DATA ─────────────────────────────
df_weather = pd.read_csv(WEATHER_CSV)
df_weather["time"] = pd.to_datetime(df_weather["time"], format="%Y%m%d%H%M")

df_stations = pd.read_csv(STATIONS_CSV)
df = df_weather.merge(df_stations[["station", "east", "north", "altitude"]], on="station", how="left")

# ── BUILD WIDE FORMAT ─────────────────────
selected_features = ['precip', 'temperature', 'East', 'North', 'pressure', 'moisture']
metadata_features = ['east', 'north', 'altitude']

all_features = selected_features + metadata_features

df_features = df[["time", "station"] + all_features].copy()
df_pivot = df_features.pivot(index="time", columns="station", values=all_features)
df_pivot.columns = [f"{feat}_{station}" for feat, station in df_pivot.columns]
df_pivot = df_pivot.sort_index()
df_pivot = df_pivot.dropna()  

# ── SCALING ───────────────────────────────
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_pivot)
data_scaled = pd.DataFrame(data_scaled, columns=df_pivot.columns, index=df_pivot.index)

# ── SAMPLE CONSTRUCTION ───────────────────
x, y = [], []

precip_cols = [col for col in df_pivot.columns if col.startswith("precip_")]
num_stations = len(precip_cols)

for i in range(HIST_LEN, len(data_scaled) - HORIZON):
    x_window = data_scaled.iloc[i - HIST_LEN:i].values
    y_window = df_pivot.iloc[i + HORIZON][precip_cols].values  # unscaled precip for all stations

    total_future_rain = np.sum(y_window)

    # Optional undersampling
    if total_future_rain == 0 and np.random.rand() > 0.1:
        continue

    x.append(x_window)
    y.append(y_window)

x = np.array(x)  
y = np.array(y)  
y = np.log1p(y)  

# ── SPLIT ─────────────────────────────────
split_idx = int(SPLIT_FRACTION * len(x))
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ── MODEL ─────────────────────────────────
input_shape = x_train.shape[1:] 
ts_input = Input(shape=input_shape)

x = layers.LSTM(64, return_sequences=True)(ts_input)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(32)(x)
x = layers.Dropout(0.3)(x)

x = layers.Reshape((1, 32))(x)
x = KANLinear(32)(x)
x = KANLinear(16)(x)

x = layers.Flatten()(x)
output = KANLinear(num_stations)(x)

model = Model(inputs=ts_input, outputs=output)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
model.summary()

# ── TRAINING ──────────────────────────────
model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

model.save("../model_testing/forecast_allstation_lstm_kan.keras")
