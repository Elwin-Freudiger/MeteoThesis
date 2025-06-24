"""
This file applies timeseries forecasting using KERAS to make forecasts for our weather stations.
"""
import pandas as pd
import numpy as np
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_efficient_kan import KANLinear

from sklearn.preprocessing import StandardScaler

# ── CONFIG ───────────────────────────────────────
WEATHER_CSV = "../data/clean/valais_clean.csv"
STATIONS_CSV = "../data/clean/valais_stations.csv"

HIST_LEN = 36      # 6 hours history @ 10-minute freq
HORIZON = 1        # predict 1 hour ahead (6 * 10 minutes)
BATCH_SIZE = 256
EPOCHS = 10
SPLIT_FRACTION = 0.8
LEARNING_RATE = 0.001

# ── LOAD & PREP DATA ─────────────────────────────
df_weather = pd.read_csv(WEATHER_CSV)
df_weather["time"] = pd.to_datetime(df_weather["time"], format="%Y%m%d%H%M")

df_stations = pd.read_csv(STATIONS_CSV)
df = df_weather.merge(df_stations[["station", "east", "north", "altitude"]], on="station", how="left")
df = df[df["station"] == "SIO"]

# ── NORMALIZE ────────────────────────────────────
selected_features = ['precip', 'temperature', 'East', 'North', 'pressure', 'moisture']
features = df[selected_features]
features.index = df["time"]

train_split = int(SPLIT_FRACTION * len(features))
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features = pd.DataFrame(features_scaled, index=features.index, columns=selected_features)

x, y = [], []

# Store all precip values separately to compute total rain in horizon
precip_series = df["precip"].values  # unscaled!

for i in range(HIST_LEN, len(features) - HORIZON):
    x_window = features.iloc[i - HIST_LEN:i].values
    y_window = precip_series[i:i + HORIZON]  # raw precip horizon
    
    total_future_rain = np.sum(y_window)
    
    # Retain all rainy examples, but keep only a subset of dry ones
    if total_future_rain == 0:
        if np.random.rand() < 0.2:  # Keep 20% of dry samples
            x.append(x_window)
            y.append(y_window[-1])  # still forecast last value or mean etc.
    else:
        x.append(x_window)
        y.append(y_window[-1])

x = np.array(x)
y = np.array(y)


split_idx = int(SPLIT_FRACTION * len(x))
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ── INPUT SHAPE ─────────────────────────────────────
inputs = layers.Input(shape=(HIST_LEN, len(selected_features)))

# ── LSTM BLOCK ──────────────────────────────────────
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(32)(x)
x = layers.Dropout(0.3)(x)

# ── KAN-LINEAR BLOCK ────────────────────────────────
x = layers.Reshape((1, 32))(x)  # reshape for KANLinear compatibility
x = KANLinear(32)(x)
x = KANLinear(16)(x)

# ── OUTPUT ──────────────────────────────────────────
x = layers.Flatten()(x)
outputs = KANLinear(1)(x)

# ── COMPILE MODEL ───────────────────────────────────
model = models.Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)
model.summary()

# ── TRAINING ────────────────────────────────────────
model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(x_val).flatten()  # or x_test if you have separate test set

# ── EVALUATE REGRESSION METRICS ───────────────────────────
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

model.save("../model_testing/forecast_lstm__kan_balanced.keras")