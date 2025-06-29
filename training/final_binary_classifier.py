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
HORIZON = 1       # 10 minutes ahead
BATCH_SIZE = 256
EPOCHS = 10
SPLIT_FRACTION = 0.8
LEARNING_RATE = 0.0001

#set the numpy seed to make sure that the unsersampling is reproducible
np.random.seed(42)

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


# ── SPLITTING ───────────────────────────────
split1 = int(0.6 * len(df_pivot))
split2 = int(0.8 * len(df_pivot))
df_train = df_pivot.iloc[:split1]
df_val   = df_pivot.iloc[split1:split2]
df_test  = df_pivot.iloc[split2:]

# ── SCALING ───────────────────────────────
scaler = StandardScaler()
data_train = scaler.fit_transform(df_train)
data_val   = scaler.transform(df_val)
data_test  = scaler.transform(df_test)

# ── SAMPLE CONSTRUCTION ───────────────────
# ── SETUP ──────────────────────────────────
precip_cols = [col for col in df_pivot.columns if col.startswith("precip_")]
num_stations = len(precip_cols)

# ── TRAIN SET CONSTRUCTION (undersampled) ─
x_train, y_train = [], []
train_scaled = pd.DataFrame(data_train, columns=df_train.columns, index=df_train.index)

for i in range(HIST_LEN, len(train_scaled) - HORIZON):
    x_window = train_scaled.iloc[i - HIST_LEN:i].values
    horizon_vals = df_train.iloc[i + 1 : i + 1 + HORIZON][precip_cols].values
    y_window = (np.any(horizon_vals > 0, axis=0)).astype(int)
    total_future_rain = np.sum(horizon_vals)

    if total_future_rain == 0 and np.random.rand() > 0.2:
        continue

    x_train.append(x_window)
    y_train.append(y_window)

# ── VALIDATION SET CONSTRUCTION (no undersampling) ─
x_val, y_val = [], []
val_scaled = pd.DataFrame(data_val, columns=df_val.columns, index=df_val.index)

for i in range(HIST_LEN, len(val_scaled) - HORIZON):
    x_window = val_scaled.iloc[i - HIST_LEN:i].values
    horizon_vals = df_val.iloc[i + 1 : i + 1 + HORIZON][precip_cols].values
    y_window = (np.any(horizon_vals > 0, axis=0)).astype(int)

    x_val.append(x_window)
    y_val.append(y_window)

# ── TEST SET CONSTRUCTION (no undersampling) ─
x_test, y_test = [], []
test_scaled = pd.DataFrame(data_test, columns=df_test.columns, index=df_test.index)

for i in range(HIST_LEN, len(test_scaled) - HORIZON):
    x_window = test_scaled.iloc[i - HIST_LEN:i].values
    horizon_vals = df_test.iloc[i + 1 : i + 1 + HORIZON][precip_cols].values
    y_window = (np.any(horizon_vals > 0, axis=0)).astype(int)

    x_test.append(x_window)
    y_test.append(y_window)

# ── CONVERT TO ARRAYS ─────────────────────
x_train = np.array(x_train) 
y_train = np.array(y_train)
x_val   = np.array(x_val)
y_val   = np.array(y_val)
x_test  = np.array(x_test)
y_test  = np.array(y_test)

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
output = layers.Dense(num_stations, activation='sigmoid')(x)

model = Model(inputs=ts_input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
model.summary()

# ── TRAINING ──────────────────────────────
model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

y_pred = model.predict(x_val)
y_pred_binary = (y_pred > 0.5).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_val.flatten(), y_pred_binary.flatten(), digits=3))

model.save("../model_testing/forecast_binary_1.keras")
