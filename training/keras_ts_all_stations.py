import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_efficient_kan import KANLinear
import keras
from tqdm import tqdm

# ── CONFIG ────────────────────────────────
WEATHER_CSV = "../data/clean/valais_clean.csv"
STATIONS_CSV = "../data/clean/valais_stations.csv"

HIST_LEN = 36     # 6 hours history
HORIZON = 1       # 1 hour ahead
BATCH_SIZE = 256
EPOCHS = 10
SPLIT_FRACTION = 0.8
LEARNING_RATE = 0.0001

# ── LOAD MODEL ─────────────────────────────
model_binary = keras.models.load_model('../model_testing/forecast_binary_6.keras')

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


# ── SPLITTING RAW TIME INDEX ─────────────────────────
split1 = int(0.6 * len(df_pivot))  # train
split2 = int(0.8 * len(df_pivot))  # val
df_train = df_pivot.iloc[:split1]
df_val   = df_pivot.iloc[split1:split2]
df_test  = df_pivot.iloc[split2:]

# ── SCALING ──────────────────────────────────────────
scaler = StandardScaler()
data_train = scaler.fit_transform(df_train)
data_val   = scaler.transform(df_val)
data_test  = scaler.transform(df_test)

data_train = pd.DataFrame(data_train, columns=df_train.columns, index=df_train.index)
data_val   = pd.DataFrame(data_val,   columns=df_val.columns,   index=df_val.index)
data_test  = pd.DataFrame(data_test,  columns=df_test.columns,  index=df_test.index)

# ── PRECIP COLUMNS ──────────────────────────────────
precip_cols = [col for col in df_pivot.columns if col.startswith("precip_")]
num_stations = len(precip_cols)
x_train, y_train = [], []

for i in range(HIST_LEN, len(data_train) - HORIZON):
    x_window = data_train.iloc[i - HIST_LEN:i].values
    y_window = df_train.iloc[i+HORIZON][precip_cols].values  # unscaled

    total_future_rain = np.sum(y_window)

    if total_future_rain == 0:
        continue

    x_train.append(x_window)
    y_train.append(np.log1p(y_window))

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val, y_val = [], []
for i in range(HIST_LEN, len(data_val) - HORIZON):
    x_window = data_val.iloc[i - HIST_LEN:i].values
    y_window = df_val.iloc[i + HORIZON][precip_cols].values
    x_val.append(x_window)
    y_val.append(np.log1p(y_window))

x_val = np.array(x_val)
y_val = np.array(y_val)

x_test, y_test = [], []
for i in range(HIST_LEN, len(data_test) - HORIZON):
    x_window = data_test.iloc[i - HIST_LEN:i].values
    y_window = df_test.iloc[i + HORIZON][precip_cols].values
    x_test.append(x_window)
    y_test.append(np.log1p(y_window))

x_test = np.array(x_test)
y_test = np.array(y_test) 

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
output = layers.Dense(1)(x) 


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

model.save("../model_testing/final_one_step_fcst.keras")
