import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ── CONFIG ────────────────────────────────
WEATHER_CSV = "../data/clean/valais_clean.csv"
STATIONS_CSV = "../data/clean/valais_stations.csv"
HIST_LEN = 36
HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.0001

# ── LOAD DATA ─────────────────────────────
df_weather = pd.read_csv(WEATHER_CSV)
df_weather["time"] = pd.to_datetime(df_weather["time"], format="%Y%m%d%H%M")
df_stations = pd.read_csv(STATIONS_CSV)
df = df_weather.merge(df_stations[["station", "east", "north", "altitude"]], on="station", how="left")


station_list = sorted(df["station"].unique())
station_to_index = {s: i for i, s in enumerate(station_list)}
num_stations = len(station_list)

# ── FEATURE ENGINEERING ────────────────────
selected_features = ['precip', 'temperature', 'pressure', 'moisture']
metadata_features = ['altitude']

all_features = selected_features + metadata_features
# ── PIVOT TO WIDE FORMAT ───────────────────
df_pivot = df.pivot(index='time', columns='station', values=all_features)
df_pivot.columns = [f"{feat}_{station}" for feat, station in df_pivot.columns]
df_pivot.sort_index(inplace=True)
df_pivot.dropna(inplace=True)

# ── SPLIT BY TIME ──────────────────────────
split1 = int(0.6 * len(df_pivot))
split2 = int(0.8 * len(df_pivot))
df_train = df_pivot.iloc[:split1]
df_val = df_pivot.iloc[split1:split2]

# ── SCALE FEATURES ─────────────────────────
scaler = StandardScaler()
df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns, index=df_val.index)

# ── BUILD SAMPLES ──────────────────────────
def build_samples(df_src, df_target, station_list, undersample):
    x, y = [], []
    station_ids = [] 

    for station in station_list:
        col_name = f"precip_{station}"
        station_index = station_to_index[station]
        one_hot = to_categorical(station_index, num_classes=num_stations)

        for i in range(HIST_LEN, len(df_src) - HORIZON):
            y_val = df_target.iloc[i + HORIZON][col_name]
            if y_val <= 0:
                continue
            if undersample and np.random.rand() > 0.4:
                continue

            x_window = df_src.iloc[i - HIST_LEN:i].values
            # Repeat one-hot encoding across HIST_LEN timesteps
            one_hot_repeated = np.tile(one_hot, (HIST_LEN, 1))
            # Concatenate
            x_window_augmented = np.concatenate([x_window, one_hot_repeated], axis=1)

            x.append(x_window_augmented)
            y.append(np.log1p(y_val))
            station_ids.append(station_index)

    return np.array(x), np.array(y), np.array(station_ids)

x_train, y_train, _ = build_samples(df_train_scaled, df_train, station_list, undersample=True)
x_val, y_val, _ = build_samples(df_val_scaled, df_val, station_list, undersample=True)

# ── MODEL ──────────────────────────────────
ts_input = Input(shape=x_train.shape[1:])
x = layers.Flatten()(ts_input)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(32, activation="relu")(x)

output = layers.Dense(1)(x)

model = Model(inputs=ts_input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ── TRAIN ──────────────────────────────────
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# ── SAVE ───────────────────────────────────
model.save("../model_testing/regressor_mlp.keras")