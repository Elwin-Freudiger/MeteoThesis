import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Concatenate, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from keras_efficient_kan import KANLinear

import keras
from keras import ops

# ── TARGET VARIABLE ───────────────────────────────────────────────────────────
TARGET_VAR = 'sum_48'

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
WEATHER_CSV   = "../data/clean/rolling_sums.csv"
STATIONS_CSV  = "../data/clean/valais_stations.csv"
HIST_LEN      =  12   # history window
HORIZON       = 6     # sum_6 target
BATCH_SIZE    = 64
EMB_DIM       = 4
KAN_HID       = 64
KAN_GRID      = 50
KAN_K         = 3
LR            = 1e-3
EPOCHS        = 10

# ── 1) LOAD & MERGE ────────────────────────────────────────────────────────────
df_weather = pd.read_csv(WEATHER_CSV, parse_dates=["time"])

df_stations = pd.read_csv(STATIONS_CSV)
df = df_weather.merge(
    df_stations[["station","east","north","altitude"]],
    on="station", how="left"
)

# ── 2) ENCODE & SCALE ──────────────────────────────────────────────────────────
df["station_idx"] = LabelEncoder().fit_transform(df["station"])
n_stations = df["station_idx"].nunique()

# static
stat_df = (
    df[["station_idx","east","north","altitude"]]
    .drop_duplicates()
    .set_index("station_idx")
    .astype(np.float32)
)
stat_df[:] = StandardScaler().fit_transform(stat_df)

# dynamic
dyn_cols = ["precip","moisture","pressure","temperature"]
df[dyn_cols] = StandardScaler().fit_transform(df[dyn_cols])

# helper for time features
def make_time_feats(ts):
    return [ts.month/12.0]

# sort by station and time_id
df = df.sort_values(["station_idx", "time"]).reset_index(drop=True)

# grab raw numpy arrays
target_vals  = df[TARGET_VAR].values.astype(np.float32)

dyn_vals   = df[dyn_cols].values.astype(np.float32)
ds_vals    = df["time"].values           
sidx_vals  = df["station_idx"].values.astype(np.int32)

# ── 3) BUILD WINDOWS per‐station ───────────────────────────────────────────────
X_hist, X_stat, X_time, X_sid, y_list = [], [], [], [], []

for sidx, grp in df.groupby("station_idx", sort=False):
    idx = grp.index.values
    arr_dyn = dyn_vals[idx]  

    # sliding windows of shape [T-HIST_LEN+1, HIST_LEN, D_dyn]
    win = sliding_window_view(arr_dyn, window_shape=(HIST_LEN, arr_dyn.shape[1]))
    win = win.reshape(-1, HIST_LEN, arr_dyn.shape[1])

    # targets & time at end of each window
    targs = target_vals[idx][HIST_LEN-1:]
    times = ds_vals  [idx][HIST_LEN-1:]

    # time‐of‐day feats
    time_feat = np.stack([ make_time_feats(pd.to_datetime(t)) for t in times ], axis=0)

    # static feats repeated
    sf = np.repeat(stat_df.loc[sidx].values[None,:], len(targs), axis=0)

    # station idx array
    sid = np.full(len(targs), sidx, dtype=np.int32)

    X_hist.append(win)
    X_time.append(time_feat)
    X_stat.append(sf)
    X_sid.append(sid)
    y_list.append(targs)

# concat all stations
X_hist = np.vstack(X_hist) 
X_time = np.vstack(X_time) 
X_stat = np.vstack(X_stat) 
X_sid  = np.concatenate(X_sid) 
y_all  = np.concatenate(y_list)

#log transform
y_all_log = np.log1p(y_all)


# ── 4) TRAIN/VAL/TEST SPLIT (by date quantile) ────────────────────────────────
q1, q2 = df["time"].quantile([0.6,0.8])
mask = np.concatenate([
    pd.to_datetime(ds_vals[df.groupby("station_idx", sort=False).get_group(sidx).index][HIST_LEN-1:])
    for sidx in sorted(df["station_idx"].unique())
])

train_idx = np.where(mask <= q1)[0]

# ── PERFORM UNDERSAMPLING ────────────────────────────────

# Get training labels
y_all_bin = (y_all > 0).astype(int)

y_train_bin = y_all_bin[train_idx]

# Identify minority and majority class indices
minority_class = 1 if np.sum(y_train_bin) < len(y_train_bin) / 2 else 0
majority_class = 1 - minority_class

minority_idx = train_idx[y_train_bin == minority_class]
majority_idx = train_idx[y_train_bin == majority_class]

# Undersample majority class
np.random.seed(42)
majority_sample = np.random.choice(majority_idx, size=len(minority_idx), replace=False)

# Combine and shuffle
undersampled_train_idx = np.concatenate([minority_idx, majority_sample])
np.random.shuffle(undersampled_train_idx)

val_idx   = np.where((mask>q1)&(mask<=q2))[0]
test_idx  = np.where(mask > q2)[0]

def make_ds(idxs):
    return (
       tf.data.Dataset.from_tensor_slices((
            {
                "hist"   : X_hist[idxs],
                "stat"   : X_stat[idxs],
                "time"   : X_time[idxs],
                "station": X_sid[idxs],
            },
            y_all_log[idxs]
        ))
        .shuffle(10_000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

train_ds = make_ds(undersampled_train_idx)
val_ds   = make_ds(val_idx)
test_ds  = make_ds(test_idx)

# ── 5) MODEL DEFINITION ────────────────────────────────────────────────────────
hist_in = Input((HIST_LEN, len(dyn_cols)), name="hist")
stat_in = Input((X_stat.shape[1],), name="stat")
time_in = Input((1,), name="time")  # Match actual feature
sid_in  = Input((), dtype=tf.int32, name="station")

emb = Embedding(n_stations, EMB_DIM)(sid_in)
emb = Flatten()(emb)
hflat = Flatten()(hist_in)

x = Concatenate()([hflat, stat_in, time_in, emb])

# --- Project to fixed size first ---
x = Dense(KAN_HID, activation="linear")(x)

# --- KAN Layers ---
x = KANLinear(KAN_HID, spline_order=KAN_K, grid_size=KAN_GRID)(x)
x = KANLinear(KAN_HID, spline_order=KAN_K, grid_size=KAN_GRID)(x)
x = KANLinear(KAN_HID, spline_order=KAN_K, grid_size=KAN_GRID)(x)

out = Dense(1, activation=None, name="log_sum_out")(x)

model_kan = Model([hist_in, stat_in, time_in, sid_in], out)
model_kan.compile(optimizer=Adam(LR), loss="mse", metrics=["mae"])

# --- Train with early stopping ---
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history_kan = model_kan.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[early_stop]
)

print("Test  :", model_kan.evaluate(test_ds))

log_preds = model_kan.predict(test_ds) 
sum_preds = np.expm1(log_preds)  

# ── 7) SAVE MODEL ───────────────────────────────────────────────────────────
model_kan.save("../model_testing/sums_48_kan.keras")