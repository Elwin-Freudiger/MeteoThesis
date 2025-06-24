import pandas as pd
import numpy as np
import time
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras_efficient_kan import KANLinear
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ── LOAD DATA ──────────────────────────────────────
df = pd.read_csv("../data/clean/valais_clean.csv")
df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M")
df.sort_values(["station", "time"], inplace=True)

# ── FEATURE ENGINEERING ────────────────────────────
features = ["temperature", "pressure", "moisture"]
target = "precip"

# Create lag-1 features
df[["temp_lag", "press_lag", "moist_lag"]] = df.groupby("station")[features].shift(1)
df = df.dropna(subset=["temp_lag", "press_lag", "moist_lag", "precip"])

# ── SCALE ──────────────────────────────────────────
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(df[["temp_lag", "press_lag", "moist_lag"]])
Y = scaler_y.fit_transform(df[[target]])

# ── MEMORY MONITOR ────────────────────────────────
def get_memory_mb():
    return psutil.Process().memory_info().rss / 1e6

# ── DENSE MODEL ───────────────────────────────────
def build_dense_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    return Model(inp, out)

# ── KAN MODEL ─────────────────────────────────────
def build_kan_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = KANLinear(64)(inp)
    x = KANLinear(32)(x)
    out = KANLinear(1)(x)
    return Model(inp, out)

# ── BENCHMARK FUNCTION ────────────────────────────
def benchmark(model_fn, X, y, label):
    print(f"\n🔍 Benchmarking: {label}")
    model = model_fn(X.shape[1])
    model.compile(optimizer=Adam(0.001), loss='mse')

    t0 = time.time()
    mem0 = get_memory_mb()

    model.fit(
        X, y,
        epochs=10,
        batch_size=256,
        validation_split=0.1,
        verbose=0,
        callbacks=[EarlyStopping(patience=2)]
    )

    elapsed = time.time() - t0
    mem1 = get_memory_mb()
    y_pred = model.predict(X, verbose=0)
    mse = mean_squared_error(y, y_pred)

    print(f"⏱ Time      : {elapsed:.2f} sec")
    print(f"🧠 Mem Δ     : {mem1 - mem0:.2f} MB")
    print(f"📉 MSE       : {mse:.6f}")
    return elapsed, mem1 - mem0, mse

# ── RUN BENCHMARKS ────────────────────────────────
benchmark(build_dense_model, X, Y, "Dense")
benchmark(build_kan_model, X, Y, "KANLinear")