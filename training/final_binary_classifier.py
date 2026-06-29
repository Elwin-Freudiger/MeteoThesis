import os
import time
import json
import platform
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam

from keras_efficient_kan import KANLinear


# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

WEATHER_CSV = "../data/clean/valais_clean.csv"
STATIONS_CSV = "../data/clean/valais_stations.csv"

MODEL_DIR = "../model_testing"
MODEL_PATH = os.path.join(MODEL_DIR, "forecast_binary_1.keras")
REPORT_CSV_PATH = os.path.join(MODEL_DIR, "model_lightweightness_report.csv")
LATEX_TABLE_PATH = os.path.join(MODEL_DIR, "model_comparison_table.tex")
JSON_REPORT_PATH = os.path.join(MODEL_DIR, "model_lightweightness_report.json")

HIST_LEN = 36
HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.0001

np.random.seed(42)
tf.random.set_seed(42)

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────
# PROFILING UTILITIES
# ──────────────────────────────────────────

class TimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.perf_counter()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f} s")

    def on_train_end(self, logs=None):
        self.total_training_time = time.perf_counter() - self.train_start_time
        print(f"\nTotal training time: {self.total_training_time:.2f} s")


def count_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def count_non_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))


def get_file_size_mb(path):
    if path is None or not os.path.exists(path):
        return None
    return os.path.getsize(path) / (1024 ** 2)


def measure_inference_latency(model, x_sample, runs=100, warmup=10):
    """
    Measures prediction latency using model(x, training=False), which avoids
    some overhead from model.predict().
    """
    x_sample = tf.convert_to_tensor(x_sample)

    for _ in range(warmup):
        _ = model(x_sample, training=False)

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x_sample, training=False)
    end = time.perf_counter()

    avg_sec = (end - start) / runs
    return avg_sec


def get_model_report(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    timing_callback=None,
    model_path=None,
    hist_len=None,
    horizon=None,
    batch_size=None,
    epochs_requested=None,
    num_stations=None,
):
    input_shape = tuple(x_train.shape[1:])
    output_shape = tuple(y_train.shape[1:])

    values_per_input_sample = int(np.prod(input_shape))
    values_per_output_sample = int(np.prod(output_shape))

    total_params = model.count_params()
    trainable_params = count_trainable_params(model)
    non_trainable_params = count_non_trainable_params(model)

    report = {
        "model_name": "Proposed LSTM-KAN station model",
        "input_type": "Multivariate station time series",
        "input_shape_per_sample": str(input_shape),
        "input_values_per_sample": values_per_input_sample,
        "input_dtype": str(x_train.dtype),
        "output_type": "Multi-label binary precipitation occurrence per station",
        "output_shape_per_sample": str(output_shape),
        "output_values_per_sample": values_per_output_sample,
        "history_length_steps": hist_len,
        "forecast_horizon_steps": horizon,
        "num_stations_predicted": num_stations,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "batch_size": batch_size,
        "epochs_requested": epochs_requested,
        "total_layers": len(model.layers),
        "trainable_layers": int(sum(1 for layer in model.layers if layer.trainable)),
        "non_trainable_layers": int(sum(1 for layer in model.layers if not layer.trainable)),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_size_mb": get_file_size_mb(model_path),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),
    }

    if timing_callback is not None:
        report["epochs_run"] = len(timing_callback.epoch_times)
        report["total_training_time_sec"] = float(timing_callback.total_training_time)
        report["average_epoch_time_sec"] = float(np.mean(timing_callback.epoch_times))
        report["fastest_epoch_time_sec"] = float(np.min(timing_callback.epoch_times))
        report["slowest_epoch_time_sec"] = float(np.max(timing_callback.epoch_times))

    return report


def build_comparison_table_own_vs_dgmr(own_report):
    """
    DGMR values below are from Ravuri et al., Nature 2021.
    Some values, like parameter count and number of layers, are not reported
    directly in the main paper, so they are marked as not reported.
    """

    own_input_shape = own_report["input_shape_per_sample"]
    own_input_values = own_report["input_values_per_sample"]

    dgmr_input_values_training_crop = 4 * 256 * 256
    dgmr_input_values_full_radar = 4 * 1536 * 1280

    rows = [
        {
            "Criterion": "Input data type",
            "Proposed model": own_report["input_type"],
            "DGMR / Ravuri et al.": "Radar image fields",
        },
        {
            "Criterion": "Input size per sample",
            "Proposed model": f"{own_input_shape} = {own_input_values:,} scalar values",
            "DGMR / Ravuri et al.": (
                f"4 × 256 × 256 = {dgmr_input_values_training_crop:,} values during crop training; "
                f"up to 4 × 1,536 × 1,280 = {dgmr_input_values_full_radar:,} values for full radar context"
            ),
        },
        {
            "Criterion": "Input spatial nature",
            "Proposed model": "Station-level / point-based features",
            "DGMR / Ravuri et al.": "Dense 2D precipitation radar grids at 1 km × 1 km resolution",
        },
        {
            "Criterion": "Temporal context",
            "Proposed model": f"{own_report['history_length_steps']} past time steps",
            "DGMR / Ravuri et al.": "4 past radar frames / previous 20 min",
        },
        {
            "Criterion": "Prediction target",
            "Proposed model": own_report["output_type"],
            "DGMR / Ravuri et al.": "18 future radar fields / next 90 min",
        },
        {
            "Criterion": "Model family",
            "Proposed model": "LSTM + KANLinear + Dense sigmoid output",
            "DGMR / Ravuri et al.": "Conditional generative adversarial nowcasting model",
        },
        {
            "Criterion": "Total parameters",
            "Proposed model": f"{own_report['total_parameters']:,}",
            "DGMR / Ravuri et al.": "Not reported in main paper",
        },
        {
            "Criterion": "Trainable parameters",
            "Proposed model": f"{own_report['trainable_parameters']:,}",
            "DGMR / Ravuri et al.": "Not reported in main paper",
        },
        {
            "Criterion": "Number of layers",
            "Proposed model": f"{own_report['total_layers']}",
            "DGMR / Ravuri et al.": "Not directly reported in main paper",
        },
        {
            "Criterion": "Training set scale",
            "Proposed model": f"{own_report['train_samples']:,} training samples",
            "DGMR / Ravuri et al.": "Approximately 1.5 million radar examples",
        },
        {
            "Criterion": "Training configuration",
            "Proposed model": (
                f"{own_report['epochs_run']} epochs actually run, "
                f"batch size {own_report['batch_size']}"
            ),
            "DGMR / Ravuri et al.": "500k generator steps, 2 discriminator steps per generator step",
        },
        {
            "Criterion": "Training time",
            "Proposed model": f"{own_report['total_training_time_sec']:.2f} s",
            "DGMR / Ravuri et al.": "One week on 16 TPU cores",
        },
        {
            "Criterion": "Average epoch time",
            "Proposed model": f"{own_report['average_epoch_time_sec']:.2f} s",
            "DGMR / Ravuri et al.": "Not applicable / not reported as epochs",
        },
        {
            "Criterion": "Saved model size",
            "Proposed model": (
                f"{own_report['model_size_mb']:.2f} MB"
                if own_report["model_size_mb"] is not None
                else "Not available"
            ),
            "DGMR / Ravuri et al.": "Not reported in main paper",
        },
        {
            "Criterion": "Reported inference speed",
            "Proposed model": "Measured separately by this script",
            "DGMR / Ravuri et al.": "Median 25.7 s/sample on CPU; 1.3 s/sample on NVIDIA V100 GPU",
        },
    ]

    return pd.DataFrame(rows)


def save_latex_table(df, path):
    latex = df.to_latex(
        index=False,
        escape=True,
        column_format="p{0.23\\linewidth}p{0.34\\linewidth}p{0.34\\linewidth}",
        caption=(
            "Comparison between the proposed lightweight station-based model "
            "and the DGMR radar-based nowcasting model."
        ),
        label="tab:model_comparison",
    )

    # Make it more paper-friendly.
    latex = latex.replace("\\begin{table}", "\\begin{table}[ht]")
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")

    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)

    return latex


# ──────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────

df_weather = pd.read_csv(WEATHER_CSV)
df_weather["time"] = pd.to_datetime(df_weather["time"], format="%Y%m%d%H%M")

df_stations = pd.read_csv(STATIONS_CSV)
df = df_weather.merge(
    df_stations[["station", "east", "north", "altitude"]],
    on="station",
    how="left"
)


# ──────────────────────────────────────────
# BUILD WIDE FORMAT
# ──────────────────────────────────────────

selected_features = ["precip", "temperature", "East", "North", "pressure", "moisture"]
metadata_features = ["east", "north", "altitude"]

all_features = selected_features + metadata_features

df_features = df[["time", "station"] + all_features].copy()
df_pivot = df_features.pivot(index="time", columns="station", values=all_features)

df_pivot.columns = [f"{feat}_{station}" for feat, station in df_pivot.columns]
df_pivot = df_pivot.sort_index()
df_pivot = df_pivot.dropna()


# ──────────────────────────────────────────
# SPLITTING
# ──────────────────────────────────────────

split1 = int(0.6 * len(df_pivot))
split2 = int(0.8 * len(df_pivot))

df_train = df_pivot.iloc[:split1]
df_val = df_pivot.iloc[split1:split2]
df_test = df_pivot.iloc[split2:]


# ──────────────────────────────────────────
# SCALING
# ──────────────────────────────────────────

scaler = StandardScaler()

data_train = scaler.fit_transform(df_train)
data_val = scaler.transform(df_val)
data_test = scaler.transform(df_test)

precip_cols = [col for col in df_pivot.columns if col.startswith("precip_")]
num_stations = len(precip_cols)


# ──────────────────────────────────────────
# WINDOW CREATION
# ──────────────────────────────────────────

def make_windows(
    scaled_array,
    original_df,
    precip_cols,
    hist_len,
    horizon,
    undersample_no_rain=False,
    keep_no_rain_probability=0.2,
):
    x, y = [], []

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=original_df.columns,
        index=original_df.index
    )

    for i in range(hist_len, len(scaled_df) - horizon):
        x_window = scaled_df.iloc[i - hist_len:i].values

        horizon_vals = original_df.iloc[i + 1:i + 1 + horizon][precip_cols].values
        y_window = (np.any(horizon_vals > 0, axis=0)).astype(np.float32)

        if undersample_no_rain:
            total_future_rain = np.sum(horizon_vals)

            if total_future_rain == 0 and np.random.rand() > keep_no_rain_probability:
                continue

        x.append(x_window)
        y.append(y_window)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return x, y


x_train, y_train = make_windows(
    data_train,
    df_train,
    precip_cols,
    HIST_LEN,
    HORIZON,
    undersample_no_rain=True,
    keep_no_rain_probability=0.2,
)

x_val, y_val = make_windows(
    data_val,
    df_val,
    precip_cols,
    HIST_LEN,
    HORIZON,
    undersample_no_rain=False,
)

x_test, y_test = make_windows(
    data_test,
    df_test,
    precip_cols,
    HIST_LEN,
    HORIZON,
    undersample_no_rain=False,
)

print("\nDataset shapes")
print("-" * 40)
print(f"x_train: {x_train.shape}, dtype={x_train.dtype}")
print(f"y_train: {y_train.shape}, dtype={y_train.dtype}")
print(f"x_val:   {x_val.shape}, dtype={x_val.dtype}")
print(f"y_val:   {y_val.shape}, dtype={y_val.dtype}")
print(f"x_test:  {x_test.shape}, dtype={x_test.dtype}")
print(f"y_test:  {y_test.shape}, dtype={y_test.dtype}")
print(f"Number of stations predicted: {num_stations}")
print(f"Input values per sample: {np.prod(x_train.shape[1:]):,}")


# ──────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────

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
output = layers.Dense(num_stations, activation="sigmoid")(x)

model = Model(inputs=ts_input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

print("\nPre-training model size")
print("-" * 40)
print(f"Total parameters:         {model.count_params():,}")
print(f"Trainable parameters:     {count_trainable_params(model):,}")
print(f"Non-trainable parameters: {count_non_trainable_params(model):,}")
print(f"Total layers:             {len(model.layers)}")
print(f"Input shape:              {model.input_shape}")
print(f"Output shape:             {model.output_shape}")


# ──────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────

timing_callback = TimingCallback()

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        timing_callback,
    ],
    verbose=1,
)


# ──────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────

y_pred = model.predict(x_val, verbose=0)
y_pred_binary = (y_pred > 0.5).astype(int)

print("\nValidation classification report")
print("-" * 40)
print(classification_report(y_val.flatten(), y_pred_binary.flatten(), digits=3))


# ──────────────────────────────────────────
# SAVE MODEL
# ──────────────────────────────────────────

model.save(MODEL_PATH)
model_size_mb = get_file_size_mb(MODEL_PATH)

print("\nSaved model")
print("-" * 40)
print(f"Path: {MODEL_PATH}")
print(f"Size: {model_size_mb:.2f} MB")


# ──────────────────────────────────────────
# INFERENCE LATENCY
# ──────────────────────────────────────────

single_sample_latency_sec = measure_inference_latency(
    model,
    x_val[:1],
    runs=100,
    warmup=10
)

batch_sample_count = min(BATCH_SIZE, len(x_val))
batch_latency_sec = measure_inference_latency(
    model,
    x_val[:batch_sample_count],
    runs=50,
    warmup=5
)

print("\nInference timing")
print("-" * 40)
print(f"Single-sample latency: {single_sample_latency_sec * 1000:.3f} ms/sample")
print(f"Batch latency:         {batch_latency_sec * 1000:.3f} ms/batch")
print(f"Batch size measured:   {batch_sample_count}")
print(f"Per-sample in batch:   {(batch_latency_sec / batch_sample_count) * 1000:.6f} ms/sample")


# ──────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────

own_report = get_model_report(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    timing_callback=timing_callback,
    model_path=MODEL_PATH,
    hist_len=HIST_LEN,
    horizon=HORIZON,
    batch_size=BATCH_SIZE,
    epochs_requested=EPOCHS,
    num_stations=num_stations,
)

own_report["single_sample_inference_latency_ms"] = single_sample_latency_sec * 1000
own_report["batch_inference_latency_ms"] = batch_latency_sec * 1000
own_report["batch_inference_size"] = batch_sample_count
own_report["batch_inference_latency_per_sample_ms"] = (
    batch_latency_sec / batch_sample_count
) * 1000

report_df = pd.DataFrame([own_report])
report_df.to_csv(REPORT_CSV_PATH, index=False)

with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(own_report, f, indent=2)

print("\nModel lightweightness report")
print("-" * 40)
print(report_df.T)


# ──────────────────────────────────────────
# LATEX COMPARISON TABLE
# ──────────────────────────────────────────

comparison_df = build_comparison_table_own_vs_dgmr(own_report)

latex_table = save_latex_table(comparison_df, LATEX_TABLE_PATH)

print("\nComparison table")
print("-" * 40)
print(comparison_df)

print("\nLaTeX table")
print("-" * 40)
print(latex_table)

print("\nFiles written")
print("-" * 40)
print(f"Model:       {MODEL_PATH}")
print(f"CSV report:  {REPORT_CSV_PATH}")
print(f"JSON report: {JSON_REPORT_PATH}")
print(f"LaTeX table: {LATEX_TABLE_PATH}")