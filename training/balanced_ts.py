"""
This model is balanced, meaning that observations without any rain are undersampled when loading.
The loss function is also weighted to give more importance to non-zero observations. 
"""

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

import torch
import torch.nn.functional as F
from neuralforecast.models.kan import KAN

# 1) Logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

import torch
import torch.nn as nn
from neuralforecast.losses.pytorch import DistributionLoss

class WeightedPoissonLoss(DistributionLoss):
    def __init__(self, pos_weight: float = 5.0):
        # build a “Poisson” distributional loss with no internal reduction
        super().__init__(distribution="Poisson")
        self.pos_weight = pos_weight

    def _compute_weights(self, y: torch.Tensor, mask: torch.Tensor):
        """
        y:    [batch, h, 1]      (true values)
        mask: [batch, h, 1]      (0/1 missing‐value mask)
        """
        # 1) get the default weights (horizon_weight * mask)
        base_w = super()._compute_weights(y, mask)    # shape [batch, h, 1]

        # 2) build a mask of positive‐rain points
        positive = (y > 0).float()                    # [batch, h, 1]

        # 3) create a factor tensor: 1 for zeros, pos_weight for positives
        factor = 1.0 + (self.pos_weight - 1.0) * positive

        # 4) apply
        return base_w * factor
    
class KANCustomWindows(KAN):
    def __init__(self, *args, zero_keep_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_keep_prob = zero_keep_prob

    def _create_windows(self, batch, step, w_idxs=None):
        # Parse common data
        window_size = self.input_size + self.h
        temporal_cols = batch["temporal_cols"]
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

            temporal = self.padder_train(temporal)

            if temporal.shape[-1] < window_size:
                raise Exception(
                    "Time series is too short for training, consider setting a smaller input size or set start_padding_enabled=True"
                )

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )

            if self.MULTIVARIATE:
                # [n_series, C, Ws, L + h] -> [Ws, L + h, C, n_series]
                windows = windows.permute(2, 3, 1, 0)
            else:
                # [n_series, C, Ws, L + h] -> [Ws * n_series, L + h, C, 1]
                windows_per_serie = windows.shape[2]
                windows = windows.permute(0, 2, 3, 1)
                windows = windows.flatten(0, 1)
                windows = windows.unsqueeze(-1)
            
            #identify windows with missing values
            missing_mask = torch.isnan(windows)
            valid_windows = ~missing_mask.any(dim=(1,2,3))

            # Get the index of the 'y' target feature
            y_idx = pd.Index(temporal_cols).get_loc("y")

            # Extract the y-values: shape [N_windows, L+h, 1, 1]
            y_vals = windows[:, :, y_idx, :]

            # Identify windows that are ALL zero
            zero_mask = torch.all(y_vals == 0.0, dim=(1, 2))

            # Create a Bernoulli mask to keep only a percentage of zero windows
            dropout_mask = torch.rand_like(zero_mask.float()) < 0.1  # keep 10% of all-zero windows

            # Final mask: keep all non-zero windows, and 10% of zero-only windows
            retain_mask = ~zero_mask | dropout_mask

            # Combine with valid (non-NaN) mask
            final_mask = valid_windows & retain_mask

            # Apply mask
            windows = windows[final_mask]

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            # Repeat static if univariate: [n_series, S] -> [Ws * n_series, S]
            if static is not None and not self.MULTIVARIATE:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )
                static = static[valid_windows]

            # Protection of empty windows
            if windows.shape[0] == 0:
                raise Exception("No valid windows available for training after filtering missing values")

            # Sample windows
            if self.windows_batch_size is not None:
                n_windows = windows.shape[0]
                w_idxs = np.random.choice(
                    n_windows,
                    size=self.windows_batch_size,
                    replace=(n_windows < self.windows_batch_size),
                )
                windows = windows[w_idxs]

                if static is not None and not self.MULTIVARIATE:
                    static = static[w_idxs]

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch

        elif step in ["predict", "val"]:

            if step == "predict":
                initial_input = temporal.shape[-1] - self.test_size
                if (
                    initial_input <= self.input_size
                ):  # There is not enough data to predict first timestamp
                    temporal = F.pad(
                        temporal,
                        pad=(self.input_size - initial_input, 0),
                        mode="constant",
                        value=0.0,
                    )
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = temporal[:, :, cutoff:]

            elif step == "val":
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff : -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]
                if temporal.shape[-1] < window_size:
                    initial_input = temporal.shape[-1] - self.val_size
                    temporal = F.pad(
                        temporal,
                        pad=(self.input_size - initial_input, 0),
                        mode="constant",
                        value=0.0,
                    )

            if (
                (step == "predict")
                and (self.test_size == 0)
                and (len(self.futr_exog_list) == 0)
            ):
                temporal = F.pad(temporal, pad=(0, self.h), mode="constant", value=0.0)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            if self.MULTIVARIATE:
                # [n_series, C, Ws, L + h] -> [Ws, L + h, C, n_series]
                windows = windows.permute(2, 3, 1, 0)
            else:
                # [n_series, C, Ws, L + h] -> [Ws * n_series, L + h, C, 1]
                windows_per_serie = windows.shape[2]
                windows = windows.permute(0, 2, 3, 1)
                windows = windows.flatten(0, 1)
                windows = windows.unsqueeze(-1)
                if static is not None:
                    static = torch.repeat_interleave(
                        static, repeats=windows_per_serie, dim=0
                    )

            # Sample windows for batched prediction
            if w_idxs is not None:
                windows = windows[w_idxs]
                if static is not None and not self.MULTIVARIATE:
                    static = static[w_idxs]

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch
        else:
            raise ValueError(f"Unknown step {step}")


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
    KANCustomWindows(
        h                = horizon,
        input_size       = input_size,
        loss             = WeightedPoissonLoss(pos_weight=10.0),
        hist_exog_list   = feature_cols,
        stat_exog_list   = ['east','north','altitude'],
        windows_batch_size= 8,
        max_steps        = 10,
        zero_keep_prob   = 0.1,   
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

nf.save(path='../checkpoints/balanced_ts',
        model_index=None, 
        overwrite=True,
        save_dataset=False)

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