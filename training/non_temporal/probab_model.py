import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kan import KAN
from sklearn.model_selection import train_test_split

# 1) PARAMETERS
WEATHER_CSV     = "../data/clean/rolling_sums.csv"
STATIONS_CSV    = "../data/clean/valais_stations.csv"
HIST_LEN        = 12      # past timesteps
HORIZON_HOURS   = 6       # we're predicting sum_6
BATCH_SIZE      = 64
EMB_DIM         = 4       # embedding dim for stations
HIDDEN_WIDTH    = [64, 1] # hidden width then output=1
GRID_SIZE       = HIST_LEN
K_ORDER         = 3
LR              = 1e-3
EPOCHS          = 20

# 2) LOAD & MERGE
df = pd.read_csv(WEATHER_CSV, parse_dates=["ds"])
df = df.rename(columns={'unique_id': 'station'})
stations = pd.read_csv(STATIONS_CSV)
df = df.merge(stations[["station","east","north","altitude"]],
              on="station", how="left")

# 3) ENCODE & SCALE
le = LabelEncoder()
df["station_idx"] = le.fit_transform(df["station"])
n_stations = df["station_idx"].nunique()

stat_feats = (
    df[["station_idx","east","north","altitude"]]
      .drop_duplicates()
      .set_index("station_idx")
      .astype(np.float32)              # ← ensure float64→float32
)
stat_scaler = StandardScaler()
stat_scaled = stat_scaler.fit_transform(stat_feats)
stat_feats = pd.DataFrame(
    stat_scaled,
    index=stat_feats.index,
    columns=stat_feats.columns
)  # ← rebuild with float dtype :contentReference[oaicite:0]{index=0}

dyn_cols = ['y', 'east_wind', 'north_wind', 'moisture', 'pressure', 'temperature']
dyn_scaler = StandardScaler()
df[dyn_cols] = dyn_scaler.fit_transform(df[dyn_cols])

def make_time_feats(ts: pd.Timestamp):
    return np.array([
        ts.hour   / 23.0,
        ts.weekday()/ 6.0,
        ts.month  / 12.0
    ], dtype=np.float32)

# 4) DATASET
class RainDataset(Dataset):
    def __init__(self, df, hist_len, horizon):
        self.df = df.reset_index(drop=True)
        self.hist_len = hist_len
        self.horizon = horizon

        valid = []
        for _, g in self.df.groupby("station_idx"):
            idx = g.index
            start = idx[0] + hist_len - 1
            end   = idx[-1] - horizon
            valid.extend(range(start, end+1))
        self.valid_idx = valid

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, ii):
        i   = self.valid_idx[ii]
        row = self.df.loc[i]
        sidx   = row["station_idx"]

        # static features
        static = stat_feats.loc[sidx].values.astype(np.float32)
        # dynamic history
        hist   = self.df.loc[i-self.hist_len+1 : i, dyn_cols].values.astype(np.float32)
        # time features
        tfeat  = make_time_feats(row["ds"])
        # target = sum_6
        y      = np.float32(row["sum_6"])

        X = {
            "hist"      : torch.from_numpy(hist),             # [HIST_LEN, D_dyn]
            "static"    : torch.from_numpy(static),           # [D_stat]
            "time_feats": torch.from_numpy(tfeat),            # [3]
            "st_idx"    : torch.tensor(sidx, dtype=torch.long)
        }
        return X, torch.tensor(y, dtype=torch.float32)

# split by time
cut1 = df["ds"].quantile(0.6)
cut2 = df["ds"].quantile(0.8)
df_train = df[df["ds"] <= cut1]
df_val   = df[(df["ds"] > cut1) & (df["ds"] <= cut2)]
df_test  = df[df["ds"] > cut2]

train_ds = RainDataset(df_train, HIST_LEN, HORIZON_HOURS)
val_ds   = RainDataset(df_val,   HIST_LEN, HORIZON_HOURS)
test_ds  = RainDataset(df_test,  HIST_LEN, HORIZON_HOURS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5) MODEL DEFINITION
class RainKANModel(nn.Module):
    def __init__(self, hist_len, dyn_dim, stat_dim, time_dim,
                 n_stations, emb_dim, kan_width, grid, k):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb_dim)
        input_dim = hist_len * dyn_dim + stat_dim + time_dim + emb_dim
        # KAN(width=[in_dim, hidden..., out_dim], grid, k)
        self.kan = KAN(width=[input_dim] + kan_width, grid=grid, k=k)

    def forward(self, hist, static, time_feats, st_idx):
        B = hist.size(0)
        hist_flat = hist.view(B, -1)
        emb       = self.emb(st_idx)               # [B, emb_dim]
        x         = torch.cat([hist_flat, static, time_feats, emb], dim=1)
        out       = self.kan(x)                    # [B, 1]
        return out.squeeze(-1)                     # → [B]

model = RainKANModel(
    hist_len = HIST_LEN,
    dyn_dim   = len(dyn_cols),
    stat_dim  = stat_feats.shape[1],
    time_dim  = 3,
    n_stations= n_stations,
    emb_dim   = EMB_DIM,
    kan_width = HIDDEN_WIDTH,
    grid      = GRID_SIZE,
    k         = K_ORDER
).to(device)

# 6) OPTIMIZER & LOSS
optimizer = Adam(model.parameters(), lr=LR)
loss_fn   = nn.MSELoss()  # for regression :contentReference[oaicite:3]{index=3}

# 7) TRAINING LOOP
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        hist       = X["hist"].to(device)
        static     = X["static"].to(device)
        time_feats = X["time_feats"].to(device)
        st_idx     = X["st_idx"].to(device)
        target     = y.to(device)

        optimizer.zero_grad()
        pred = model(hist, static, time_feats, st_idx)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * target.size(0)

    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            hist       = X["hist"].to(device)
            static     = X["static"].to(device)
            time_feats = X["time_feats"].to(device)
            st_idx     = X["st_idx"].to(device)
            target     = y.to(device)

            pred = model(hist, static, time_feats, st_idx)
            loss = loss_fn(pred, target)
            val_loss += loss.item() * target.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

# 8) TEST EVALUATION
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X, y in test_loader:
        hist       = X["hist"].to(device)
        static     = X["static"].to(device)
        time_feats = X["time_feats"].to(device)
        st_idx     = X["st_idx"].to(device)
        target     = y.to(device)

        pred = model(hist, static, time_feats, st_idx)
        test_loss += loss_fn(pred, target).item() * target.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test MSE: {test_loss:.4f}")