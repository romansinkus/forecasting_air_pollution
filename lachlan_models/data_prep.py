# lachlan_models/data_prep.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# LOAD BASE DATAFRAME
# =====================================================================

def load_base_dataframe():
    """
    Loads, normalises, and returns the dataframe from eda_dpp_utils.preProcessing().
    Ensures timestamp exists both as column AND as index.
    """
    eda_dpp_utils = importlib.import_module('eda_dpp_utils')
    df, df_unnormalised, numeric_cols = eda_dpp_utils.preProcessing()

    # -----------------------------------------------------------------
    # FIX TIMESTAMP: eda_dpp_utils sets timestamp as index, not column
    # -----------------------------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex) and "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})

    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

    # sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # add year field (used for 2004/2005 split)
    df["year"] = df["timestamp"].dt.year

    return df, df_unnormalised, numeric_cols


# =====================================================================
# DATASET FOR SEQUENCE MODELS
# =====================================================================

class AirQualitySeqDataset(Dataset):
    """
    Generates sliding-window sequences of numerical features,
    with multi-target regression and classification output.
    """

    def __init__(self, df, feature_cols, reg_targets, class_target,
                 window_size=24, horizon=1):

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove rows with missing features/targets
        df = df.dropna(subset=feature_cols + reg_targets + [class_target])
        df = df.reset_index(drop=True)

        self.X = df[feature_cols].values.astype(np.float32)
        self.Y_reg = df[reg_targets].values.astype(np.float32)
        self.Y_cls = df[class_target].values.astype(np.int64)

        self.window_size = window_size
        self.horizon = horizon

        # number of valid starting positions
        self.max_start = len(df) - (window_size + horizon) + 1
        self.max_start = max(self.max_start, 0)

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        target_idx = idx + self.window_size + self.horizon - 1

        # Convert to tensor channels-first
        x = torch.from_numpy(x_window).permute(1, 0)
        y_reg = torch.from_numpy(self.Y_reg[target_idx])
        y_cls = torch.tensor(self.Y_cls[target_idx], dtype=torch.long)

        return x, y_reg, y_cls


# =====================================================================
# MAIN DATALOADER FACTORY
# =====================================================================

def get_dataloaders(window_size=24, horizon=1, batch_size=64, val_fraction=0.2):
    """
    Creates train/val/test loaders based on:
    - Train = 2004
    - Test  = 2005
    - Val   = last val_fraction of 2004
    """

    df, df_unnormalised, numeric_cols = load_base_dataframe()

    # regression targets
    reg_targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

    # classification target: bins of CO(GT)
    df['CO_class'] = pd.cut(
        df['CO(GT)'].astype(float),
        bins=[-np.inf, 1.5, 2.5, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # candidate feature columns = all numeric except targets and engineered metadata
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    exclude = set(reg_targets + ["CO_class", "year"])
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # -------------------------
    # Train/Val/Test Split
    # -------------------------
    df_train_full = df[df["year"] == 2004].copy()
    df_test = df[df["year"] == 2005].copy()

    n = len(df_train_full)
    val_size = int(n * val_fraction)

    df_val = df_train_full.iloc[-val_size:].copy()
    df_train = df_train_full.iloc[:-val_size].copy()

    # -------------------------
    # Build datasets
    # -------------------------
    train_dataset = AirQualitySeqDataset(
        df_train, feature_cols, reg_targets, "CO_class",
        window_size, horizon
    )
    val_dataset = AirQualitySeqDataset(
        df_val, feature_cols, reg_targets, "CO_class",
        window_size, horizon
    )
    test_dataset = AirQualitySeqDataset(
        df_test, feature_cols, reg_targets, "CO_class",
        window_size, horizon
    )

    # -------------------------
    # Build dataloaders
    # -------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Metadata for models
    # -------------------------
    meta = {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "reg_targets": reg_targets,
        "num_reg_targets": len(reg_targets),
        "num_classes": 3,
        "window_size": window_size,
        "horizon": horizon
    }

    return train_loader, val_loader, test_loader, meta


# =====================================================================
# TEST HELPER
# =====================================================================

if __name__ == "__main__":
    t, v, te, meta = get_dataloaders()
    print("Metadata:", meta)
