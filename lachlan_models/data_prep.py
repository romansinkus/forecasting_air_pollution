# lachlan_models/data_prep.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow importing eda_dpp_utils

import importlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Load and preprocess
# ---------------------------------------------------------

def load_base_dataframe():
    eda_dpp_utils = importlib.import_module('eda_dpp_utils')
    df, df_unnormalised, numeric_cols = eda_dpp_utils.preProcessing()

    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['year'] = df['timestamp'].dt.year

    return df, df_unnormalised, numeric_cols


# ---------------------------------------------------------
# Sequence Dataset
# ---------------------------------------------------------

class AirQualitySeqDataset(Dataset):
    def __init__(self, df, feature_cols, reg_targets, class_target,
                 window_size=24, horizon=1):
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.dropna(subset=feature_cols + reg_targets + [class_target])
        df = df.reset_index(drop=True)

        self.X = df[feature_cols].values.astype(np.float32)
        self.Y_reg = df[reg_targets].values.astype(np.float32)
        self.Y_cls = df[class_target].values.astype(np.int64)

        self.feature_cols = feature_cols
        self.reg_targets = reg_targets
        self.class_target = class_target
        self.window_size = window_size
        self.horizon = horizon

        self.max_start = len(df) - (window_size + horizon) + 1
        if self.max_start < 1:
            self.max_start = 0

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        target_idx = idx + self.window_size + self.horizon - 1

        x = torch.from_numpy(x_window).permute(1, 0)
        y_reg = torch.from_numpy(self.Y_reg[target_idx])
        y_cls = torch.tensor(self.Y_cls[target_idx])

        return x, y_reg, y_cls


# ---------------------------------------------------------
# Main DataLoader factory
# ---------------------------------------------------------

def get_dataloaders(window_size=24, horizon=1, batch_size=64, val_fraction=0.2):
    df, df_unnormalised, numeric_cols = load_base_dataframe()

    reg_targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

    df['CO_class'] = pd.cut(
        df['CO(GT)'].astype(float),
        bins=[-np.inf, 1.5, 2.5, np.inf],
        labels=[0,1,2]
    ).astype(int)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    exclude = set(reg_targets + ['year', 'CO_class'])
    feature_cols = [c for c in numeric_cols if c not in exclude]

    df_train_full = df[df['year'] == 2004].copy()
    df_test = df[df['year'] == 2005].copy()

    n = len(df_train_full)
    val_size = int(n * val_fraction)

    df_val = df_train_full.iloc[-val_size:].copy()
    df_train = df_train_full.iloc[:-val_size].copy()

    train_dataset = AirQualitySeqDataset(df_train, feature_cols, reg_targets, 'CO_class',
                                         window_size, horizon)
    val_dataset   = AirQualitySeqDataset(df_val, feature_cols, reg_targets, 'CO_class',
                                         window_size, horizon)
    test_dataset  = AirQualitySeqDataset(df_test, feature_cols, reg_targets, 'CO_class',
                                         window_size, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    meta = {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "reg_targets": reg_targets,
        "num_reg_targets": len(reg_targets),
        "num_classes": 3,
        "window_size": window_size
    }

    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    t, v, ts, meta = get_dataloaders()
    print(meta)
