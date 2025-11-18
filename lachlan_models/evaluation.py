# lachlan_models/evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ================================================================
# RMSE CALCULATION
# ================================================================

def compute_rmse(y_true, y_pred):
    """
    y_true, y_pred: shape (N, D)
    Returns RMSE per column
    """
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rmses.append(rmse)
    return np.array(rmses)


# ================================================================
# BASELINE PREDICTION
# ================================================================

def naive_baseline(y_true, horizon):
    """
    y_true: (N, D) true pollutant concentrations at target times.
    We need y at time t, which corresponds to y_true shifted by -horizon.
    """
    if horizon >= len(y_true):
        raise ValueError("Horizon too large for baseline.")

    # y_true corresponds to target times
    # naive prediction y_pred_baseline[t] = y_true[t - horizon]
    y_pred = y_true[:-horizon]
    y_true_adj = y_true[horizon:]

    return y_true_adj, y_pred


def naive_baseline_classification(y_cls, horizon):
    """
    y_cls: class labels (N,)
    naive prediction = label at time t
    target = label at time t + horizon
    """
    if horizon >= len(y_cls):
        raise ValueError("Horizon too large for classification baseline.")

    y_pred = y_cls[:-horizon]
    y_true_adj = y_cls[horizon:]
    acc = (y_pred == y_true_adj).mean()

    return acc, y_true_adj, y_pred


# ================================================================
# PLOTS
# ================================================================

def plot_residuals(y_true, y_pred, target_names, save_dir, prefix):
    ensure_dir(save_dir)
    residuals = y_true - y_pred

    for i, tname in enumerate(target_names):
        plt.figure(figsize=(8, 4))
        plt.hist(residuals[:, i], bins=50, alpha=0.7)
        plt.title(f"Residual Distribution for {tname}")
        plt.xlabel("Residual (True - Pred)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_residuals_{tname}.png"))
        plt.close()


def plot_timeseries(y_true, y_pred, target_names, save_dir, prefix, max_points=1000):
    ensure_dir(save_dir)

    if y_true.shape[0] > max_points:
        y_true = y_true[:max_points]
        y_pred = y_pred[:max_points]

    t = np.arange(len(y_true))

    for i, tname in enumerate(target_names):
        plt.figure(figsize=(10, 4))
        plt.plot(t, y_true[:, i], label='True', linewidth=1)
        plt.plot(t, y_pred[:, i], label='Predicted', linewidth=1)
        plt.title(f"Time Series Prediction vs Truth for {tname}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_timeseries_{tname}.png"))
        plt.close()
