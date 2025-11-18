# lachlan_models/train_cnn.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lachlan_models.data_prep import get_dataloaders
from lachlan_models.utils_plots import plot_training_curves, plot_regression_scatter
from lachlan_models.evaluation import (
    compute_rmse,
    naive_baseline,
    naive_baseline_classification,
    plot_residuals,
    plot_timeseries
)

# =====================================================================
# Multi-Head CNN Model
# =====================================================================

class CNNMultiHead(nn.Module):
    def __init__(self, in_channels, seq_len, num_reg_targets=5, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)

        # Determine flattened dimension via dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            h = self._forward_features(dummy)
            self.flat_dim = h.shape[1]

        self.fc_shared = nn.Linear(self.flat_dim, 128)
        self.fc_reg = nn.Linear(128, num_reg_targets)
        self.fc_cls = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        h = self._forward_features(x)
        h = F.relu(self.fc_shared(h))
        return self.fc_reg(h), self.fc_cls(h)


# =====================================================================
# Training helper
# =====================================================================

def run_epoch(model, loader, reg_criterion, cls_criterion,
              optimizer=None, device="cpu",
              lambda_reg=1.0, lambda_cls=1.0):

    train = optimizer is not None
    model.train() if train else model.eval()

    total_reg_loss = 0
    total_cls_loss = 0
    total_samples = 0
    correct = 0

    for x, y_reg, y_cls in loader:
        x = x.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        if train:
            optimizer.zero_grad()

        pred_reg, pred_cls = model(x)

        loss_reg = reg_criterion(pred_reg, y_reg)
        loss_cls = cls_criterion(pred_cls, y_cls)
        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_samples += bs
        total_reg_loss += loss_reg.item() * bs
        total_cls_loss += loss_cls.item() * bs
        correct += (pred_cls.argmax(1) == y_cls).sum().item()

    return (total_reg_loss / total_samples,
            total_cls_loss / total_samples,
            correct / total_samples)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default="cnn_multitask.pt")
    args = parser.parse_args()

    # Output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "cnn")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = "cnn"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device, flush=True)

    # Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        window_size=args.window_size,
        horizon=args.horizon,
        batch_size=args.batch_size
    )

    # Model
    model = CNNMultiHead(
        in_channels=meta["num_features"],
        seq_len=meta["window_size"],
        num_reg_targets=meta["num_reg_targets"],
        num_classes=meta["num_classes"]
    ).to(device)

    print(model, flush=True)

    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track history
    history = {k: [] for k in
               ["train_reg","train_cls","train_acc","val_reg","val_cls","val_acc"]}

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(1, args.epochs + 1):
        tr_reg, tr_cls, tr_acc = run_epoch(
            model, train_loader, reg_criterion, cls_criterion,
            optimizer, device,
            args.lambda_reg, args.lambda_cls
        )
        va_reg, va_cls, va_acc = run_epoch(
            model, val_loader, reg_criterion, cls_criterion,
            None, device,
            args.lambda_reg, args.lambda_cls
        )

        history["train_reg"].append(tr_reg)
        history["train_cls"].append(tr_cls)
        history["train_acc"].append(tr_acc)
        history["val_reg"].append(va_reg)
        history["val_cls"].append(va_cls)
        history["val_acc"].append(va_acc)

        print(f"[{epoch}] "
              f"TrainRMSE={tr_reg:.4f} ValRMSE={va_reg:.4f} | "
              f"TrainCls={tr_cls:.4f} ValCls={va_cls:.4f} | "
              f"TrainAcc={tr_acc:.3f} ValAcc={va_acc:.3f}",
              flush=True)

    # ============================================================
    # TEST SET PREDICTIONS
    # ============================================================

    y_true_reg = []
    y_pred_reg = []
    y_true_cls = []
    y_pred_cls = []

    model.eval()
    with torch.no_grad():
        for x, y_reg, y_cls in test_loader:
            x = x.to(device)

            pred_reg, pred_cls = model(x)

            y_true_reg.append(y_reg.numpy())
            y_pred_reg.append(pred_reg.cpu().numpy())

            y_true_cls.append(y_cls.numpy())
            y_pred_cls.append(pred_cls.cpu().numpy().argmax(1))

    y_true_reg = np.vstack(y_true_reg)
    y_pred_reg = np.vstack(y_pred_reg)
    y_true_cls = np.concatenate(y_true_cls)
    y_pred_cls = np.concatenate(y_pred_cls)

    # ============================================================
    # RMSE
    # ============================================================
    rmses = compute_rmse(y_true_reg, y_pred_reg)
    with open(os.path.join(OUTPUT_DIR, "rmse.txt"), "w") as f:
        for name, val in zip(meta["reg_targets"], rmses):
            f.write(f"{name}: {val:.4f}\n")

    # ============================================================
    # NAIVE BASELINE REGRESSION
    # ============================================================
    try:
        y_true_bl, y_pred_bl = naive_baseline(y_true_reg, args.horizon)
        baseline_rmse = compute_rmse(y_true_bl, y_pred_bl)
        with open(os.path.join(OUTPUT_DIR, "baseline_rmse.txt"), "w") as f:
            for name, val in zip(meta["reg_targets"], baseline_rmse):
                f.write(f"{name}: {val:.4f}\n")
    except Exception as e:
        print("Baseline regression skipped:", e)

    # ============================================================
    # NAIVE CLASSIFICATION
    # ============================================================
    try:
        baseline_acc, _, _ = naive_baseline_classification(y_true_cls, args.horizon)
        with open(os.path.join(OUTPUT_DIR, "baseline_classification.txt"), "w") as f:
            f.write(f"Naive baseline accuracy: {baseline_acc:.4f}\n")
    except Exception as e:
        print("Baseline classification skipped:", e)

    # ============================================================
    # PLOTS
    # ============================================================
    plot_training_curves(history, OUTPUT_DIR, prefix)
    plot_regression_scatter(y_true_reg, y_pred_reg, meta["reg_targets"], OUTPUT_DIR, prefix)
    plot_residuals(y_true_reg, y_pred_reg, meta["reg_targets"], OUTPUT_DIR, prefix)
    plot_timeseries(y_true_reg, y_pred_reg, meta["reg_targets"], OUTPUT_DIR, prefix)

    # ============================================================
    # SAVE MODEL
    # ============================================================
    save_path = os.path.join(OUTPUT_DIR, args.save_path)
    torch.save({
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "horizon": args.horizon
    }, save_path)

    print("Saved model to:", save_path, flush=True)


if __name__ == "__main__":
    main()
