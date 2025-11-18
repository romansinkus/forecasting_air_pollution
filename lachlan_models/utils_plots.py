# lachlan_models/utils_plots.py

import os
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_training_curves(history, save_dir, prefix="model"):
    ensure_dir(save_dir)

    epochs = np.arange(1, len(history["train_reg"]) + 1)

    # Regression Loss
    plt.figure()
    plt.plot(epochs, history["train_reg"], label="train")
    plt.plot(epochs, history["val_reg"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Regression Loss (MSE)")
    plt.title("Regression Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}_reg_loss.png"))
    plt.close()

    # Classification Loss
    plt.figure()
    plt.plot(epochs, history["train_cls"], label="train")
    plt.plot(epochs, history["val_cls"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title("Classification Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}_cls_loss.png"))
    plt.close()

    # Classification Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}_cls_acc.png"))
    plt.close()


def plot_regression_scatter(y_true, y_pred, target_names, save_dir, prefix="model"):
    ensure_dir(save_dir)

    for i, tname in enumerate(target_names):
        plt.figure()
        plt.scatter(y_true[:,i], y_pred[:,i], s=3)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Regression Scatter: {tname}")
        plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_{tname}.png"))
        plt.close()
