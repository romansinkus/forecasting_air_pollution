from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_forecast_linear_regression(df, target, targets, df_unnormalised=None, horizons=[1,6,12,24]):
    
    # Normal split: train=2004, test=2005
    if target != "NMHC(GT)":
        train_df = df[df.index.year == 2004].copy()
        test_df  = df[df.index.year == 2005].copy()
    else:
        df_clean = df.dropna(subset=['NMHC(GT)']).copy()
        split_idx = int(0.8 * len(df_clean))
        train_df = df_clean.iloc[:split_idx].copy()
        test_df  = df_clean.iloc[split_idx:].copy()

    features = [c for c in df.columns if c not in targets and 'NMHC(GT)' not in c]

    results = {}

    if df_unnormalised is None:
        raise ValueError("Please pass df_unnormalised (the unscaled dataframe) so we can report RMSE in original units.")

    # Prepare grids
    n_h = len(horizons)
    n_rows = 1
    n_cols = n_h  # 1 row, as many columns as horizons

    fig_res, axes_res = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6), sharey=True)
    axes_res = axes_res.flatten()

    fig_ts, axes_ts = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4), sharey=False)
    axes_ts = axes_ts.flatten()

    for i, h in enumerate(horizons):
        # Shift target
        train_valid = train_df.copy()
        test_valid  = test_df.copy()
        train_valid[f"{target}_future_{h}h"] = train_valid[target].shift(-h)
        test_valid[f"{target}_future_{h}h"] = test_valid[target].shift(-h)
        train_valid = train_valid.dropna(subset=[f"{target}_future_{h}h"])
        test_valid  = test_valid.dropna(subset=[f"{target}_future_{h}h"])

        X_train = train_valid[features]
        y_train = train_valid[f"{target}_future_{h}h"]
        X_test  = test_valid[features]
        y_test  = test_valid[f"{target}_future_{h}h"]

        # Drop NaNs for the horizon-specific training set
        nan_rows_train = X_train.isna().any(axis=1) | y_train.isna()
        if nan_rows_train.any():
            X_train = X_train[~nan_rows_train]
            y_train = y_train[~nan_rows_train]

        # Drop NaNs from the horizon-specific test set
        nan_rows_test = X_test.isna().any(axis=1) | y_test.isna()
        if nan_rows_test.any():
            X_test = X_test[~nan_rows_test]
            y_test = y_test[~nan_rows_test]

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # RMSE
        t_mean = df_unnormalised[target].mean()
        t_std  = df_unnormalised[target].std()

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        y_test_real = (y_test_arr * t_std) + t_mean
        y_pred_real = (y_pred_arr * t_std) + t_mean

        # Compute RMSE in original units
        rmse = root_mean_squared_error(y_test_real, y_pred_real)

        # Residuals
        residuals = y_test - y_pred
        axes_res[i].scatter(y_test.index, residuals)
        axes_res[i].axhline(0, color='black', linewidth=1)
        axes_res[i].set_title(f"{h}h ahead Residuals\n")
        axes_res[i].set_xlabel("Time")
        axes_res[i].set_ylabel("Residual")
        for tick in axes_res[i].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        axes_res[i].text(
            0.95, 0.95,
            f"LR RMSE={rmse:.3f}\n",
            transform=axes_res[i].transAxes,
            ha='right', va='top',
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

        # Predicted vs Observed
        axes_ts[i].plot(y_test.index, y_test, label="Observed")
        axes_ts[i].plot(y_test.index, y_pred, label="Predicted")
        axes_ts[i].set_title(f"{h}h ahead Predicted vs Observed")
        axes_ts[i].set_xlabel("Time")
        axes_ts[i].set_ylabel(target)
        axes_ts[i].grid(True)
        for tick in axes_ts[i].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')

    # Hide empty subplots if any
    for j in range(i+1, len(axes_res)):
        axes_res[j].axis('off')
        axes_ts[j].axis('off')

    plt.tight_layout(pad=3.0)
    fig_res.suptitle(f"{target} Residuals per Horizon", y=1.05, fontsize=16)
    fig_ts.suptitle(f"{target} Predicted vs Observed per Horizon", y=1.02, fontsize=16)
    plt.show()

    return results
