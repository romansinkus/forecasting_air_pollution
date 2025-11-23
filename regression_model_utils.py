from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_forecast_random_forest(df, target, targets, horizons=[1,6,12,24],
                                      n_estimators=500, random_state=42):
    
    # Normal split: train=2004, test=2005
    if target != "NMHC(GT)":
        train_df = df[df.index.year == 2004].copy()
        test_df  = df[df.index.year == 2005].copy()
    else:
        df_clean = df.dropna(subset=['NMHC(GT)']).copy()
        split_idx = int(0.8 * len(df_clean))
        train_df = df_clean.iloc[:split_idx].copy()
        test_df  = df_clean.iloc[split_idx:].copy()

    # Features are all non-target columns
    features = [c for c in df.columns if c not in targets]

    results = {}

    # Compute training RMSE once
    full_X_train = train_df[features]
    full_y_train = train_df[target]
    y_train_pred = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state
    ).fit(full_X_train, full_y_train).predict(full_X_train)
    train_rmse = root_mean_squared_error(full_y_train, y_train_pred)
    train_rel_rmse = train_rmse / full_y_train.mean()
    print(f"Training RMSE for {target} = {train_rmse:.3f}, Relative = {train_rel_rmse:.3f}")

    # Prepare grids
    n_h = len(horizons)
    n_cols = 2
    n_rows = int(np.ceil(n_h / n_cols))
    
    fig_res, axes_res = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*8), sharey=True)
    axes_res = axes_res.flatten()
    
    fig_ts, axes_ts = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*4))
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

        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # RMSE
        rmse = root_mean_squared_error(y_test, y_pred)
        rel_rmse = rmse / y_test.mean()
        results[h] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "rmse": rmse,
            "relative_rmse": rel_rmse
        }

        # Naive baseline vs Observed 
        y_naive = y_test.shift(h).dropna() 
        common_idx = y_test.index.intersection(y_naive.index) 
        y_test_aligned = y_test.loc[common_idx] 
        y_naive_aligned = y_naive.loc[common_idx] 

        # RMSE for naive baseline 
        naive_rmse = root_mean_squared_error(y_test_aligned, y_naive_aligned) 
        naive_rel_rmse = naive_rmse / y_test_aligned.mean() 
    
        # -------------------------
        # Residuals in grid
        # -------------------------
        residuals = y_test - y_pred
        axes_res[i].scatter(y_test.index, residuals)
        axes_res[i].axhline(0, color='black', linewidth=1)
        axes_res[i].set_title(f"{h}h ahead Residuals\n")
        axes_res[i].set_xlabel("Time")
        axes_res[i].set_ylabel("Residual")

        axes_res[i].text(
            0.95, 0.95,
            f"RF Rel RMSE={rel_rmse:.3f}\nNaive Rel RMSE={naive_rel_rmse:.3f}",
            transform=axes_res[i].transAxes,
            ha='right', va='top',
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

        # -------------------------
        # Predicted vs Observed in grid
        # -------------------------
        axes_ts[i].plot(y_test.index, y_test, label="Observed")
        axes_ts[i].plot(y_test.index, y_pred, label=f"Predicted")
        axes_ts[i].set_title(f"{h}h ahead Predicted vs Observed")
        axes_ts[i].set_xlabel("Time")
        axes_ts[i].set_ylabel(target)
        axes_ts[i].legend(framealpha=0.5)
        axes_ts[i].grid(True)

    # Hide empty subplots if any
    for j in range(i+1, len(axes_res)):
        axes_res[j].axis('off')
        axes_ts[j].axis('off')

    plt.tight_layout(pad=3.0)
    fig_res.suptitle(f"{target} Residuals per Horizon", fontsize=16)
    fig_ts.suptitle(f"{target} Predicted vs Observed per Horizon", y=1.02, fontsize=16)
    plt.show()

    return results
