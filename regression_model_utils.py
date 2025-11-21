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
        # NMHC(GT) special: split 2004 into train/test (e.g., 80/20 split)
        # Drop rows where NMHC is NaN
        df_clean = df.dropna(subset=['NMHC(GT)']).copy()
        split_idx = int(0.8 * len(df_clean))
        train_df = df_clean.iloc[:split_idx].copy()
        test_df  = df_clean.iloc[split_idx:].copy()

    # Use all columns that are NOT targets
    features = [c for c in df.columns if c not in targets]

    results = {}

    for h in horizons:
        print(f"\n--- Horizon: {h} hour(s) ahead ---")

        # Shift target for the horizon
        train_valid = train_df.copy()
        test_valid  = test_df.copy()

        train_valid[f"{target}_future_{h}h"] = train_valid[target].shift(-h)
        test_valid[f"{target}_future_{h}h"] = test_valid[target].shift(-h)

        # Drop rows with NaN after shift
        train_valid = train_valid.dropna(subset=[f"{target}_future_{h}h"])
        test_valid  = test_valid.dropna(subset=[f"{target}_future_{h}h"])

        X_train = train_valid[features]
        y_train = train_valid[f"{target}_future_{h}h"]

        X_test  = test_valid[features]
        y_test  = test_valid[f"{target}_future_{h}h"]

        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # RMSE for random forest
        rmse = root_mean_squared_error(y_test, y_pred)
        rel_rmse = rmse / y_test.mean()

        # Store results
        results = {
            "y_test": y_test,
            "y_pred": y_pred,
            "rmse": rmse,
            "relative_rmse": rel_rmse
        }

        # ----------------------------
        # Visualisations
        # ----------------------------
        residuals = y_test - y_pred

        # Residuals
        plt.figure(figsize=(12,4))
        plt.plot(residuals)
        plt.title(f"Residuals for {target} ({h}h ahead)")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.show()

        print(f"Random Forest Relative RMSE = {rel_rmse:.3f}")
        # Random Forest predicted vs Observed 
        plt.figure(figsize=(12,4))
        plt.plot(y_test.index, y_test, label="Observed")
        plt.plot(y_test.index, y_pred, label=f"Random Forest Predicted: RMSE = {rel_rmse:.3f}")
        plt.title(f"{target}: Random Forest Predicted vs Observed ({h}h ahead)")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.legend(framealpha=0.5)
        plt.grid(True)
        plt.show()

        # Naive baseline vs Observed
        y_naive = y_test.shift(h).dropna()
        common_idx = y_test.index.intersection(y_naive.index)
        y_test_aligned = y_test.loc[common_idx]
        y_naive_aligned = y_naive.loc[common_idx]

        # RMSE for naive baseline
        naive_rmse = root_mean_squared_error(y_test_aligned, y_naive_aligned)
        naive_rel_rmse = naive_rmse / y_test_aligned.mean()
        print(f"Naive Baseline Relative RMSE = {naive_rel_rmse:.3f}")

        plt.figure(figsize=(12,4))
        plt.plot(y_test_aligned.index, y_test_aligned, label="Observed", alpha=0.8)
        plt.plot(y_test_aligned.index, y_naive_aligned, label=f"Naive Baseline: RMSE = {naive_rel_rmse:.3f}")
        plt.title(f"{target}: Naive Baseline vs Observed ({h}h ahead)")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.legend(framealpha=0.5)
        plt.grid(True)
        plt.show()

    return results
