import eda_dpp_utils
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Modified Z-Score for anomaly labeling (ground truth)
def modified_z_score(y, threshold=3.5):
    m = np.median(y)
    mad = np.median(np.abs(y - m))
    mod_z = 0.6745 * (y - m) / mad
    return (np.abs(mod_z) > threshold).astype(int)

# Select best contamination based on average precision
def select_best_contamination(df, features, target, cont_vals=[0.01,0.02,0.033,0.05,0.1]):
    df['is_anomaly'] = modified_z_score(df[target])
    best_ap = -1
    best_c = cont_vals[0]

    for c in cont_vals:
        iso = IsolationForest(contamination=c, random_state=42)
        iso.fit(df[features])
        preds = (iso.predict(df[features]) == -1).astype(int)
        avg_prec = average_precision_score(df['is_anomaly'], preds)
        if avg_prec > best_ap:
            best_ap = avg_prec
            best_c = c
    return best_c

def detect_anomalies_all_targets(df, features, targets, cont_vals=[0.01,0.02,0.033,0.05,0.1]):
    for target in targets:
        print(f"\nProcessing target: {target}")

        # If NMHC, drop all NaN rows
        if target == "NMHC(GT)":
            df_fit = df.dropna(subset=[target]).copy()
        else:
            df_fit = df.copy()

        # Compute best contamination using ONLY df_fit
        best_c = select_best_contamination(df_fit.copy(), features, target, cont_vals)
        print(f"Best contamination for {target}: {best_c}")

        # 2. Fit Isolation Forest
        iso = IsolationForest(contamination=best_c, random_state=42)
        iso.fit(df_fit[features])

        # Predict ONLY for df_fit
        preds_fit = iso.predict(df_fit[features])

        # 3. Allocate anomaly column to FULL df (same index)
        anomaly_col = f"anomaly_{target}"
        df[anomaly_col] = np.nan               # initialize full column
        df.loc[df_fit.index, anomaly_col] = preds_fit  # fill in predictions

        # 4. Plot anomalies
        plt.figure(figsize=(14,6))
        plt.plot(df.index, df[target], label=target)

        anomalies = df[df[anomaly_col] == -1]
        plt.scatter(anomalies.index, anomalies[target],
                    color='red', s=30, label='Anomaly')

        plt.xlabel("Time")
        plt.ylabel(target)
        plt.title(f"{target} with Isolation Forest Anomalies (cont={best_c})")
        plt.legend()
        plt.show()

        # 5. PR curve 
        df_fit['is_anomaly'] = modified_z_score(df_fit[target])
        anomaly_scores = -iso.decision_function(df_fit[features])

        precision, recall, thresholds = precision_recall_curve(
            df_fit['is_anomaly'], anomaly_scores
        )
        avg_prec = average_precision_score(df_fit['is_anomaly'], anomaly_scores)

        preds_best = (preds_fit == -1).astype(int)
        tp = ((preds_best == 1) & (df_fit['is_anomaly'] == 1)).sum()
        fp = ((preds_best == 1) & (df_fit['is_anomaly'] == 0)).sum()
        fn = ((preds_best == 0) & (df_fit['is_anomaly'] == 1)).sum()

        precision_best = tp / (tp + fp + 1e-8)
        recall_best = tp / (tp + fn + 1e-8)

        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label=f"{target} (AP={avg_prec:.3f})")
        plt.scatter(recall_best, precision_best, color='red', s=100,
                    label=f"Best cont={best_c}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve for {target}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return df

# Identify trends in anomalies
def summarise_anomalies(df, targets, time_col='T', weekday_col='weekday', bins=10):
    summary_list = []

    for target in targets:
        anomaly_col = f'anomaly_{target}'
        
        # 1. Weekday vs Weekend anomaly fractions
        df['is_weekend'] = df[weekday_col].isin([5,6])
        ct = pd.crosstab(df['is_weekend'], df[anomaly_col], normalize='index')
        
        for is_weekend_val in ct.index:
            anomaly_frac = ct.loc[is_weekend_val, -1] if -1 in ct.columns else 0.0
            summary_list.append({
                'target': target,
                'analysis': 'weekday_vs_weekend',
                'is_weekend': is_weekend_val,
                'anomaly_fraction': anomaly_frac
            })
        
        # 2. Group anomalies by bins of timestamp column
        grouped = df.groupby(pd.cut(df[time_col], bins=bins))[anomaly_col].apply(lambda x: (x==-1).mean())
        for t_bin, anomaly_frac in grouped.items():
            summary_list.append({
                'target': target,
                'analysis': 'T_bin',
                'T_bin': t_bin,
                'anomaly_fraction': anomaly_frac
            })

    summary_df = pd.DataFrame(summary_list)
    return summary_df


def compare_clean_vs_unclean(df, target, features):
    # -------------------------
    # 1. Train model on original (uncleaned) data
    # -------------------------
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        df[features], df[target], test_size=0.2, shuffle=False
    )

    model_unclean = RandomForestRegressor(random_state=42)
    model_unclean.fit(X_train_u, y_train_u)
    preds_unclean = model_unclean.predict(X_test_u)
    residuals_unclean = y_test_u - preds_unclean

    # -------------------------
    # 2. Train model on anomaly-cleaned data
    # -------------------------
    # Keep only normal rows for this target
    clean_df = df[df[f'anomaly_{target}'] == 1]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        clean_df[features], clean_df[target], test_size=0.2, shuffle=False
    )

    model_clean = RandomForestRegressor(random_state=42)
    model_clean.fit(X_train_c, y_train_c)
    preds_clean = model_clean.predict(X_test_c)
    residuals_clean = y_test_c - preds_clean

    # -------------------------
    # 3. Side-by-side residual plot
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16,5), sharey=True)

    axes[0].scatter(y_test_u.index, residuals_unclean)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_title(f"Uncleaned Model Residuals ({target})")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residuals")

    axes[1].scatter(y_test_c.index, residuals_clean)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title(f"Cleaned Model Residuals ({target})")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

    return {
        "unclean_mae": abs(residuals_unclean).mean(),
        "clean_mae": abs(residuals_clean).mean()
    }