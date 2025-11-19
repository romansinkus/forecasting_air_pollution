import eda_dpp_utils
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

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

# Detect anomalies for multiple targets with Isolation Forest
def detect_anomalies_all_targets(df, features, targets, cont_vals=[0.01,0.02,0.033,0.05,0.1]):
    for target in targets:
        print(f"\nProcessing target: {target}")
        # 1. Compute best contamination
        best_c = select_best_contamination(df.copy(), features, target, cont_vals)
        print(f"Best contamination for {target}: {best_c}")
        
        # 2. Fit final Isolation Forest
        iso = IsolationForest(contamination=best_c, random_state=42)
        iso.fit(df[features])
        df[f'anomaly_{target}'] = iso.predict(df[features])  # -1 = anomaly, 1 = normal
        
        # 3. Plot time series with anomalies
        plt.figure(figsize=(14,6))
        plt.plot(df.index, df[target], label=target)
        anomalies = df[df[f'anomaly_{target}'] == -1]
        plt.scatter(anomalies.index, anomalies[target], color='red', s=30, label='Anomaly')
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.title(f"{target} with Isolation Forest Anomalies (cont={best_c})")
        plt.legend()
        plt.show()
        
        # 4. PR curve vs Z-score anomalies
        df['is_anomaly'] = modified_z_score(df[target])
        anomaly_scores = -iso.decision_function(df[features])
        precision, recall, thresholds = precision_recall_curve(df['is_anomaly'], anomaly_scores)
        avg_prec = average_precision_score(df['is_anomaly'], anomaly_scores)
        
        # Compute precision/recall for the chosen contamination
        preds_best = (iso.predict(df[features]) == -1).astype(int)
        tp = ((preds_best == 1) & (df['is_anomaly'] == 1)).sum()
        fp = ((preds_best == 1) & (df['is_anomaly'] == 0)).sum()
        fn = ((preds_best == 0) & (df['is_anomaly'] == 1)).sum()
        precision_best = tp / (tp + fp + 1e-8)
        recall_best = tp / (tp + fn + 1e-8)
        
        # Plot PR curve with the best contamination point
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label=f"{target} (AP={avg_prec:.3f})")
        plt.scatter(recall_best, precision_best, color='red', s=100, label=f"Best cont={best_c}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve for {target}")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    df = df.drop(columns=['is_anomaly']) # clean up temporary column
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