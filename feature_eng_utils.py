import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

def addLagFeatures(df, lags, targets):
    for tgt in targets:
        for lag in lags:
            df[f'{tgt}_lag{lag}'] = df[tgt].shift(lag)
    return df

def addMovingAverages(df, windows, targets):
    for tgt in targets:
        for w in windows:
            df[f'{tgt}_MA{w}'] = df[tgt].rolling(window=w, min_periods=1).mean()
    return df

def LagMACorrelations(df, targets):
    rows = []
    for tgt in targets:
        ma_cols = [c for c in df.columns if c.startswith(f'{tgt}_MA')]
        for col in ma_cols:
            corr = df[[tgt, col]].corr().iloc[0,1]
            rows.append((tgt, col, corr))
        
        lag_cols = [c for c in df.columns if c.startswith(f'{tgt}_lag')]
        for col in lag_cols:
            corr = df[[tgt, col]].corr().iloc[0,1]
            rows.append((tgt, col, corr))
    
    return pd.DataFrame(rows, columns=['target','feature','correlation'])

def plot_lag_and_time_scatter(df, targets, lags):

    time_feats = ['hour', 'weekday', 'month']

    for tgt in targets:
        lag_cols = [f"{tgt}_lag{lag}" for lag in lags if f"{tgt}_lag{lag}" in df.columns]
        plot_cols = lag_cols + [feat for feat in time_feats if feat in df.columns]

        if not plot_cols:
            print(f"No features to plot for target {tgt}.")
            continue

        n_cols = len(plot_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4), squeeze=False)

        for j, feat in enumerate(plot_cols):
            ax = axes[0][j]
            sns.scatterplot(data=df, x=feat, y=tgt, alpha=0.5, s=20, ax=ax)
            ax.set_title(f"{tgt} vs {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel(tgt)

        plt.tight_layout()
        plt.show()
    return

def bin_features(df):
    # hr: bins of 6 (0-5, 6-11, 12-17, 18-23)
    hour_bins = [0, 6, 12, 18, 24]
    hour_labels = ['0-5','6-11','12-17','18-23']
    binned_features = ['hour_bin', 'is_sunday', 'month_bin']
    df['hour_bin'] = pd.cut(df['hour'], bins=hour_bins, labels=hour_labels, right=False)

    # weekday: Sunday or not Sunday
    df['is_sunday'] = (df['weekday'] == 6).astype(int)  # 0 = monday

    # month: bins of 3 months (1-3, 4-6, 7-9, 10-12)
    month_bins = [1,4,7,10,13]
    month_labels = ['Jan-Mar','Apr-Jun','July-Sep', 'Oct-Dec']
    df['month_bin'] = pd.cut(df['month'], bins=month_bins, labels=month_labels, right=False)
    return df, binned_features

def plot_binned_features(df, targets, binned_features):
    for feat in binned_features:
        n_targets = len(targets)
        n_cols = 2  # number of columns in the subplot grid
        n_rows = (n_targets + n_cols - 1) // n_cols  # ceil division for rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten()  # flatten for easy indexing

        for i, tgt in enumerate(targets):
            sns.boxplot(x=feat, y=tgt, data=df, ax=axes[i])
            axes[i].set_title(f'{tgt} vs {feat}')
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel(tgt)

        # Remove empty subplots if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    return

def binned_correlation(df, targets, binned_features):
    corr_dict = {}
    
    for feat in binned_features:
        # One-hot encode categorical/binned features (skip binary like is_sunday)
        if df[feat].nunique() > 2:
            feat_dummies = pd.get_dummies(df[feat], prefix=feat)
        else:
            feat_dummies = df[[feat]]
        
        for col in feat_dummies.columns:
            for tgt in targets:
                corr = feat_dummies[col].corr(df[tgt])
                corr_dict[(col, tgt)] = corr
                
    # Convert to DataFrame
    corr_df = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['correlation'])
    corr_df.index = pd.MultiIndex.from_tuples(corr_df.index, names=['feature','target'])
    return corr_df