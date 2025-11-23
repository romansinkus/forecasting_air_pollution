import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# -------------------------
# 1️⃣ Feature Preparation
# -------------------------
def featurePrep(df, targets):
    df = df.copy()
    # if 'timestamp' in df.columns:
    #     df['timestamp'] = df['timestamp']

    # Sensor transforms (optional)
    if 'PT08.S3(NOx)' in df.columns:
        df['PT08.S3(NOx)'] = 1 / ((df['PT08.S3(NOx)'] + 1e-6)**2)
    if 'PT08.S2(NMHC)' in df.columns:
        df['PT08.S2(NMHC)'] = df['PT08.S2(NMHC)'] ** 2
    
    # Add lag-0
    for t in targets:
        df[f"{t}_lag0"] = df[t]
    
    # One-hot encode binned features
    bin_cols = [c for c in ['hour_bin', 'month_bin', 'weekday_bin'] if c in df.columns]
    if bin_cols:
        df = pd.get_dummies(df, columns=bin_cols, drop_first=True)
    
    # Convert bools to int
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    
    return df

# -------------------------
# Add future targets
# -------------------------
def add_future_targets(df, targets, horizons=[1,6,12,24]):
    df = df.copy()
    for t in targets:
        for h in horizons:
            df[f"{t}_t{h}"] = df[t].shift(-h)
    return df

# -------------------------
# Split by horizon
# -------------------------
def split_by_horizon(df, target_col):
    """
    Split data into train/test sets for a given target and horizon.
    Uses index.year for chronological splitting (2004 train, 2005 test).
    Drops rows with NaNs in features or target.
    """
    df_h = df.copy()
    
    # Drop rows where target is NaN
    df_h = df_h.dropna(subset=[target_col])
    
    # Features: all except target
    feature_cols = [c for c in df_h.columns if c != target_col]
    
    X = df_h[feature_cols]
    y = df_h[target_col]
    
    # Drop rows with NaN in any numeric feature
    numeric_cols = X.select_dtypes(include=np.number).columns
    X = X.dropna(subset=numeric_cols)
    y = y.loc[X.index]  # align y
    
    # Chronological split
    train_mask = X.index.year == 2004
    test_mask  = X.index.year == 2005
    
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    
    if X_train.empty or X_test.empty:
        print(f"Skipping {target_col}: empty train/test")
        return None, None, None, None
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test

# -------------------------
# Lasso Regression
# -------------------------
def lasso_regression(X, y, alphas=np.logspace(-4,1,20), n_splits=5, max_iter=10000, random_state=42):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=max_iter, random_state=random_state)
    lasso_cv.fit(X, y)
    
    best_alpha = float(lasso_cv.alpha_)
    lasso_model = Lasso(alpha=best_alpha, max_iter=max_iter, random_state=random_state)
    lasso_model.fit(X, y)
    
    selected_features = X.columns[lasso_model.coef_ != 0].tolist()
    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lasso_model.coef_})
    
    return lasso_model, selected_features, coef_df, best_alpha

# -------------------------
# Residual Analysis
# -------------------------
def plot_residuals(model, X_test, y_test, selected_features=None, title=None):
    if selected_features:
        for f in selected_features:
            if f not in X_test.columns:
                X_test[f] = 0
        X_test = X_test[selected_features]

    y_pred = pd.Series(model.predict(X_test))
    residuals = y_test - y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{title} RMSE: {rmse:.4f}")

    # Plots
    fig, axes = plt.subplots(1,3,figsize=(20,5))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], s=40, alpha=0.6)
    axes[0].axhline(0, linestyle='--', color='black')
    axes[0].set(xlabel='Predicted', ylabel='Residual', title='Residuals vs Predicted')
    
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")
    
    sns.lineplot(x=np.arange(len(y_test)), y=y_test, ax=axes[2], label="Observed")
    sns.lineplot(x=np.arange(len(y_test)), y=y_pred, ax=axes[2], label="Predicted")
    axes[2].set_title("Predicted vs Observed")
    axes[2].set(xlabel='Index', ylabel='Target')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return rmse
