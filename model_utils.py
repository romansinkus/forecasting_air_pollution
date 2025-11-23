# model_utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

# -------------------------
# Regression / Lasso utils
# -------------------------
def lassoRegression(X, y, alphas, n_splits=5, max_iter=10000, random_state=42):
    """
    Fit Lasso with time-series cross-validation (TimeSeriesSplit).
    Returns:
      lasso_model, selected_features, coef_df, best_alpha
    """
    # Ensure X is DataFrame & y is Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]

    # TimeSeriesSplit for CV (no leakage)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=max_iter, random_state=random_state)
    lasso_cv.fit(X, y)

    best_alpha = float(lasso_cv.alpha_)
    print(f"Best alpha selected by CV: {best_alpha:.5e}")

    # Fit final Lasso using best alpha on full training set
    lasso_model = Lasso(alpha=best_alpha, max_iter=max_iter, random_state=random_state)
    lasso_model.fit(X, y)

    # Selected features (non-zero coefficients)
    selected_idx = np.where(lasso_model.coef_ != 0)[0]
    selected_features = X.columns[selected_idx].tolist()

    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lasso_model.coef_})
    print("\nSelected (non-zero) coefficients:")
    print(coef_df[coef_df['coefficient'] != 0].sort_values(by='coefficient', key=lambda s: s.abs(), ascending=False))

    return lasso_model, selected_features, coef_df, best_alpha


def plotResiduals(lasso_model, X_test, y_test, selected_features=None, target_name=None):
    """
    Plot residual diagnostics for a fitted Lasso model.
    X_test: DataFrame or numpy array
    y_test: Series, 1D array or single-column DataFrame
    selected_features: list of feature names to use from X_test (if None, use X_test.columns)
    """
    # Convert to DataFrame / Series
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]
    y_test = pd.Series(y_test).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    target_name = target_name or "Target"

    # Keep only selected features (if provided). If a feature is missing, create zero column.
    if selected_features is not None:
        for col in selected_features:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[selected_features]

    # Predictions & residuals
    y_pred = pd.Series(lasso_model.predict(X_test))
    residuals = y_test - y_pred

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{target_name} RMSE: {rmse:.4f}")

    # Detect outliers via IQR
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    outliers = np.where((residuals < q1 - 1.5 * iqr) | (residuals > q3 + 1.5 * iqr))[0]

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Residual Analysis ({target_name})", fontsize=16, y=1.02)

    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], s=40, alpha=0.6)
    if len(outliers):
        sns.scatterplot(x=y_pred.iloc[outliers], y=residuals.iloc[outliers], ax=axes[0], s=80, color='red')
    axes[0].axhline(0, linestyle='--', color='black')
    axes[0].set(xlabel='Predicted', ylabel='Residual', title='Residuals vs Predicted')

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")

    sns.lineplot(x=np.arange(len(y_test)), y=y_test, ax=axes[2], label="Observed")
    sns.lineplot(x=np.arange(len(y_test)), y=y_pred, ax=axes[2], label="Predicted")
    if len(outliers):
        axes[2].scatter(outliers, y_test.iloc[outliers], color='red', s=50, label="Outliers")
    axes[2].set(xlabel='Index', ylabel=target_name, title='Predicted vs Actual (Time Series)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return outliers


# -------------------------
# Classification utils
# -------------------------
def discretize_CO(df):
    bins = [-np.inf, 1.5, 2.5, np.inf]
    labels = ['low', 'mid', 'high']
    df = df.copy()
    df['CO_class'] = pd.cut(df['CO(GT)'], bins=bins, labels=labels)
    return df


def add_future_classes(df, target="CO(GT)"):
    df = df.copy()
    bins = [-np.inf, 1.5, 2.5, np.inf]
    labels = ['low', 'mid', 'high']
    
    for k in [1, 6, 12, 24]:
        shifted = df[target].shift(-k)
        cat = pd.cut(shifted, bins=bins, labels=labels)
        df[f"CO_class_t{k}"] = cat  # keep as categorical
    return df



def classificationSplit(df, target_col):
    df = df.copy().reset_index().rename(columns={"index": "timestamp"})
    df = df.dropna(subset=[target_col])  # drop NaNs here

    # Temporal split
    train_df = df[df["timestamp"].dt.year < 2005]
    test_df  = df[df["timestamp"].dt.year == 2005]

    # Drop CO targets to avoid leakage
    drop_cols = ["CO(GT)", "CO_class", 
                 "CO_class_t1", "CO_class_t6", "CO_class_t12", "CO_class_t24", 
                 "timestamp"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test  = test_df.drop(columns=drop_cols, errors="ignore")

    # Label encode **after dropping NaNs**
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_test  = le.transform(test_df[target_col])

    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, X_train.columns.tolist(), le


def logisticLasso(X_train, y_train, X_test, y_test, alphas=None, target_name=None):
    """
    Fit L1-penalised logistic regression using time-series CV.
    Returns: best_model, coef_df, y_pred
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 10)

    # logistic with L1 penalty
    logit = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, multi_class='ovr')

    # Build parameter grid for C = 1/alpha
    C_vals = (1.0 / alphas).tolist()
    param_grid = {'C': C_vals}

    tscv = TimeSeriesSplit(n_splits=5)
    clf = GridSearchCV(logit, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    print(f"\n=== Logistic Lasso for {target_name} ===")
    print("Best C (inverse of regularization strength):", clf.best_params_['C'])

    # Coefficients: features x classes (if binary/multiclass using one-vs-rest shape)
    coef_df = pd.DataFrame(best_model.coef_.T, index=X_train.columns, columns=best_model.classes_)
    print("\nNon-zero coefficients (feature x classes):")
    nz = coef_df.loc[(coef_df.abs().sum(axis=1) > 0), :]
    print(nz)

    # Predict and evaluate
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({target_name})")
    plt.show()

    # Per-class accuracy (safe division)
    denom = cm.sum(axis=1).astype(float)
    per_class_acc = np.divide(np.diagonal(cm), denom, out=np.zeros_like(np.diagonal(cm), dtype=float), where=denom != 0)
    classes = [str(c) for c in best_model.classes_]
    plt.figure(figsize=(6, 4))
    plt.bar(classes, per_class_acc * 100)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    for i, acc in enumerate(per_class_acc):
        plt.text(i, acc * 100 + 1, f"{acc * 100:.1f}%", ha='center', fontsize=10)
    plt.show()

    return best_model, coef_df, y_pred

# -------------------------
# Utility splitter for regression main workflow
# -------------------------
def splitSets(df, targets, horizons=[1,6,12,24]):
    """
    Prepares train/test split for regression for multiple horizons.
    Returns:
        feature_names: list of feature columns
        X_train_values: np.array
        y_train_df: DataFrame with all shifted targets
        X_test_values: np.array
        y_test_df: DataFrame with all shifted targets
    """
    df = df.copy().reset_index().rename(columns={'index':'timestamp'})

    # Add shifted targets
    shifted_cols = []
    for t in targets:
        for k in horizons:
            col = f"{t}_t{k}"
            if col not in df.columns:
                df[col] = df[t].shift(-k)
            shifted_cols.append(col)

    # Train/test split by year
    train_df = df[df['timestamp'].dt.year < 2005]
    test_df  = df[df['timestamp'].dt.year == 2005]

    # Features: drop all original & shifted target columns + timestamp
    drop_cols = targets + shifted_cols + ['timestamp']
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    X_test  = test_df.drop(columns=drop_cols, errors='ignore')

    # Targets: all shifted targets
    y_train = train_df[shifted_cols].copy()
    y_test  = test_df[shifted_cols].copy()

    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    feature_names = X_train.columns.tolist()
    return feature_names, X_train.values, y_train, X_test.values, y_test


def add_future_regression_targets(df, targets):
    df = df.copy()
    for t in targets:
        for k in [1, 6, 12, 24]:
            df[f"{t}_t{k}"] = df[t].shift(-k)
    return df

