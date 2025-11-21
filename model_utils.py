
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from feature_eng_utils import addLagFeatures, bin_features

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

def featurePrep(df):
    x = 'PT08.S3(NOx)'
    df[x] = 1/((df[x] + 1e-6)**2)
    x = 'PT08.S2(NMHC)'
    df[x] = (df[x]**2)

    df = addLagFeatures(df, [1], targets)
    df, binned_features = bin_features(df)

    # 1 hot encoding the bins
    cat_cols = ['hour_bin', 'month_bin']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Select only dummy columns (they will contain True/False)
    dummy_cols = [c for c in df.columns if any(c.startswith(col) for col in cat_cols)]
    # Convert True/False to 0/1 integers
    df[dummy_cols] = df[dummy_cols].astype(int)
    df = df.drop(columns=['hour', 'weekday', 'month']) # remove OG derived cols

    df = df.iloc[1:] # lag1 cols have nan in 1st row

    return df

def splitSets(df):
    # train test split. split into (train + validation)/test
    feature_names = df.columns
    df = df.reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    train_df = df[df['timestamp'].dt.year < 2005]
    test_df = df[df['timestamp'].dt.year == 2005]

    X_train = train_df.drop(columns=targets + ['timestamp'])
    y_train = train_df[targets]
    
    X_test = test_df.drop(columns=targets + ['timestamp'])
    y_test = test_df[targets]

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # Feature names
    feature_names = X_train.columns.tolist()
    return feature_names, X_train.values, y_train, X_test.values, y_test

def alphaCVPlot(alphas, rmse_scores):
    # plt.figure(figsize=(8,5))
    # plt.plot(alphas, rmse_scores, marker='o')
    # plt.xscale('log')
    # plt.xlabel('Alpha')
    # plt.ylabel('CV RMSE')
    # plt.title('Lasso CV: RMSE vs Alpha')
    # plt.grid(True)
    # plt.show()

    optimal_alpha = alphas[np.argmin(rmse_scores)]
    print(f"Optimal alpha: {optimal_alpha}")
    print(f"Minimum CV RMSE: {min(rmse_scores)}")
    return optimal_alpha

"""
    Fits Lasso regression with cross-validated alpha (L1 regularization).
    
    Parameters:
    - X: DataFrame or array of features
    - y: Series or array of target
    - alphas: array of alpha values to search over
    - n_splits: number of CV folds
    - max_iter: maximum iterations for solver
    
    Returns:
    - lasso_model: fitted Lasso model with best alpha
    - selected_features: list of features with non-zero coefficients
    - coef_df: DataFrame of coefficients for all features
    - best_alpha: alpha selected via CV
"""

def lassoRegression(X, y, alphas, n_splits=5, max_iter=10000, random_state=42):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:,0]
    
    # LassoCV automatically finds best alpha
    lasso_cv = LassoCV(alphas=alphas, cv=n_splits, max_iter=max_iter, random_state=random_state)
    lasso_cv.fit(X, y)
    
    best_alpha = lasso_cv.alpha_
    print(f"Best alpha selected by CV: {best_alpha:.5f}")
    
    # Fit final model with best alpha
    lasso_model = Lasso(alpha=best_alpha, max_iter=max_iter)
    lasso_model.fit(X, y)
    
    # Collect non-zero features
    selected_idx = np.where(lasso_model.coef_ != 0)[0]
    selected_features = X.columns[selected_idx].tolist()
    
    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lasso_model.coef_})
    
    print("\nSelected features with non-zero coefficients:")
    print(coef_df[coef_df['coefficient'] != 0].sort_values(by='coefficient', key=abs, ascending=False))
    
    return lasso_model, selected_features, coef_df, best_alpha

"""
    Plots residuals and identifies outliers for a scikit-learn Lasso model.

    Parameters:
    - lasso_model: fitted scikit-learn Lasso model
    - X_test: DataFrame or array of test features
    - y_test: Series, array, or single-column DataFrame of true target
    - target_name: optional name for plots
    Returns:
    - outliers: indices of detected outliers based on IQR
"""

def plotResiduals(lasso_model, X_test, y_test, selected_features=None, target_name=None):
    # Ensure DataFrame/Series
    if isinstance(X_test, np.ndarray): X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1]==1: y_test = y_test.iloc[:,0]
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    target_name = target_name or "Target"

    # Align test set to training features
    if selected_features is not None:
        X_test = X_test.reindex(columns=selected_features, fill_value=0)

    # Predictions & residuals
    y_pred = pd.Series(lasso_model.predict(X_test))
    residuals = y_test - y_pred
    # Compute RMSE
    rmse = rmse = root_mean_squared_error(y_test, y_pred)
    print(f"{target_name} RMSE: {rmse:.4f}")

    # Outliers using IQR
    q1, q3 = np.percentile(residuals, [25,75])
    iqr = q3 - q1
    outliers = np.where((residuals < q1-1.5*iqr) | (residuals > q3+1.5*iqr))[0]

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1,3,figsize=(20,5))
    fig.suptitle(f"Residual Analysis ({target_name})", fontsize=16, y=1.02)

    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], s=40, alpha=0.6)
    if len(outliers): sns.scatterplot(x=y_pred.iloc[outliers], y=residuals.iloc[outliers], ax=axes[0], s=80, color='red')
    axes[0].axhline(0, linestyle='--', color='black'); axes[0].set(xlabel='Predicted', ylabel='Residual', title='Residuals vs Predicted')

    stats.probplot(residuals, dist="norm", plot=axes[1]); axes[1].set_title("Q-Q Plot")

    sns.lineplot(x=np.arange(len(y_test)), y=y_test, ax=axes[2], label="Observed", color="blue")
    sns.lineplot(x=np.arange(len(y_test)), y=y_pred, ax=axes[2], label="Predicted", color="green")
    if len(outliers): axes[2].scatter(outliers, y_test.iloc[outliers], color='red', s=50, label="Outliers")
    axes[2].set(xlabel='Index', ylabel=target_name, title='Predicted vs Actual (Time Series)'); axes[2].legend()

    plt.tight_layout(); plt.show()
    return outliers


"""
    Prepares data for multiclass classification using CO(GT) as target.
    Discretizes CO(GT) into low, mid, high and splits based on year.
    
    Parameters:
    - df: DataFrame containing features + target_col
    - target_col: name of the target column
    
    Returns:
    - feature_names: list of feature column names
    - X_train: training features (numpy array)
    - y_train: training target (Series)
    - X_test: test features (numpy array)
    - y_test: test target (Series)
    - le: fitted LabelEncoder to decode class integers
"""
def classificationSplitSets(df, target_col='CO(GT)'):
    df = df.copy().reset_index().rename(columns={'index': 'timestamp'})
    
    # Discretize target into 3 classes
    bins = [-np.inf, 1.5, 2.5, np.inf]
    labels = ['low', 'mid', 'high']
    df['CO_class'] = pd.cut(df[target_col], bins=bins, labels=labels)
    
    # Split train/test by year
    train_df = df[df['timestamp'].dt.year < 2005]
    test_df  = df[df['timestamp'].dt.year == 2005]
    
    # Features and target
    X_train = train_df.drop(columns=[target_col, 'CO_class', 'timestamp'])
    X_test  = test_df.drop(columns=[target_col, 'CO_class', 'timestamp'])
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['CO_class'])
    y_test  = le.transform(test_df['CO_class'])
    
    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, X_train.columns.tolist(), le

# Discretize CO(GT) into 3 classes
def discretize_CO(df):
    bins = [-np.inf, 1.5, 2.5, np.inf]
    labels = ['low', 'mid', 'high']
    df['CO_class'] = pd.cut(df['CO(GT)'], bins=bins, labels=labels)
    return df

"""
    Fit L1-regularized logistic regression with cross-validation,
    print coefficients, and plot confusion matrix.
    
    Parameters:
    - X_train, X_test: DataFrames of features (scaled)
    - y_train, y_test: Series of multiclass target
    - alphas: list of regularization strengths (C = 1/alpha)
    - target_name: optional name for printing
"""
def logisticLasso(X_train, y_train, X_test, y_test, alphas=None, target_name=None):
    # Default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 4, 10)
    
    # Setup Logistic Regression with L1 (Lasso) penalty
    logit = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=5000
    )
    
    # GridSearch over inverse regularization strength C
    param_grid = {'C': 1/alphas}  # C = 1/lambda

    clf = GridSearchCV(logit, param_grid, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    
    best_model = clf.best_estimator_
    print(f"\n=== Logistic Lasso for {target_name} ===")
    print("Best C (inverse of regularization strength):", clf.best_params_['C'])
    
    # Display coefficients
    coef_df = pd.DataFrame(best_model.coef_.T, index=X_train.columns, columns=best_model.classes_)
    print("\nCoefficients (features x classes):")
    print(coef_df[coef_df.abs().sum(axis=1) > 0])  # only non-zero for clarity
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({target_name})")
    plt.show()
    
    classes = ['low', 'mid', 'high']
    per_class_acc = cm.diagonal() / cm.sum(axis=1)  # TP / total per class
    plt.figure(figsize=(6,4))
    plt.bar(classes, per_class_acc*100, color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("CO_class Per-Class Accuracy")
    
    # Annotate accuracy on bars
    for i, acc in enumerate(per_class_acc):
        plt.text(i, acc*100 + 1, f"{acc*100:.1f}%", ha='center', fontsize=10)
    plt.show()
    
    return best_model, coef_df, y_pred

