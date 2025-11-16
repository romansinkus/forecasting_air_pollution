

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ucimlrepo import fetch_ucirepo
from tabulate import tabulate


def preProcessing(): 
    air_quality = fetch_ucirepo(id=360) 
    X = air_quality.data.features 
    y = air_quality.data.targets
    df = pd.concat([X, y], axis=1)

    df.replace(-200, np.nan, inplace=True) 
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].replace(-200, np.nan) 
    df[num_cols] = (df[num_cols].ffill() + df[num_cols].bfill()) / 2

    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['Date','Time'], inplace=True)
    df_unnormalised = df.copy()
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    df['hour'], df['weekday'], df['month'] = df.index.hour, df.index.weekday, df.index.month
    return df, df_unnormalised, num_cols


def summaryStats_corrHeatmap(df, df_unnormalised, num_cols):
    exc = ['hour', 'weekday', 'month']
    numericCols = [c for c in df_unnormalised.select_dtypes(include='number').columns if c not in exc]
    summary_stats = df_unnormalised[numericCols].describe().T
    summary_stats = summary_stats.round(4)
    print(tabulate(summary_stats, headers='keys', tablefmt='psql'))

    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(len(num_cols)*0.7 + 5, len(num_cols)*0.7 + 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return summary_stats, corr_matrix


def normalisedTimeSeriesPlots(df, num_cols):
    plt.figure(figsize=(15, len(num_cols)*2))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(len(num_cols), 1, i)
        plt.plot(df.index, df[col])
        plt.title(col)
        plt.xlabel('Time')
        plt.ylabel('Normalized value')
    plt.tight_layout()
    plt.show()
    return


def scatter_targetNontarget(df, targets):
    non_target_cols = [c for c in df.columns if '(GT)' not in c and c not in ['hour','weekday','month']]

    for target in targets:
        g = sns.PairGrid(df[non_target_cols + [target]], y_vars=[target], x_vars=non_target_cols, height=2.5)
        g.map(sns.scatterplot, s=15)
        g.fig.suptitle(f'{target} vs Non-targets', y=1.05)
        plt.show()
    return


def best_distribution_fit(series):
    dist_list = [stats.norm, stats.lognorm, stats.expon, stats.gamma, stats.beta]
    hist, bins = np.histogram(series.dropna(), bins=30, density=True)
    bc = (bins[1:] + bins[:-1])/2
    best_sse = np.inf
    for dist in dist_list:
        params = dist.fit(series.dropna())
        sse = np.sum((hist - dist.pdf(bc, *params))**2)
        if sse < best_sse:
            best_sse, best_fit_name, best_params = sse, dist.name, params
    return best_fit_name, best_params


def plot_distribution_fit(df_unnormalised):
    original_cols = [c for c in df_unnormalised.columns if c not in ['hour','weekday','month']]
    fig, axes = plt.subplots(len(original_cols), 1, figsize=(15, len(original_cols)*3))
    for ax, col in zip(axes, original_cols):
        
        series = df_unnormalised[col].dropna()
        sns.histplot(series, bins=30, kde=False, stat='density', color='skyblue', ax=ax)
        dist_name, params = best_distribution_fit(series)
        x = np.linspace(series.min(), series.max(), 100)
        ax.plot(
            x, 
            getattr(stats, dist_name).pdf(x, *params), 
            'r', 
            label=f"Best fit: {dist_name}\nParams: {', '.join([f'{p:.3g}' for p in params])}"
        )
        ax.set_title(col)
        ax.legend()

    plt.tight_layout()
    plt.show()
    return


