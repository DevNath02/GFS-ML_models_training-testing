"""
visualization.py
================
All visualizations for the Gesture Phase Segmentation project.
Saves plots to 'plots/' directory and displays them inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (save to file)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Global style 
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"  💾 Saved: {path}")



# 1. Countplot – Label Distribution

def plot_label_distribution(df, label_col='phase'):
    """Bar chart showing how many samples belong to each gesture phase."""
    print("\n📊 Plotting label distribution…")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[label_col].value_counts().index
    sns.countplot(data=df, x=label_col, order=order, palette=COLORS, ax=ax)
    ax.set_title("Gesture Phase Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Gesture Phase")
    ax.set_ylabel("Sample Count")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    _save(fig, "01_label_distribution.png")


# 2. Correlation Heatmap

def plot_correlation_heatmap(df):
    """Heatmap of feature correlations (numeric columns only)."""
    print("\n📊 Plotting correlation heatmap…")

    numeric_df = df.select_dtypes(include='number')
    # Drop encoded label and timestamp for cleaner heatmap
    drop = [c for c in ['phase_encoded', 'timestamp'] if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=drop)

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    _save(fig, "02_correlation_heatmap.png")



# 3. Feature Boxplots by Gesture Phase

def plot_feature_boxplots(df, features=('rhx', 'rhy', 'rhz', 'lhx', 'lhy', 'lhz')):
    """Box plots showing feature distribution per gesture phase."""
    print("\n📊 Plotting feature boxplots…")

    feats = [f for f in features if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    phases = sorted(df['phase'].unique())

    for i, feat in enumerate(feats[:6]):
        sns.boxplot(data=df, x='phase', y=feat, palette=COLORS,
                    order=phases, ax=axes[i])
        axes[i].set_title(f"Feature: {feat}", fontsize=11)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=20)

    plt.suptitle("Feature Distributions per Gesture Phase", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, "03_feature_boxplots.png")



# 4. Confusion Matrices (all classifiers)

def plot_confusion_matrices(clf_results, y_test, class_names):
    """One confusion matrix per classifier in a grid layout."""
    print("\n📊 Plotting confusion matrices…")

    n = len(clf_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5 + 1))
    axes = axes.flatten()

    for i, result in enumerate(clf_results):
        cm = confusion_matrix(y_test, result["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[i], cbar=False
        )
        axes[i].set_title(result["name"], fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].tick_params(axis='y', rotation=0)

    # Hide any unused subplot panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices – All Classifiers", fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save(fig, "04_confusion_matrices.png")



# 5. Accuracy Comparison Bar Chart

def plot_accuracy_comparison(clf_results):
    """Horizontal bar chart comparing all classifier accuracies."""
    print("\n Plotting accuracy comparison…")

    names     = [r["name"]     for r in clf_results]
    accs      = [r["accuracy"] for r in clf_results]
    precs     = [r["precision"] for r in clf_results]
    recalls   = [r["recall"]   for r in clf_results]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, accs,   width, label="Accuracy",  color="#4C72B0")
    b2 = ax.bar(x,          precs,  width, label="Precision", color="#55A868")
    b3 = ax.bar(x + width,  recalls, width, label="Recall",   color="#DD8452")

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    label_bars(b1); label_bars(b2); label_bars(b3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Performance Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    _save(fig, "05_accuracy_comparison.png")



# 6. Cluster Visualizations (PCA 2D)

def plot_cluster_visualizations(cluster_results):
    """2D PCA scatter plots for each clustering algorithm."""
    print("\n Plotting cluster visualizations…")

    n = len(cluster_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, cluster_results):
        X_2d   = result["X_2d"]
        labels = result["labels"]
        name   = result["name"]
        unique = np.unique(labels)

        cmap = plt.cm.get_cmap('tab10', len(unique))

        for j, lbl in enumerate(unique):
            mask = labels == lbl
            color = '#888888' if lbl == -1 else cmap(j)
            label_str = "Noise" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], label=label_str, alpha=0.5, s=10, edgecolors='none')

        score_str = f"{result['silhouette']:.4f}" if result['silhouette'] else "N/A"
        ax.set_title(f"{name}\nSilhouette: {score_str}", fontsize=12, fontweight='bold')
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend(markerscale=2, fontsize=8, loc='best')

    plt.suptitle("Cluster Visualizations (PCA 2D Projection)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "06_cluster_visualizations.png")


# 7. Silhouette Score Comparison

def plot_silhouette_comparison(cluster_results):
    """Bar chart comparing silhouette scores across clustering algorithms."""
    print("\n Plotting silhouette score comparison…")

    names  = [r["name"]       for r in cluster_results]
    scores = [r["silhouette"] if r["silhouette"] is not None else 0
              for r in cluster_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color=COLORS[:len(names)], edgecolor='white', width=0.5)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.005,
                f'{s:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(scores) * 1.2 + 0.05)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Algorithm Comparison (Silhouette Score)", fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.5)')
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_silhouette_comparison.png")


# Master: Run All Visualizations

def run_all_visualizations(df, clf_results, cluster_results, y_test, class_names):
    """Call all visualization functions in sequence."""
    print("\n" + "="*60)
    print(" GENERATING ALL VISUALIZATIONS")
    print("="*60)

    plot_label_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_boxplots(df)
    plot_confusion_matrices(clf_results, y_test, class_names)
    plot_accuracy_comparison(clf_results)
    plot_cluster_visualizations(cluster_results)
    plot_silhouette_comparison(cluster_results)

    print(f"\n All plots saved to '{PLOTS_DIR}/' folder.")
