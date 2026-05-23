"""
clustering_models.py
====================
Implements unsupervised clustering: K-Means, Hierarchical, DBSCAN.
Evaluates with Silhouette Score (higher = better separated clusters).
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Helper: reduce to 2D for visualization

def reduce_to_2d(X):
    """Use PCA to project features to 2D for cluster visualization."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA variance explained: {explained:.1f}%")
    return X_2d, pca



# Silhouette Score Helper

def compute_silhouette(X, labels, name):
    """
    Silhouette Score ranges from -1 to 1.
    > 0.5 = strong clusters, 0.25–0.5 = moderate, < 0.25 = weak.
    """
    unique = np.unique(labels)
    # Need at least 2 clusters (and DBSCAN may produce noise = -1)
    valid = labels[labels != -1]
    if len(np.unique(valid)) < 2:
        print(f"  ⚠️ {name}: Not enough valid clusters for silhouette score.")
        return None

    X_valid = X[labels != -1]
    labels_valid = labels[labels != -1]

    try:
        score = silhouette_score(X_valid, labels_valid, sample_size=2000, random_state=42)
        print(f"  🏅 Silhouette Score ({name}): {score:.4f}")
        return score
    except Exception as e:
        print(f"  ⚠️ Silhouette error for {name}: {e}")
        return None



# 1. K-Means Clustering

def kmeans_clustering(X, n_clusters=5):
    """
    K-Means: Partitions data into k clusters by minimizing within-cluster variance.
    We use k=5 matching the 5 gesture phases.
    """
    print(f"\n{'='*60}")
    print("🔵 K-Means Clustering")
    print('='*60)

    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',    # Smart initialization for faster convergence
        n_init=10,            # 10 random inits, pick best
        max_iter=300,
        random_state=42
    )
    labels = model.fit_predict(X)

    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Clusters found: {len(unique)}")
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c} samples")

    score = compute_silhouette(X, labels, "K-Means")

    return {
        "name":       "K-Means",
        "model":      model,
        "labels":     labels,
        "silhouette": score,
        "n_clusters": n_clusters,
    }



# 2. Hierarchical (Agglomerative) Clustering

def hierarchical_clustering(X, n_clusters=5):
    """
    Agglomerative Clustering: Bottom-up merging of closest clusters.
    Ward linkage minimizes the total within-cluster variance.
    """
    print(f"\n{'='*60}")
    print("🌳 Hierarchical (Agglomerative) Clustering")
    print('='*60)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'       # Ward: minimizes variance within clusters
    )
    labels = model.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Clusters found: {len(unique)}")
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c} samples")

    score = compute_silhouette(X, labels, "Hierarchical")

    return {
        "name":       "Hierarchical",
        "model":      model,
        "labels":     labels,
        "silhouette": score,
        "n_clusters": n_clusters,
    }


# ─────────────────────────────────────────────
# 3. DBSCAN Clustering
# ─────────────────────────────────────────────
def dbscan_clustering(X, eps=1.5, min_samples=10):
    """
    DBSCAN: Density-based clustering.
    - Doesn't need a pre-defined k.
    - Labels noise points as -1.
    - Good for irregular-shaped clusters.
    """
    print(f"\n{'='*60}")
    print("🔴 DBSCAN Clustering")
    print('='*60)

    model = DBSCAN(
        eps=eps,             # Max distance between two samples to be neighbors
        min_samples=min_samples,  # Min samples to form a dense region (core point)
        n_jobs=-1
    )
    labels = model.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len([u for u in unique if u != -1])
    n_noise    = counts[unique == -1][0] if -1 in unique else 0

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points  : {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    for u, c in zip(unique, counts):
        label_str = "Noise" if u == -1 else f"Cluster {u}"
        print(f"    {label_str}: {c} samples")

    score = compute_silhouette(X, labels, "DBSCAN")

    return {
        "name":       "DBSCAN",
        "model":      model,
        "labels":     labels,
        "silhouette": score,
        "n_clusters": n_clusters,
    }


# Run All Clustering Models

def run_all_clustering(X, n_classes=5):
    """
    Runs K-Means, Hierarchical, and DBSCAN clustering.
    Also returns 2D PCA projection for visualization.
    """
    print("\n" + "="*60)
    print("🚀 RUNNING ALL CLUSTERING MODELS")
    print("="*60)

    # Reduce to 2D once (reused in visualizations)
    print("\n📐 Reducing to 2D via PCA for visualization:")
    X_2d, pca = reduce_to_2d(X)

    results = []
    results.append(kmeans_clustering(X, n_clusters=n_classes))
    results.append(hierarchical_clustering(X, n_clusters=n_classes))
    results.append(dbscan_clustering(X, eps=1.5, min_samples=10))

    # Attach 2D projection to all results
    for r in results:
        r["X_2d"] = X_2d

    return results
