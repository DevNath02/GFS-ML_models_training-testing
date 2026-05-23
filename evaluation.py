"""
evaluation.py
=============
Generates comparison tables for classification and clustering results,
identifies best models, and prints a final conclusion summary.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Classification Comparison Table
# ─────────────────────────────────────────────
def classification_comparison_table(results):
    """
    Build and print a comparison table of all classifier metrics.
    Returns a DataFrame.
    """
    print("\n" + "="*60)
    print("📊 CLASSIFICATION MODEL COMPARISON TABLE")
    print("="*60)

    rows = []
    for r in results:
        rows.append({
            "Model":     r["name"],
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df.index += 1  # Start rank from 1

    print(df.to_string())
    return df


# ─────────────────────────────────────────────
# 2. Best Classification Model
# ─────────────────────────────────────────────
def find_best_classifier(results):
    """Identify and announce the best performing classifier by accuracy."""
    best = max(results, key=lambda r: r["accuracy"])

    print(f"\n🏆 BEST CLASSIFICATION MODEL: {best['name']}")
    print(f"   Accuracy  : {best['accuracy']*100:.2f}%")
    print(f"   Precision : {best['precision']:.4f}")
    print(f"   Recall    : {best['recall']:.4f}")
    return best


# ─────────────────────────────────────────────
# 3. Clustering Comparison Table
# ─────────────────────────────────────────────
def clustering_comparison_table(cluster_results):
    """
    Build and print a comparison table of clustering results.
    """
    print("\n" + "="*60)
    print("🔵 CLUSTERING MODEL COMPARISON TABLE")
    print("="*60)

    rows = []
    for r in cluster_results:
        rows.append({
            "Algorithm":       r["name"],
            "Clusters Found":  r["n_clusters"],
            "Silhouette Score": round(r["silhouette"], 4) if r["silhouette"] is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 4. Best Clustering Algorithm
# ─────────────────────────────────────────────
def find_best_clustering(cluster_results):
    """Identify best clustering algorithm by silhouette score."""
    valid = [r for r in cluster_results if r["silhouette"] is not None]
    if not valid:
        print("⚠️ No valid silhouette scores available.")
        return None

    best = max(valid, key=lambda r: r["silhouette"])
    print(f"\n🏆 BEST CLUSTERING ALGORITHM: {best['name']}")
    print(f"   Silhouette Score: {best['silhouette']:.4f}")
    print(f"   Clusters Found  : {best['n_clusters']}")
    return best


# ─────────────────────────────────────────────
# 5. Final Conclusion
# ─────────────────────────────────────────────
def print_final_conclusion(clf_results, cluster_results, class_names):
    """Print a human-readable final conclusion summarizing all findings."""
    best_clf     = find_best_classifier(clf_results)
    best_cluster = find_best_clustering(cluster_results)

    print("\n" + "="*60)
    print("🎯 FINAL CONCLUSION")
    print("="*60)

    print(f"""
Project : Gesture Phase Segmentation
Classes : {', '.join(class_names)}
Dataset : Combined raw CSV files from UCI repository

─── Classification ───────────────────────────────────
• {len(clf_results)} models were trained and evaluated.
• Best model   : {best_clf['name']}
  - Accuracy   : {best_clf['accuracy']*100:.2f}%
  - Precision  : {best_clf['precision']:.4f}
  - Recall     : {best_clf['recall']:.4f}

• All model accuracies:""")

    for r in sorted(clf_results, key=lambda x: x["accuracy"], reverse=True):
        bar = "█" * int(r["accuracy"] * 20)
        print(f"    {r['name']:<30} {r['accuracy']*100:6.2f}%  {bar}")

    print(f"""
─── Clustering ───────────────────────────────────────
• 3 clustering algorithms were tested.
• Best algorithm : {best_cluster['name'] if best_cluster else 'N/A'}
  - Silhouette   : {f"{best_cluster['silhouette']:.4f}" if best_cluster else "N/A"}

• Silhouette scores (closer to 1.0 = better):""")

    for r in cluster_results:
        score_str = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
        print(f"    {r['name']:<25} {score_str}")

    print(f"""
─── Summary ──────────────────────────────────────────
  Random Forest and SVM typically excel at gesture
  phase segmentation due to their ability to capture
  non-linear boundaries in high-dimensional space.

  K-Means produces the most interpretable clusters
  because k=5 matches the number of gesture phases.

  Recommendation: Deploy {best_clf['name']} for real-time
  gesture phase classification.
""")
    print("="*60)
