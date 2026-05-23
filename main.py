"""
main.py
=======
Entry point for the Gesture Phase Segmentation ML project.
Orchestrates: Preprocessing → EDA → Classification → Clustering → Evaluation → Visualization
"""

import warnings
warnings.filterwarnings("ignore")   # Suppress sklearn convergence warnings for clean output

import sys
import os
import time

# Ensure console output does not break on CP1252 when emoji characters are used.
try:
    sys.stdout.reconfigure(errors='backslashreplace')
    sys.stderr.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

# ── Local module imports ──────────────────────
from preprocessing import run_preprocessing
from classification_models import run_all_classifiers
from clustering_models import run_all_clustering
from evaluation import (
    classification_comparison_table,
    clustering_comparison_table,
    print_final_conclusion,
)
from visualization import run_all_visualizations



# MAIN PIPELINE

def main():
    start_time = time.time()

    print("""

      GESTURE PHASE SEGMENTATION - ML PROJECT         
    Phases: Preparation, Stroke, Hold,             
           Retraction, Rest                        

    """)

    # STEP 1: Preprocessing 
    print("STEP 1: DATA PREPROCESSING")
    print("-"*60)
    data = run_preprocessing("dataset.csv")

    df           = data["df"]
    X_train_sc   = data["X_train_sc"]
    X_test_sc    = data["X_test_sc"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    label_encoder = data["label_encoder"]

    class_names = list(label_encoder.classes_)
    print(f"\n  Gesture classes: {class_names}")

    # ── STEP 2: EDA Visualizations (non-blocking) ─
    print("\n" + "─"*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("-"*60)
    print("  (EDA plots will be generated with all other plots at the end)")

    # ── STEP 3: Classification 
    print("\n" + "─"*60)
    print("STEP 3: CLASSIFICATION MODELS")
    print("-"*60)
    clf_results = run_all_classifiers(
        X_train_sc, X_test_sc, y_train, y_test, class_names
    )

    # ── STEP 4: Clustering 
    print("\n" + "─"*60)
    print("STEP 4: CLUSTERING MODELS")
    print("-"*60)
    # Use scaled features for clustering as well
    cluster_results = run_all_clustering(X_train_sc, n_classes=len(class_names))

    # ── STEP 5: Evaluation 
    print("\n" + "─"*60)
    print("STEP 5: EVALUATION & COMPARISON")
    print("-"*60)
    classification_comparison_table(clf_results)
    clustering_comparison_table(cluster_results)
    print_final_conclusion(clf_results, cluster_results, class_names)

    # ── STEP 6: Visualizations 
    print("\n" + "─"*60)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("-"*60)
    run_all_visualizations(df, clf_results, cluster_results, y_test, class_names)

    elapsed = time.time() - start_time
    print(f"\n✅ Project completed in {elapsed:.1f} seconds.")
    print("📁 Check the 'plots/' folder for all saved visualizations.\n")


if __name__ == "__main__":
    main()
