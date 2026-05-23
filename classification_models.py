

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report
)



# Helper: train + evaluate one model

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, class_names):
    """
    Fit model, predict on test set, print metrics.
    Returns dict with name, model, predictions, and scores.
    """
    print(f"\n{'='*60}")
    print(f"🤖 Model: {name}")
    print('='*60)

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)

    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return {
        "name":      name,
        "model":     model,
        "y_pred":    y_pred,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
    }



# 1. Logistic Regression

def logistic_regression(X_train, X_test, y_train, y_test, class_names):
    """
    Logistic Regression: Linear classifier using sigmoid/softmax.
    Works well when classes are linearly separable.
    """
    model = LogisticRegression(
        max_iter=1000,       
        random_state=42,
    )
    return train_and_evaluate(
        "Logistic Regression", model,
        X_train, X_test, y_train, y_test, class_names
    )



# 2. Decision Tree

def decision_tree(X_train, X_test, y_train, y_test, class_names):
    """
    Decision Tree: Rule-based tree that splits on feature thresholds.
    Easy to interpret. Can overfit without depth limit.
    """
    model = DecisionTreeClassifier(
        max_depth=10,        # Limit depth to reduce overfitting
        random_state=42
    )
    return train_and_evaluate(
        "Decision Tree", model,
        X_train, X_test, y_train, y_test, class_names
    )



# 3. Random Forest

def random_forest(X_train, X_test, y_train, y_test, class_names):
    """
    Random Forest: Ensemble of Decision Trees using bagging.
    More robust, less prone to overfitting than a single tree.
    """
    model = RandomForestClassifier(
        n_estimators=100,    # 100 trees in the forest
        max_depth=None,      # Let trees grow fully
        random_state=42,
        n_jobs=-1            # Use all CPU cores
    )
    return train_and_evaluate(
        "Random Forest", model,
        X_train, X_test, y_train, y_test, class_names
    )



# 4. Support Vector Machine (SVM)

def support_vector_machine(X_train, X_test, y_train, y_test, class_names):
    """
    SVM: Finds optimal hyperplane to separate classes.
    RBF kernel handles non-linear boundaries.
    Works best with normalized features.
    """
    model = SVC(
        kernel='rbf',        # Radial Basis Function for non-linear separation
        C=1.0,               # Regularization parameter
        gamma='scale',       # Kernel coefficient
        random_state=42
    )
    return train_and_evaluate(
        "SVM (RBF Kernel)", model,
        X_train, X_test, y_train, y_test, class_names
    )



# 5. K-Nearest Neighbors (KNN)

def knn(X_train, X_test, y_train, y_test, class_names):
    """
    KNN: Classifies based on majority vote of k nearest neighbors.
    Simple but memory-intensive. Sensitive to feature scaling.
    """
    model = KNeighborsClassifier(
        n_neighbors=7,       # Consider 7 nearest neighbors
        metric='euclidean',
        n_jobs=-1
    )
    return train_and_evaluate(
        "K-Nearest Neighbors (KNN)", model,
        X_train, X_test, y_train, y_test, class_names
    )



# Run All Classification Models

def run_all_classifiers(X_train, X_test, y_train, y_test, class_names):
    """
    Sequentially trains all 5 models and returns list of result dicts.
    X inputs should be already normalized (StandardScaler).
    """
    print("\n" + "="*60)
    print("🚀 RUNNING ALL CLASSIFICATION MODELS")
    print("="*60)

    results = []

    # Run each model
    results.append(logistic_regression(X_train, X_test, y_train, y_test, class_names))
    results.append(decision_tree(X_train, X_test, y_train, y_test, class_names))
    results.append(random_forest(X_train, X_test, y_train, y_test, class_names))
    results.append(support_vector_machine(X_train, X_test, y_train, y_test, class_names))
    results.append(knn(X_train, X_test, y_train, y_test, class_names))

    return results
