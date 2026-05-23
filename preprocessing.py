"""
preprocessing.py
================
Handles all data loading, cleaning, encoding, normalization, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



# 1. Load Dataset

def load_data(filepath="dataset.csv"):
    """Load the gesture phase dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully from '{filepath}'")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found.")
        raise


# 2. Display Basic Info

def display_basic_info(df):
    """Print head, shape, info, and missing value summary."""
    print("\n" + "="*60)
    print("📋 DATASET OVERVIEW")
    print("="*60)

    print("\n🔹 First 5 Rows:")
    print(df.head())

    print(f"\n🔹 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n🔹 Data Types & Non-Null Counts:")
    print(df.info())

    print("\n🔹 Missing Values Per Column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  → No missing values found.")

    print("\n🔹 Label (Phase) Distribution:")
    print(df['phase'].value_counts())
    print("="*60)



# 3. Handle Missing Values

def handle_missing_values(df):
    """
    Fill missing values:
    - Numeric columns → fill with column mean
    - Categorical columns → fill with mode
    """
    before = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    after = df.isnull().sum().sum()
    print(f"\n✅ Missing values handled: {before} → {after}")
    return df



# 4. Fix Label Typos

def clean_labels(df, label_col='phase'):
    """Fix known label typos in the dataset (e.g., Portuguese artifact)."""
    typo_map = {
        'Preparação': 'Preparation',
        'Prepara\u00e7\u00e3o': 'Preparation',
    }
    df[label_col] = df[label_col].replace(typo_map)
    print(f"\n✅ Labels cleaned. Unique phases: {sorted(df[label_col].unique())}")
    return df



# 5. Encode Labels

def encode_labels(df, label_col='phase'):
    """
    Encode string gesture phase labels to integers.
    Returns: df, label_encoder, encoded_series
    """
    le = LabelEncoder()
    df['phase_encoded'] = le.fit_transform(df[label_col])

    print("\n🔹 Label Encoding Map:")
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        print(f"  {idx} → {cls}")

    return df, le



# 6. Extract Features and Target

def extract_features_target(df, drop_cols=('phase', 'phase_encoded', 'timestamp')):
    """
    Separate feature matrix X and target vector y.
    Drops non-feature columns (labels, timestamps).
    """
    drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop)
    y = df['phase_encoded']
    print(f"\n✅ Features: {X.shape[1]} columns | Target: 'phase_encoded'")
    return X, y



# 7. Normalize Features

def normalize_features(X_train, X_test):
    """
    Standardize features using StandardScaler (fit on train, transform both).
    Z-score normalization: mean=0, std=1.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("\n✅ Features normalized using StandardScaler (Z-score).")
    return X_train_scaled, X_test_scaled, scaler



# 8. Train-Test Split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets (80% / 20%).
    Stratified to preserve class balance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n✅ Train-Test Split: {X_train.shape[0]} train | {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test



# 9. Full Preprocessing Pipeline

def run_preprocessing(filepath="dataset.csv"):
    """
    Master function: loads → cleans → encodes → splits → normalizes.
    Returns everything needed for modelling.
    """
    df = load_data(filepath)
    display_basic_info(df)
    df = handle_missing_values(df)
    df = clean_labels(df)
    df, label_encoder = encode_labels(df)
    X, y = extract_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = normalize_features(X_train, X_test)

    return {
        "df":            df,
        "X":             X,
        "y":             y,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "X_train_sc":    X_train_sc,
        "X_test_sc":     X_test_sc,
        "scaler":        scaler,
        "label_encoder": label_encoder,
    }
