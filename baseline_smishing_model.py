"""
Baseline Smishing Detection Model
==================================
TF-IDF + Logistic Regression baseline for signal validation.
Uses word and character n-grams on imbalanced data.

Author: Applied ML / Security NLP Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v1.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES_WORD = 20000
MAX_FEATURES_CHAR = 20000

# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 60)
    print("BASELINE SMISHING DETECTION MODEL")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"      Total rows: {len(df):,}")

    # Step 2: Create binary label
    print("\n[2/6] Creating binary labels...")
    df["y"] = (df["attack_label"] == "smishing").astype(int)

    # Handle missing clean_text
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    print(f"      Rows after dropping nulls: {len(df):,}")

    X = df["clean_text"].values
    y = df["y"].values

    print(f"      Class distribution:")
    print(f"        Benign (0): {(y == 0).sum():,}")
    print(f"        Smishing (1): {(y == 1).sum():,}")

    # Step 3: Stratified train/test split
    print("\n[3/6] Splitting data (stratified 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"      Train size: {len(X_train):,}")
    print(f"      Test size: {len(X_test):,}")
    print(f"      Train smishing: {y_train.sum():,}")
    print(f"      Test smishing: {y_test.sum():,}")

    # Step 4: Build TF-IDF features (word n-grams only for baseline)
    print("\n[4/6] Building TF-IDF features...")

    # Word n-grams (1,2) - simpler baseline
    print("      Fitting word n-gram vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=MAX_FEATURES_WORD,
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    X_train_combined = vectorizer.fit_transform(X_train)
    X_test_combined = vectorizer.transform(X_test)
    print(f"      Total features: {X_train_combined.shape[1]:,}")

    # Step 5: Train Logistic Regression
    print("\n[5/6] Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    model.fit(X_train_combined, y_train)
    print("      Training complete.")

    # Step 6: Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    y_pred = model.predict(X_test_combined)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Benign  Smishing")
    print(f"  Actual Benign   {tn:>6}    {fp:>6}")
    print(f"  Actual Smishing {fn:>6}    {tp:>6}")

    # Metrics for smishing class
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\nSmishing Class Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\nFalse Negatives (missed smishing): {fn}")
    print(f"False Positives (benign flagged):  {fp}")

    # Full classification report
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Smishing"]))

    print("=" * 60)
    print("BASELINE MODEL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

