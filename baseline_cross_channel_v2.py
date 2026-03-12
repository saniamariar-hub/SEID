"""
Cross-Channel Baseline Model V2
================================
Binary classifier for malicious (phishing + smishing) vs benign.
TF-IDF + Logistic Regression on cross-channel corpus.

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
    f1_score,
    roc_auc_score
)
# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v2.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 30000  # Conservative for memory

# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 70)
    print("CROSS-CHANNEL BASELINE MODEL V2")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    print(f"      Total rows: {len(df):,}")

    # Step 2: Create binary label (malicious = phishing OR smishing)
    print("\n[2/7] Creating binary labels...")
    df["y"] = df["attack_label"].isin(["smishing", "phishing"]).astype(int)

    print(f"      Class distribution:")
    print(f"        Benign (0):    {(df['y'] == 0).sum():,}")
    print(f"        Malicious (1): {(df['y'] == 1).sum():,}")
    print(f"      Attack breakdown:")
    print(f"        Phishing: {(df['attack_label'] == 'phishing').sum():,}")
    print(f"        Smishing: {(df['attack_label'] == 'smishing').sum():,}")

    X = df["clean_text"].values
    y = df["y"].values

    # Step 3: Stratified train/test split
    print("\n[3/7] Splitting data (stratified 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train size: {len(X_train):,} (malicious: {y_train.sum():,})")
    print(f"      Test size:  {len(X_test):,} (malicious: {y_test.sum():,})")

    # Step 4: Build TF-IDF features (word n-grams for stability)
    print("\n[4/7] Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        lowercase=True, sublinear_tf=True,
        min_df=2, max_df=0.95
    )
    X_train_combined = vectorizer.fit_transform(X_train)
    X_test_combined = vectorizer.transform(X_test)
    print(f"      Total features: {X_train_combined.shape[1]:,}")

    # Step 5: Train model
    print("\n[5/7] Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    model.fit(X_train_combined, y_train)
    print("      Training complete.")

    # Step 6: Predictions
    print("\n[6/7] Generating predictions...")
    y_pred = model.predict(X_test_combined)
    y_proba = model.predict_proba(X_test_combined)[:, 1]

    # Step 7: Evaluation
    print("\n[7/7] Evaluation Results")
    print("=" * 70)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Benign  Malicious")
    print(f"  Actual Benign   {tn:>7}    {fp:>7}")
    print(f"  Actual Malicious{fn:>7}    {tp:>7}")

    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nMalicious Class Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print(f"\nFalse Negatives (missed attacks): {fn}")
    print(f"False Positives (benign flagged): {fp}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    print("=" * 70)
    print("CROSS-CHANNEL BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

