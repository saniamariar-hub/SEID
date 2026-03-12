"""
Baseline Model Error Analysis
==============================
Post-hoc analysis of TF-IDF + Logistic Regression smishing detector.
Identifies error patterns and inspects feature weights.

Author: Applied ML / Security NLP Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v1.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES_WORD = 20000

# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 70)
    print("BASELINE MODEL ERROR ANALYSIS")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["y"] = (df["attack_label"] == "smishing").astype(int)
    print(f"      Total rows: {len(df):,}")

    X = df["clean_text"].values
    y = df["y"].values

    # Step 2: Reproduce train/test split
    print("\n[2/6] Reproducing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Get indices for error analysis
    df_test = df.iloc[len(X_train):].reset_index(drop=True)
    
    # Actually need to track original indices
    indices = np.arange(len(df))
    _, test_indices, _, _ = train_test_split(
        indices, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    df_test = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"      Test size: {len(X_test):,}")

    # Step 3: Rebuild TF-IDF vectorizer
    print("\n[3/6] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=MAX_FEATURES_WORD,
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"      Features: {X_train_tfidf.shape[1]:,}")

    # Step 4: Rebuild model
    print("\n[4/6] Training model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    model.fit(X_train_tfidf, y_train)

    # Get predictions
    y_pred = model.predict(X_test_tfidf)

    # Step 5: Error analysis
    print("\n[5/6] Error Analysis")
    print("=" * 70)

    # False negatives: actual=1, predicted=0
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    fn_samples = df_test.iloc[fn_indices]

    print(f"\nFALSE NEGATIVES (Missed Smishing): {len(fn_samples)}")
    print("-" * 70)
    for i, (_, row) in enumerate(fn_samples.head(5).iterrows()):
        text = row["clean_text"][:200] + "..." if len(row["clean_text"]) > 200 else row["clean_text"]
        print(f"\n[FN-{i+1}] {text}")

    # False positives: actual=0, predicted=1
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    fp_samples = df_test.iloc[fp_indices]

    print(f"\n\nFALSE POSITIVES (Benign flagged as Smishing): {len(fp_samples)}")
    print("-" * 70)
    for i, (_, row) in enumerate(fp_samples.head(5).iterrows()):
        text = row["clean_text"][:200] + "..." if len(row["clean_text"]) > 200 else row["clean_text"]
        channel = row["channel"]
        print(f"\n[FP-{i+1}] [{channel}] {text}")

    # Step 6: Feature inspection
    print("\n\n[6/6] Feature Inspection")
    print("=" * 70)

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Sort by coefficient value
    sorted_indices = np.argsort(coefficients)

    # Top 20 smishing indicators (highest positive weights)
    print("\nTOP 20 SMISHING INDICATORS (positive weights):")
    print("-" * 50)
    top_smishing = sorted_indices[-20:][::-1]
    for i, idx in enumerate(top_smishing):
        print(f"  {i+1:2}. {feature_names[idx]:<30} {coefficients[idx]:+.4f}")

    # Top 20 benign indicators (most negative weights)
    print("\nTOP 20 BENIGN INDICATORS (negative weights):")
    print("-" * 50)
    top_benign = sorted_indices[:20]
    for i, idx in enumerate(top_benign):
        print(f"  {i+1:2}. {feature_names[idx]:<30} {coefficients[idx]:+.4f}")

    # N-gram analysis
    print("\n\nN-GRAM BREAKDOWN:")
    print("-" * 50)
    unigrams_smishing = [f for f in feature_names[top_smishing] if " " not in f]
    bigrams_smishing = [f for f in feature_names[top_smishing] if " " in f]
    print(f"  Smishing unigrams: {len(unigrams_smishing)}")
    print(f"  Smishing bigrams:  {len(bigrams_smishing)}")

    unigrams_benign = [f for f in feature_names[top_benign] if " " not in f]
    bigrams_benign = [f for f in feature_names[top_benign] if " " in f]
    print(f"  Benign unigrams:   {len(unigrams_benign)}")
    print(f"  Benign bigrams:    {len(bigrams_benign)}")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

