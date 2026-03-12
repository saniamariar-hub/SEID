"""
Save TF-IDF + Logistic Regression Model
=======================================
Trains and saves the baseline model for use in SEID Engine.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v2.csv"
OUTPUT_DIR = "./tfidf_model"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 30000

def main():
    print("=" * 70)
    print("SAVING TF-IDF + LOGISTIC REGRESSION MODEL")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["y"] = df["attack_label"].isin(["smishing", "phishing"]).astype(int)
    print(f"      Total: {len(df):,} | Malicious: {df['y'].sum():,}")

    X = df["clean_text"].values
    y = df["y"].values

    # Split
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Vectorizer
    print("\n[3/4] Training TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        lowercase=True, sublinear_tf=True,
        min_df=2, max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=RANDOM_STATE
    )
    model.fit(X_train_tfidf, y_train)

    # Save
    print("\n[4/4] Saving models...")
    joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(model, f"{OUTPUT_DIR}/logistic_regression.joblib")
    print(f"      Saved: {OUTPUT_DIR}/tfidf_vectorizer.joblib")
    print(f"      Saved: {OUTPUT_DIR}/logistic_regression.joblib")

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()

