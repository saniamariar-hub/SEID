"""
Threshold Analysis for Cross-Channel Model
===========================================
Evaluates model behavior under different decision thresholds
for security-critical risk tolerance tuning.

Author: Applied ML / Security NLP Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v2.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 30000

# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 70)
    print("THRESHOLD ANALYSIS FOR CROSS-CHANNEL MODEL")
    print("=" * 70)

    # Step 1: Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["y"] = df["attack_label"].isin(["smishing", "phishing"]).astype(int)

    X = df["clean_text"].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Test size: {len(X_test):,} (malicious: {y_test.sum():,})")

    # Step 2: Build model
    print("\n[2/5] Building TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        lowercase=True, sublinear_tf=True,
        min_df=2, max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=RANDOM_STATE
    )
    model.fit(X_train_tfidf, y_train)
    print("      Model trained.")

    # Step 3: Get probabilities
    print("\n[3/5] Generating predictions...")
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"      ROC-AUC: {roc_auc:.4f}")

    # Step 4: Threshold sweep
    print("\n[4/5] Sweeping thresholds...")
    thresholds = np.arange(0.0, 1.01, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fn": fn,
            "fp": fp,
            "tp": tp,
            "tn": tn
        })

    results_df = pd.DataFrame(results)

    # Step 5: Find optimal thresholds
    print("\n[5/5] Optimal Thresholds")
    print("=" * 70)

    # Threshold for ≥99% recall
    high_recall = results_df[results_df["recall"] >= 0.99]
    if len(high_recall) > 0:
        best_99 = high_recall.loc[high_recall["precision"].idxmax()]
        print(f"\n≥99% Recall Threshold:")
        print(f"  Threshold: {best_99['threshold']:.2f}")
        print(f"  Precision: {best_99['precision']:.4f}")
        print(f"  Recall:    {best_99['recall']:.4f}")
        print(f"  F1:        {best_99['f1']:.4f}")
        print(f"  FN:        {int(best_99['fn'])}")
        print(f"  FP:        {int(best_99['fp'])}")

    # Threshold that minimizes FN
    min_fn = results_df.loc[results_df["fn"].idxmin()]
    print(f"\nMinimum False Negatives:")
    print(f"  Threshold: {min_fn['threshold']:.2f}")
    print(f"  Precision: {min_fn['precision']:.4f}")
    print(f"  Recall:    {min_fn['recall']:.4f}")
    print(f"  FN:        {int(min_fn['fn'])}")
    print(f"  FP:        {int(min_fn['fp'])}")

    # Threshold that maximizes F1
    max_f1 = results_df.loc[results_df["f1"].idxmax()]
    print(f"\nMaximum F1-Score:")
    print(f"  Threshold: {max_f1['threshold']:.2f}")
    print(f"  Precision: {max_f1['precision']:.4f}")
    print(f"  Recall:    {max_f1['recall']:.4f}")
    print(f"  F1:        {max_f1['f1']:.4f}")
    print(f"  FN:        {int(max_f1['fn'])}")
    print(f"  FP:        {int(max_f1['fp'])}")

    # Default threshold comparison
    default = results_df[results_df["threshold"] == 0.50].iloc[0]
    print(f"\nDefault Threshold (0.50):")
    print(f"  Precision: {default['precision']:.4f}")
    print(f"  Recall:    {default['recall']:.4f}")
    print(f"  F1:        {default['f1']:.4f}")
    print(f"  FN:        {int(default['fn'])}")
    print(f"  FP:        {int(default['fp'])}")

    # Print threshold table
    print("\n" + "=" * 70)
    print("THRESHOLD TABLE (selected values)")
    print("=" * 70)
    print(f"{'Thresh':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FN':<8} {'FP':<8}")
    print("-" * 56)
    for t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        # Use approximate matching for floating point comparison
        row = results_df[np.isclose(results_df["threshold"], t, atol=0.001)]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"{t:<8.2f} {row['precision']:<8.4f} {row['recall']:<8.4f} "
                  f"{row['f1']:<8.4f} {int(row['fn']):<8} {int(row['fp']):<8}")

    # Plot
    print("\n[Generating plots...]")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Precision-Recall curve
        axes[0].plot(results_df["recall"], results_df["precision"], "b-", linewidth=2)
        axes[0].set_xlabel("Recall", fontsize=12)
        axes[0].set_ylabel("Precision", fontsize=12)
        axes[0].set_title("Precision vs Recall Curve", fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

        # Threshold vs Recall/Precision
        axes[1].plot(results_df["threshold"], results_df["recall"], "g-", label="Recall", linewidth=2)
        axes[1].plot(results_df["threshold"], results_df["precision"], "r-", label="Precision", linewidth=2)
        axes[1].plot(results_df["threshold"], results_df["f1"], "b--", label="F1", linewidth=2)
        axes[1].axvline(x=0.5, color="gray", linestyle=":", label="Default (0.5)")
        axes[1].set_xlabel("Threshold", fontsize=12)
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_title("Metrics vs Threshold", fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("threshold_analysis_plot.png", dpi=150)
        print("      Saved: threshold_analysis_plot.png")
    except Exception as e:
        print(f"      Plot failed: {e}")

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

