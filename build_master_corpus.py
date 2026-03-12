"""
Master Corpus Builder
=====================
Merges canonical datasets into a single model-ready corpus
with strict schema validation for social engineering detection.

Author: Data Engineering / Security NLP Team
"""

import pandas as pd
import sys

# =========================
# CONFIG
# =========================
INPUT_FILES = [
    "enron_canonical_ready.csv",
    "sms_canonical_ready.csv"
]
OUTPUT_FILE = "master_corpus_v1.csv"

# Required fields that must NOT be null
REQUIRED_FIELDS = ["clean_text", "attack_label", "channel"]

# Expected canonical column order
CANONICAL_COLUMNS = [
    "message_id", "raw_text", "clean_text", "channel", "language",
    "sender_id", "receiver_id", "sender_role", "timestamp",
    "attack_label", "manipulation_tactic", "intent_stage",
    "source_dataset", "confidence_label"
]


# =========================
# VALIDATION FUNCTIONS
# =========================

def validate_schema(df, filename):
    """
    Validate that DataFrame matches canonical schema exactly.
    Fails loudly on mismatch.
    """
    errors = []

    # Check column names match
    df_cols = list(df.columns)
    if df_cols != CANONICAL_COLUMNS:
        missing = set(CANONICAL_COLUMNS) - set(df_cols)
        extra = set(df_cols) - set(CANONICAL_COLUMNS)
        if missing:
            errors.append(f"Missing columns: {missing}")
        if extra:
            errors.append(f"Unexpected columns: {extra}")
        if df_cols != CANONICAL_COLUMNS and not missing and not extra:
            errors.append(f"Column order mismatch. Expected: {CANONICAL_COLUMNS}, Got: {df_cols}")

    if errors:
        print(f"\n[SCHEMA VALIDATION FAILED] {filename}")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def validate_required_fields(df, filename):
    """
    Validate that required fields have no unexpected nulls.
    """
    errors = []

    for field in REQUIRED_FIELDS:
        null_count = df[field].isna().sum()
        if null_count > 0:
            errors.append(f"Field '{field}' has {null_count:,} null values")

    if errors:
        print(f"\n[REQUIRED FIELD VALIDATION FAILED] {filename}")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 60)
    print("MASTER CORPUS BUILDER")
    print("=" * 60)

    # Step 1: Load datasets
    print("\n[1/5] Loading datasets...")
    dataframes = []

    for filepath in INPUT_FILES:
        try:
            df = pd.read_csv(filepath)
            print(f"      Loaded {filepath}: {len(df):,} rows")
            dataframes.append((filepath, df))
        except FileNotFoundError:
            print(f"\n[FATAL ERROR] File not found: {filepath}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[FATAL ERROR] Failed to load {filepath}: {e}")
            sys.exit(1)

    # Step 2: Validate schemas
    print("\n[2/5] Validating schemas...")
    all_valid = True

    for filepath, df in dataframes:
        schema_ok = validate_schema(df, filepath)
        fields_ok = validate_required_fields(df, filepath)
        if schema_ok and fields_ok:
            print(f"      ✓ {filepath} - schema valid")
        else:
            all_valid = False

    if not all_valid:
        print("\n[FATAL ERROR] Schema validation failed. Aborting merge.")
        sys.exit(1)

    # Step 3: Concatenate
    print("\n[3/5] Merging datasets...")
    master_df = pd.concat([df for _, df in dataframes], ignore_index=True)
    print(f"      Merged corpus size: {len(master_df):,} rows")

    # Step 4: Health checks
    print("\n[4/5] Dataset Health Checks")
    print("-" * 60)

    # Total row count
    print(f"\nTotal rows: {len(master_df):,}")

    # Distribution of attack_label
    print("\nAttack Label Distribution:")
    attack_dist = master_df["attack_label"].value_counts()
    for label, count in attack_dist.items():
        print(f"  {label}: {count:,} ({100*count/len(master_df):.2f}%)")

    # Distribution of channel
    print("\nChannel Distribution:")
    channel_dist = master_df["channel"].value_counts()
    for channel, count in channel_dist.items():
        print(f"  {channel}: {count:,} ({100*count/len(master_df):.2f}%)")

    # Cross-tab of channel × attack_label
    print("\nCross-tab (Channel × Attack Label):")
    crosstab = pd.crosstab(master_df["channel"], master_df["attack_label"], margins=True)
    print(crosstab.to_string())

    # Step 5: Save
    print(f"\n[5/5] Saving to {OUTPUT_FILE}...")
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Master corpus saved to: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE - READY FOR MODELING")
    print("=" * 60)


if __name__ == "__main__":
    main()

