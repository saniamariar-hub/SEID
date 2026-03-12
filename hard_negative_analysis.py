"""
Hard Negative Mining Analysis
=============================
Identifies benign samples that resemble malicious linguistic patterns.
Useful for robustness hardening and false positive analysis.

Author: Applied ML / Security NLP Team
"""

import pandas as pd
import re
import random

# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v2.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 20

# Pattern definitions
URL_PATTERN = re.compile(r'(https?://|www\.)', re.IGNORECASE)
PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}')

AUTHORITY_KEYWORDS = [
    "bank", "account", "verify", "verification", "payment", "invoice",
    "paypal", "security", "confirm", "password", "update your", "login",
    "suspend", "billing", "transaction", "credit card", "debit"
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "deadline", "action required", "act now",
    "expires", "limited time", "asap", "right away", "final notice",
    "last chance", "respond now", "within 24", "within 48"
]

# =========================
# DETECTION FUNCTIONS
# =========================

def has_url(text):
    """Check if text contains URL patterns."""
    if not isinstance(text, str):
        return False
    return bool(URL_PATTERN.search(text))

def has_phone(text):
    """Check if text contains phone number patterns."""
    if not isinstance(text, str):
        return False
    return bool(PHONE_PATTERN.search(text))

def has_authority_language(text):
    """Check if text contains authority/financial keywords."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in AUTHORITY_KEYWORDS)

def has_urgency_language(text):
    """Check if text contains urgency keywords."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in URGENCY_KEYWORDS)

# =========================
# MAIN PIPELINE
# =========================

def main():
    random.seed(RANDOM_STATE)
    
    print("=" * 70)
    print("HARD NEGATIVE MINING ANALYSIS")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    print(f"      Total rows: {len(df):,}")

    # Step 2: Filter benign only
    print("\n[2/4] Filtering benign samples...")
    benign_df = df[df["attack_label"] == "benign"].copy()
    print(f"      Benign samples: {len(benign_df):,}")

    # Step 3: Detect high-risk patterns
    print("\n[3/4] Detecting high-risk linguistic patterns...")
    
    benign_df["has_url"] = benign_df["clean_text"].apply(has_url)
    benign_df["has_phone"] = benign_df["clean_text"].apply(has_phone)
    benign_df["has_authority"] = benign_df["clean_text"].apply(has_authority_language)
    benign_df["has_urgency"] = benign_df["clean_text"].apply(has_urgency_language)
    
    # Count each category
    url_count = benign_df["has_url"].sum()
    phone_count = benign_df["has_phone"].sum()
    authority_count = benign_df["has_authority"].sum()
    urgency_count = benign_df["has_urgency"].sum()
    
    # Any high-risk pattern
    benign_df["any_suspicious"] = (
        benign_df["has_url"] | benign_df["has_phone"] | 
        benign_df["has_authority"] | benign_df["has_urgency"]
    )
    suspicious_count = benign_df["any_suspicious"].sum()
    
    # Multiple patterns (more suspicious)
    benign_df["pattern_count"] = (
        benign_df["has_url"].astype(int) + 
        benign_df["has_phone"].astype(int) +
        benign_df["has_authority"].astype(int) + 
        benign_df["has_urgency"].astype(int)
    )
    multi_pattern = (benign_df["pattern_count"] >= 2).sum()

    # Step 4: Report
    print("\n[4/4] Hard Negative Statistics")
    print("=" * 70)
    
    total_benign = len(benign_df)
    print(f"\nTotal Benign Samples: {total_benign:,}")
    print(f"\nPattern Detection:")
    print(f"  URLs (http/www):      {url_count:,} ({100*url_count/total_benign:.2f}%)")
    print(f"  Phone Numbers:        {phone_count:,} ({100*phone_count/total_benign:.2f}%)")
    print(f"  Authority Language:   {authority_count:,} ({100*authority_count/total_benign:.2f}%)")
    print(f"  Urgency Language:     {urgency_count:,} ({100*urgency_count/total_benign:.2f}%)")
    print(f"\nAggregated:")
    print(f"  Any Suspicious Pattern:     {suspicious_count:,} ({100*suspicious_count/total_benign:.2f}%)")
    print(f"  Multiple Patterns (≥2):     {multi_pattern:,} ({100*multi_pattern/total_benign:.2f}%)")

    # Print examples for each category
    categories = [
        ("URL", "has_url"),
        ("Phone Number", "has_phone"),
        ("Authority Language", "has_authority"),
        ("Urgency Language", "has_urgency")
    ]
    
    for cat_name, col_name in categories:
        subset = benign_df[benign_df[col_name]]
        print(f"\n{'='*70}")
        print(f"CATEGORY: {cat_name} ({len(subset):,} samples)")
        print("=" * 70)
        
        if len(subset) == 0:
            print("  No samples found.")
            continue
            
        sample_indices = random.sample(range(len(subset)), min(SAMPLE_SIZE, len(subset)))
        for i, idx in enumerate(sample_indices, 1):
            row = subset.iloc[idx]
            text = row["clean_text"][:300].replace("\n", " ").strip()
            channel = row["channel"]
            print(f"\n[{i}] Channel: {channel}")
            print(f"    {text}...")

    # High-risk: multiple patterns
    print(f"\n{'='*70}")
    print(f"HIGH-RISK: Multiple Patterns (≥2) - {multi_pattern:,} samples")
    print("=" * 70)
    
    multi_df = benign_df[benign_df["pattern_count"] >= 2]
    if len(multi_df) > 0:
        sample_indices = random.sample(range(len(multi_df)), min(SAMPLE_SIZE, len(multi_df)))
        for i, idx in enumerate(sample_indices, 1):
            row = multi_df.iloc[idx]
            text = row["clean_text"][:300].replace("\n", " ").strip()
            patterns = []
            if row["has_url"]: patterns.append("URL")
            if row["has_phone"]: patterns.append("Phone")
            if row["has_authority"]: patterns.append("Authority")
            if row["has_urgency"]: patterns.append("Urgency")
            print(f"\n[{i}] Patterns: {', '.join(patterns)} | Channel: {row['channel']}")
            print(f"    {text}...")

    print("\n" + "=" * 70)
    print("HARD NEGATIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

