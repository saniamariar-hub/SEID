"""
Phishing Email Dataset Preprocessing Pipeline
==============================================
Converts Nazario phishing emails into canonical schema
for cross-channel social engineering detection.

Author: Data Engineering / Security NLP Team
"""

import pandas as pd
import re
import uuid

# =========================
# CONFIG
# =========================
INPUT_FILE = "Nazario_5.csv"
OUTPUT_CSV = "phishing_email_canonical_ready.csv"
SOURCE_DATASET = "nazario_phishing"

# =========================
# KEYWORD DICTIONARIES FOR MANIPULATION TACTIC DETECTION
# =========================
TACTIC_KEYWORDS = {
    "urgency": [
        "urgent", "immediately", "now", "asap", "expire", "deadline", 
        "hurry", "limited time", "act now", "within 24", "within 48",
        "suspend", "suspended", "terminated", "closing"
    ],
    "authority": [
        "bank", "paypal", "ebay", "amazon", "irs", "government", 
        "official", "administrator", "support team", "security team",
        "account manager", "compliance", "legal", "federal"
    ],
    "fear": [
        "alert", "warning", "problem", "blocked", "unauthorized",
        "suspicious", "fraud", "stolen", "hacked", "compromised",
        "locked", "disabled", "restricted", "violation"
    ],
    "reward": [
        "free", "win", "winner", "prize", "cash", "reward", "bonus",
        "gift", "lottery", "inheritance", "million", "congratulations"
    ],
}


# =========================
# HELPERS
# =========================

def generate_message_id():
    """Generate a unique message ID."""
    return str(uuid.uuid4())


def clean_text(text):
    """
    LIGHT cleaning only - security safe.
    - Normalizes whitespace
    - Does NOT remove stopwords or punctuation
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Normalize line breaks and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()

    return text if text else None


def detect_manipulation_tactic(text):
    """
    Rule-based keyword detection for manipulation tactics.
    Returns the first matching tactic or 'unknown'.
    """
    if not isinstance(text, str):
        return "unknown"

    text_lower = text.lower()

    # Check in priority order
    for tactic in ["authority", "urgency", "fear", "reward"]:
        for keyword in TACTIC_KEYWORDS[tactic]:
            if keyword in text_lower:
                return tactic

    return "unknown"


def anonymize_email(email_str):
    """Anonymize email address for privacy."""
    if not isinstance(email_str, str) or not email_str.strip():
        return "unknown"
    # Extract just domain if possible, otherwise return unknown
    match = re.search(r"@([\w.-]+)", email_str)
    if match:
        return f"user@{match.group(1)}"
    return "unknown"


def parse_timestamp_safe(date_str):
    """Safely parse timestamp to UTC."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        return pd.to_datetime(date_str, utc=True, errors="coerce")
    except Exception:
        return None


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 60)
    print("PHISHING EMAIL PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load dataset
    print(f"\n[1/5] Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"      Total rows: {len(df):,}")
    print(f"      Label distribution: {df['label'].value_counts().to_dict()}")

    # Step 2: Filter phishing emails only (label=1)
    print("\n[2/5] Filtering phishing emails (label=1)...")
    phishing_df = df[df["label"] == 1].copy().reset_index(drop=True)
    print(f"      Phishing emails: {len(phishing_df):,}")

    # Step 3: Build canonical records
    print("\n[3/5] Building canonical records...")
    results = []

    for _, row in phishing_df.iterrows():
        raw_text = row.get("body", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raw_text = ""

        clean = clean_text(raw_text)
        # Ensure clean_text is never null - use raw_text as fallback
        if clean is None:
            clean = raw_text.strip() if raw_text else "[empty]"

        # Combine subject + body for tactic detection
        full_text = f"{row.get('subject', '')} {raw_text}"

        results.append({
            "message_id": generate_message_id(),
            "raw_text": raw_text,
            "clean_text": clean,
            "channel": "email",
            "language": "en",
            "sender_id": anonymize_email(row.get("sender", "")),
            "receiver_id": anonymize_email(row.get("receiver", "")),
            "sender_role": "unknown",
            "timestamp": parse_timestamp_safe(row.get("date", "")),
            "attack_label": "phishing",
            "manipulation_tactic": detect_manipulation_tactic(full_text),
            "intent_stage": "exploit",
            "source_dataset": SOURCE_DATASET,
            "confidence_label": "high"
        })

    # Step 4: Create DataFrame
    print("\n[4/5] Dataset Health Check")
    print("-" * 60)
    final_columns = [
        "message_id", "raw_text", "clean_text", "channel", "language",
        "sender_id", "receiver_id", "sender_role", "timestamp",
        "attack_label", "manipulation_tactic", "intent_stage",
        "source_dataset", "confidence_label"
    ]
    canonical_df = pd.DataFrame(results, columns=final_columns)

    total_rows = len(canonical_df)
    non_empty_clean = canonical_df["clean_text"].notna().sum()
    missing_ts = canonical_df["timestamp"].isna().sum()

    print(f"Total rows:           {total_rows:,}")
    print(f"Non-empty clean_text: {non_empty_clean:,} ({100*non_empty_clean/total_rows:.1f}%)")
    print(f"Missing timestamps:   {missing_ts:,} ({100*missing_ts/total_rows:.1f}%)")

    print("\nManipulation Tactic Distribution:")
    tactic_counts = canonical_df["manipulation_tactic"].value_counts()
    for tactic, count in tactic_counts.items():
        print(f"  {tactic}: {count:,} ({100*count/total_rows:.1f}%)")

    # Step 5: Save
    print(f"\n[5/5] Saving to {OUTPUT_CSV}...")
    canonical_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Output saved to: {OUTPUT_CSV}")

    # Sample inspection
    print("\n" + "=" * 60)
    print("SAMPLE INSPECTION")
    print("=" * 60)
    sample = canonical_df.iloc[0]
    print(f"message_id: {sample['message_id']}")
    print(f"attack_label: {sample['attack_label']}")
    print(f"manipulation_tactic: {sample['manipulation_tactic']}")
    clean_preview = sample['clean_text'][:200] if sample['clean_text'] else "N/A"
    print(f"clean_text (first 200 chars):\n{clean_preview}...")


if __name__ == "__main__":
    main()

