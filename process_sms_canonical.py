"""
UCI SMS Spam Collection Preprocessing Pipeline
===============================================
Converts raw SMS spam dataset into canonical schema
matching Enron baseline for social engineering detection.

Author: Data Engineering / Security NLP Team
"""

import pandas as pd
import re
import uuid

# =========================
# CONFIG
# =========================
INPUT_FILE = "SMSSpamCollection"
OUTPUT_CSV = "sms_canonical_ready.csv"

# =========================
# KEYWORD DICTIONARIES FOR MANIPULATION TACTIC DETECTION
# =========================
TACTIC_KEYWORDS = {
    "urgency": ["urgent", "now", "immediately", "claim", "hurry", "fast", "quick", "asap", "expire"],
    "fear": ["alert", "suspended", "problem", "blocked", "warning", "security", "verify", "confirm"],
    "reward": ["free", "win", "prize", "cash", "winner", "congratulations", "award", "bonus", "gift"],
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
    - Normalizes whitespace and line breaks
    - Does NOT remove stopwords or punctuation
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Normalize line breaks and whitespace
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text if text else None


def detect_manipulation_tactic(text):
    """
    Rule-based keyword detection for manipulation tactics.
    Returns the first matching tactic or 'none'.
    """
    if not isinstance(text, str):
        return "none"

    text_lower = text.lower()

    for tactic, keywords in TACTIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return tactic

    return "none"


def parse_line(line):
    """
    Parse a single line from SMSSpamCollection.
    Format: <label>\t<message>
    Returns (label, message) or None on failure.
    """
    if not isinstance(line, str) or not line.strip():
        return None

    # Split on first tab only
    parts = line.strip().split("\t", 1)

    if len(parts) != 2:
        return None

    label = parts[0].strip().lower()
    message = parts[1].strip()

    if label not in ("ham", "spam"):
        return None

    if not message:
        return None

    return (label, message)


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 50)
    print("UCI SMS SPAM PREPROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Read raw file
    print(f"\n[1/4] Reading raw file: {INPUT_FILE}...")

    results = []
    skipped = 0

    try:
        with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                parsed = parse_line(line)

                if parsed is None:
                    skipped += 1
                    continue

                label, message = parsed

                # Map to canonical fields
                is_spam = (label == "spam")

                results.append({
                    "message_id": generate_message_id(),
                    "raw_text": message,
                    "clean_text": clean_text(message),
                    "channel": "sms",
                    "language": "en",
                    "sender_id": "unknown",
                    "receiver_id": "unknown",
                    "sender_role": "unknown",
                    "timestamp": None,
                    "attack_label": "smishing" if is_spam else "benign",
                    "manipulation_tactic": detect_manipulation_tactic(message) if is_spam else "none",
                    "intent_stage": "exploit" if is_spam else "benign",
                    "source_dataset": "uci_sms",
                    "confidence_label": "high"
                })

    except FileNotFoundError:
        print(f"ERROR: File '{INPUT_FILE}' not found.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return

    print(f"      Lines parsed: {len(results):,}")
    print(f"      Lines skipped: {skipped:,}")

    # Step 2: Create DataFrame
    print("\n[2/4] Building canonical DataFrame...")
    final_columns = [
        "message_id", "raw_text", "clean_text", "channel", "language",
        "sender_id", "receiver_id", "sender_role", "timestamp",
        "attack_label", "manipulation_tactic", "intent_stage",
        "source_dataset", "confidence_label"
    ]
    df = pd.DataFrame(results, columns=final_columns)

    # Step 3: Health checks
    print("\n[3/4] Dataset Health Check")
    print("-" * 40)
    total_rows = len(df)
    benign_count = (df["attack_label"] == "benign").sum()
    smishing_count = (df["attack_label"] == "smishing").sum()
    non_empty_clean = df["clean_text"].notna().sum()

    print(f"Total rows:           {total_rows:,}")
    print(f"Benign (ham):         {benign_count:,} ({100*benign_count/total_rows:.1f}%)")
    print(f"Smishing (spam):      {smishing_count:,} ({100*smishing_count/total_rows:.1f}%)")
    print(f"Non-empty clean_text: {non_empty_clean:,} ({100*non_empty_clean/total_rows:.1f}%)")

    # Tactic distribution for spam
    print("\nManipulation Tactic Distribution (spam only):")
    spam_df = df[df["attack_label"] == "smishing"]
    tactic_counts = spam_df["manipulation_tactic"].value_counts()
    for tactic, count in tactic_counts.items():
        print(f"  {tactic}: {count:,} ({100*count/len(spam_df):.1f}%)")

    # Step 4: Save
    print(f"\n[4/4] Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Output saved to: {OUTPUT_CSV}")

    # Sample inspection
    print("\n" + "=" * 50)
    print("SAMPLE INSPECTION")
    print("=" * 50)

    # Show one benign and one smishing
    benign_sample = df[df["attack_label"] == "benign"].iloc[0] if benign_count > 0 else None
    smishing_sample = df[df["attack_label"] == "smishing"].iloc[0] if smishing_count > 0 else None

    if benign_sample is not None:
        print("\n[BENIGN SAMPLE]")
        print(f"  message_id: {benign_sample['message_id']}")
        print(f"  attack_label: {benign_sample['attack_label']}")
        print(f"  clean_text: {benign_sample['clean_text'][:100]}...")

    if smishing_sample is not None:
        print("\n[SMISHING SAMPLE]")
        print(f"  message_id: {smishing_sample['message_id']}")
        print(f"  attack_label: {smishing_sample['attack_label']}")
        print(f"  manipulation_tactic: {smishing_sample['manipulation_tactic']}")
        print(f"  clean_text: {smishing_sample['clean_text'][:100]}...")


if __name__ == "__main__":
    main()

