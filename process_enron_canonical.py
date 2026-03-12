"""
Enron Email Dataset Preprocessing Pipeline
===========================================
Converts raw Enron emails into a canonical, model-ready dataset
for social engineering detection (benign baseline).

Author: Data Engineering / Security NLP Team
"""

import pandas as pd
import re
from email import policy
from email.parser import Parser
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_CSV = "emails.csv"
OUTPUT_CSV = "enron_canonical_ready.csv"
SAMPLE_SIZE = 50000
RANDOM_STATE = 42

# =========================
# HELPERS
# =========================

# Regex patterns for stripping residual headers from body text
HEADER_PATTERNS = [
    r"^mime-version:.*$",
    r"^content-type:.*$",
    r"^content-transfer-encoding:.*$",
    r"^x-.*:.*$",
    r"^message-id:.*$",
    r"^date:.*$",
    r"^from:.*$",
    r"^to:.*$",
    r"^subject:.*$",
    r"^cc:.*$",
    r"^bcc:.*$",
]
HEADER_REGEX = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE | re.MULTILINE)


def parse_email_message(raw_message):
    """
    Parse raw email string using RFC-aware email library.
    Returns dict with message_id, from, to, date, body or None on failure.
    """
    if not isinstance(raw_message, str) or not raw_message.strip():
        return None

    try:
        msg = Parser(policy=policy.default).parsestr(raw_message)

        # Extract plain-text body only
        body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_content()
                    except Exception:
                        body = None
                    break
        else:
            try:
                body = msg.get_content()
            except Exception:
                body = None

        # Ensure body is string
        if body is not None and not isinstance(body, str):
            body = str(body)

        return {
            "message_id": msg.get("Message-ID", None),
            "from": msg.get("From", None),
            "to": msg.get("To", None),
            "date": msg.get("Date", None),
            "body": body
        }
    except Exception:
        return None


def clean_text(text):
    """
    LIGHT cleaning only - security safe.
    - Strips residual email headers
    - Normalizes whitespace
    - Does NOT remove stopwords or punctuation
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Remove residual header lines
    text = HEADER_REGEX.sub("", text)

    # Normalize line breaks and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)      # Collapse spaces/tabs
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()

    return text if text else None


def parse_timestamp_safe(date_str):
    """
    Safely parse timestamp to UTC. Returns None on failure.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        return pd.to_datetime(date_str, utc=True, errors="coerce")
    except Exception:
        return None


def normalize_email_address(addr):
    """
    Normalize email address: lowercase, strip whitespace.
    """
    if not isinstance(addr, str):
        return ""
    return addr.lower().strip()


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 50)
    print("ENRON EMAIL PREPROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Load and sample dataset
    print(f"\n[1/5] Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    total_available = len(df)
    print(f"      Total emails available: {total_available:,}")

    print(f"\n[2/5] Sampling {SAMPLE_SIZE:,} emails (random_state={RANDOM_STATE})...")
    df = df.sample(n=min(SAMPLE_SIZE, total_available), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"      Sample size: {len(df):,}")

    # Step 2: Parse emails
    print("\n[3/5] Parsing email messages (RFC-aware)...")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
        parsed = parse_email_message(row.get("message", ""))

        if parsed is None or not parsed.get("message_id"):
            continue

        raw_text = parsed["body"]
        clean = clean_text(raw_text) if raw_text else None

        results.append({
            "message_id": parsed["message_id"],
            "raw_text": raw_text,
            "clean_text": clean,
            "channel": "email",
            "language": "en",
            "sender_id": normalize_email_address(parsed["from"]),
            "receiver_id": normalize_email_address(parsed["to"]),
            "sender_role": "unknown",
            "timestamp": parse_timestamp_safe(parsed["date"]),
            "attack_label": "benign",
            "manipulation_tactic": "none",
            "intent_stage": "benign",
            "source_dataset": "enron",
            "confidence_label": "high"
        })

    # Step 3: Create final DataFrame
    print("\n[4/5] Building canonical DataFrame...")
    final_columns = [
        "message_id", "raw_text", "clean_text", "channel", "language",
        "sender_id", "receiver_id", "sender_role", "timestamp",
        "attack_label", "manipulation_tactic", "intent_stage",
        "source_dataset", "confidence_label"
    ]
    canonical_df = pd.DataFrame(results, columns=final_columns)

    # Step 4: Health checks
    print("\n[5/5] Dataset Health Check")
    print("-" * 40)
    total_rows = len(canonical_df)
    non_empty_raw = canonical_df["raw_text"].notna().sum()
    non_empty_clean = canonical_df["clean_text"].notna().sum()
    missing_ts = canonical_df["timestamp"].isna().sum()

    print(f"Total rows:           {total_rows:,}")
    print(f"Non-empty raw_text:   {non_empty_raw:,} ({100*non_empty_raw/total_rows:.1f}%)")
    print(f"Non-empty clean_text: {non_empty_clean:,} ({100*non_empty_clean/total_rows:.1f}%)")
    print(f"Missing timestamps:   {missing_ts:,} ({100*missing_ts/total_rows:.1f}%)")

    # Step 5: Save
    print(f"\nSaving to {OUTPUT_CSV}...")
    canonical_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Output saved to: {OUTPUT_CSV}")

    # Sample inspection
    print("\n" + "=" * 50)
    print("SAMPLE INSPECTION (first valid row)")
    print("=" * 50)
    if len(canonical_df) > 0:
        sample = canonical_df.iloc[0]
        print(f"message_id: {sample['message_id']}")
        print(f"sender_id:  {sample['sender_id']}")
        print(f"timestamp:  {sample['timestamp']}")
        clean_preview = sample['clean_text'][:200] if sample['clean_text'] else "N/A"
        print(f"clean_text (first 200 chars):\n{clean_preview}")


if __name__ == "__main__":
    main()
