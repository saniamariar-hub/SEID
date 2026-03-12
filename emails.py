import pandas as pd
import re

df = pd.read_csv("emails.csv", names=["file_path", "raw_message"])

def extract(pattern, text):
    m = re.search(pattern, str(text))
    return m.group(1) if m else None

canonical = pd.DataFrame({
    "file_path": df["file_path"],
    "message_id": df["raw_message"].apply(lambda x: extract(r"Message-ID:\s*<([^>]+)>", x)),
    "from": df["raw_message"].apply(lambda x: extract(r"From:\s*(.*)", x)),
    "to": df["raw_message"].apply(lambda x: extract(r"To:\s*(.*)", x)),
    "date": df["raw_message"].apply(lambda x: extract(r"Date:\s*(.*)", x)),
    "subject": df["raw_message"].apply(lambda x: extract(r"Subject:\s*(.*)", x)),
})

canonical.to_csv("canonical_emails.csv", index=False)
