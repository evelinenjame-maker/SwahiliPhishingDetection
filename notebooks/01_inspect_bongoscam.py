"""
Day 2 - Step 5: Inspect the BongoSCAM dataset.

Run from project root:
    python notebooks\01_inspect_bongoscam.py
"""

from pathlib import Path
import sys
import pandas as pd

# 1. Locate the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bongoscam"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists():
    print(f"[ERROR] Folder not found: {RAW_DIR}")
    sys.exit(1)

csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    print(f"[ERROR] No CSV files found inside {RAW_DIR}")
    sys.exit(1)

csv_path = csv_files[0]
print(f"[OK] Found CSV: {csv_path.name}")
print(f"     Size: {csv_path.stat().st_size / 1024:.1f} KB\n")

# 2. Load and inspect
df = pd.read_csv(csv_path)
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}\n")

print("First 3 rows:")
print(df.head(3).to_string())
print()

# 3. Auto-detect text and label columns (handles legacy 'object' and modern 'str' dtypes)
text_col = None
label_col = None
for col in df.columns:
    avg_len = df[col].astype(str).str.len().mean()
    if avg_len > 30:
        text_col = col
    elif df[col].nunique() <= 5:
        label_col = col

if text_col is None or label_col is None:
    print("[WARN] Could not auto-detect columns. Available columns:")
    for col in df.columns:
        print(f"  {col}: dtype={df[col].dtype}, "
              f"unique={df[col].nunique()}, "
              f"avg_len={df[col].astype(str).str.len().mean():.1f}")
    sys.exit(1)

print(f"[OK] Detected text column: '{text_col}'")
print(f"[OK] Detected label column: '{label_col}'\n")

# 4. Label distribution
print("=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)
print(df[label_col].value_counts())
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate messages: {df[text_col].duplicated().sum()}\n")

# 5. Length stats
df["msg_len"] = df[text_col].astype(str).str.len()
df["word_count"] = df[text_col].astype(str).str.split().str.len()
print("=" * 60)
print("MESSAGE LENGTH STATISTICS")
print("=" * 60)
print(df.groupby(label_col)[["msg_len", "word_count"]].describe().round(1))
print()

# 6. Sample messages per class
print("=" * 60)
print("SAMPLE MESSAGES PER CLASS")
print("=" * 60)
for label in df[label_col].unique():
    print(f"\n--- {str(label).upper()} ---")
    samples = df[df[label_col] == label][text_col].sample(
        min(3, (df[label_col] == label).sum()), random_state=42
    )
    for i, msg in enumerate(samples, 1):
        display = msg[:200] + "..." if len(msg) > 200 else msg
        print(f"  [{i}] {display}")

# 7. Save standardised clean version
clean = df[[text_col, label_col]].copy()
clean.columns = ["message", "label"]
label_map = {}
for v in clean["label"].unique():
    s = str(v).lower()
    if s in ("scam", "phishing", "spam", "fraud", "1", "true"):
        label_map[v] = "phishing"
    elif s in ("trust", "legit", "legitimate", "ham", "0", "false"):
        label_map[v] = "legitimate"
    else:
        label_map[v] = s
clean["label"] = clean["label"].map(label_map)
clean["source"] = "bongoscam"

clean = clean.dropna(subset=["message"])
clean = clean[clean["message"].str.strip() != ""]
clean = clean.drop_duplicates(subset=["message"])

out_path = PROCESSED_DIR / "bongoscam_clean.csv"
clean.to_csv(out_path, index=False, encoding="utf-8")
print("\n" + "=" * 60)
print(f"[OK] Saved cleaned dataset: {out_path}")
print(f"     {len(clean)} unique messages")
print(f"     Label counts: {clean['label'].value_counts().to_dict()}")
print("=" * 60)