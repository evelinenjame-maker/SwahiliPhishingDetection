"""
Day 2 - Step 8: Combine BongoSCAM + synthetic phishing + legitimate news
into a single unified training dataset.

Run from project root:
    python notebooks\04_combine_dataset.py

Output:
    data/processed/combined_dataset.csv
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCES = [
    "bongoscam_clean.csv",
    "synthetic_phishing.csv",
    "legitimate_swahili.csv",
]

print("=" * 60)
print("COMBINING DATASETS")
print("=" * 60)

dfs = []
for filename in SOURCES:
    path = PROCESSED_DIR / filename
    if not path.exists():
        print(f"[WARN] Missing: {filename} - skipping")
        continue
    df = pd.read_csv(path)
    print(f"[OK] Loaded {filename}: {len(df)} rows, "
          f"labels = {df['label'].value_counts().to_dict()}")
    dfs.append(df)

if not dfs:
    print("[ERROR] No source files found. Run scripts 01, 02, 03 first.")
    raise SystemExit(1)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.dropna(subset=["message"])
combined = combined[combined["message"].str.strip() != ""]
before_dedup = len(combined)
combined = combined.drop_duplicates(subset=["message"]).reset_index(drop=True)

print("\n" + "=" * 60)
print("FINAL COMBINED DATASET")
print("=" * 60)
print(f"Total messages: {len(combined)} ({before_dedup - len(combined)} duplicates removed)")
print(f"\nLabel distribution:")
print(combined["label"].value_counts().to_string())
print(f"\nSource distribution:")
print(combined["source"].value_counts().to_string())

# Class balance
counts = combined["label"].value_counts()
ratio = counts.max() / counts.min()
print(f"\nClass imbalance ratio: {ratio:.2f}:1 "
      f"({'balanced enough' if ratio < 2 else 'consider rebalancing'})")

# Length stats
combined["msg_len"] = combined["message"].str.len()
print(f"\nMessage length by class:")
print(combined.groupby("label")["msg_len"].describe().round(1).to_string())

# Save
out_path = PROCESSED_DIR / "combined_dataset.csv"
combined.drop(columns=["msg_len"]).to_csv(out_path, index=False, encoding="utf-8")
print(f"\n[OK] Saved: {out_path}")
print("=" * 60)
