"""
Day 4 - Step 1: Create stratified train/validation/test splits.

This script MUST run before feature extraction (Script 09) so that TF-IDF
is fitted only on training data, avoiding data leakage.

Splits:
  - main_train (60%) - for fitting TF-IDF and training models
  - main_val (20%)   - for hyperparameter tuning and model selection
  - main_test (20%)  - held out for final reporting
  - bongoscam_test  - additional held-out test using BongoSCAM-only
                       messages from main_test, for honest evaluation
                       reporting on real-world data only

All splits are stratified on the 'label' column to preserve the 1.01:1
class balance. Random seed = 42 throughout for reproducibility.

Run from project root:
    python notebooks\\08_create_splits.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42

print("=" * 60)
print("CREATE STRATIFIED TRAIN/VAL/TEST SPLITS")
print("=" * 60)

# Load normalized dataset
df = pd.read_csv(PROCESSED_DIR / "normalized_dataset.csv")
print(f"[OK] Loaded {len(df)} messages from normalized_dataset.csv")

# Drop rows where clean_text is missing (just in case)
before = len(df)
df = df.dropna(subset=["clean_text", "label"]).reset_index(drop=True)
if len(df) < before:
    print(f"[WARN] Dropped {before - len(df)} rows with missing text/label")

# Show label distribution before split
print(f"\n[INFO] Label distribution:")
print(df["label"].value_counts().to_string())
print(f"\n[INFO] Source distribution:")
print(df["source"].value_counts().to_string())

# Add row_id for traceability
df["row_id"] = range(len(df))

# 60/20/20 stratified split
# First: separate test (20%)
train_val, test = train_test_split(
    df, test_size=0.20, stratify=df["label"],
    random_state=RANDOM_STATE,
)
# Then: from remaining 80%, take 25% as val (=20% of original)
train, val = train_test_split(
    train_val, test_size=0.25, stratify=train_val["label"],
    random_state=RANDOM_STATE,
)

print("\n" + "=" * 60)
print("SPLIT RESULTS")
print("=" * 60)
print(f"  Train: {len(train):>5d} ({len(train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val):>5d} ({len(val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test):>5d} ({len(test)/len(df)*100:.1f}%)")
print(f"  Total: {len(df):>5d}")

# Verify class balance preserved
print(f"\n[INFO] Phishing fraction in each split:")
for name, split in [("train", train), ("val", val), ("test", test)]:
    frac = (split["label"] == "phishing").mean()
    print(f"  {name:6s}: {frac*100:.2f}%")

# Verify source balance per split
print(f"\n[INFO] Source distribution in each split:")
for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"\n  {name}:")
    print(split["source"].value_counts().to_string())

# Carve out a BongoSCAM-only subset of the test set for honest evaluation
bongoscam_test = test[test["source"] == "bongoscam"].copy()
print("\n" + "=" * 60)
print(f"[INFO] BongoSCAM-only test subset: {len(bongoscam_test)} messages")
print(f"  Phishing fraction: {(bongoscam_test['label']=='phishing').mean()*100:.2f}%")
print("=" * 60)

# Save splits to disk
splits_dir = PROCESSED_DIR / "splits"
splits_dir.mkdir(exist_ok=True)

for name, split in [("train", train), ("val", val), ("test", test),
                    ("bongoscam_test", bongoscam_test)]:
    out_path = splits_dir / f"{name}.csv"
    split.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {out_path.relative_to(PROJECT_ROOT)} ({len(split)} rows)")

print("\n" + "=" * 60)
print("Next step: run notebooks/09_feature_extraction.py")
print("=" * 60)
