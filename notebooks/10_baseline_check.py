"""
Day 4 - Step 3: Quick baseline sanity check.

This is NOT the full evaluation. It is a smoke test to confirm:
  1. The feature matrices loaded correctly
  2. A simple Naive Bayes can learn from them
  3. Performance is in a sensible range (60-99% accuracy expected)

If this returns suspiciously perfect (100%) or suspiciously bad (50% =
random), there's a methodology bug to fix BEFORE running the full
evaluation tomorrow.

Run from project root:
    python notebooks\\10_baseline_check.py
"""

from pathlib import Path
import json
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

print("=" * 60)
print("BASELINE SANITY CHECK (not full evaluation)")
print("=" * 60)

# Load master index
with open(FEATURES_DIR / "all_configs.json", "r") as f:
    all_configs = json.load(f)

results = []

for cfg_info in all_configs:
    name = cfg_info["config"]
    cfg_dir = FEATURES_DIR / f"config_{name}"

    X_train = sparse.load_npz(cfg_dir / "X_train.npz")
    y_train = np.load(cfg_dir / "y_train.npy")
    X_val = sparse.load_npz(cfg_dir / "X_val.npz")
    y_val = np.load(cfg_dir / "y_val.npy")
    X_bongo = sparse.load_npz(cfg_dir / "X_bongoscam_test.npz")
    y_bongo = np.load(cfg_dir / "y_bongoscam_test.npy")

    # Multinomial NB requires non-negative inputs.
    # Our standardized engineered features have negatives, so for this
    # smoke test we use ComplementNB (handles negatives via abs internally
    # is wrong - actually we use min-max approach: shift).
    # Simplest: clip negatives to 0 for the smoke test only.
    X_train_pos = X_train.copy()
    X_train_pos.data = np.clip(X_train_pos.data, 0, None)
    X_val_pos = X_val.copy()
    X_val_pos.data = np.clip(X_val_pos.data, 0, None)
    X_bongo_pos = X_bongo.copy()
    X_bongo_pos.data = np.clip(X_bongo_pos.data, 0, None)

    clf = MultinomialNB()
    clf.fit(X_train_pos, y_train)

    val_pred = clf.predict(X_val_pos)
    bongo_pred = clf.predict(X_bongo_pos)

    val_acc = accuracy_score(y_val, val_pred)
    bongo_acc = accuracy_score(y_bongo, bongo_pred)

    results.append({
        "config": name,
        "n_features": X_train.shape[1],
        "val_acc": val_acc,
        "bongoscam_test_acc": bongo_acc,
    })

    print(f"\n  Config: {name}")
    print(f"    Features:           {X_train.shape[1]}")
    print(f"    Val accuracy:       {val_acc:.4f}")
    print(f"    BongoSCAM test acc: {bongo_acc:.4f}")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY (Multinomial NB smoke test)")
print("=" * 60)
print(f"{'Config':<20} {'Features':<10} {'Val acc':<10} {'BongoSCAM acc':<14}")
print("-" * 60)
for r in results:
    print(f"{r['config']:<20} {r['n_features']:<10} "
          f"{r['val_acc']:<10.4f} {r['bongoscam_test_acc']:<14.4f}")

# Sanity check
print("\n" + "=" * 60)
print("SANITY ASSESSMENT")
print("=" * 60)
all_acc = [r["val_acc"] for r in results]
if max(all_acc) >= 0.99 and min(all_acc) >= 0.99:
    print("[!] All configs at 99%+ accuracy. Possible data leakage or")
    print("    feature too dominant. Will investigate ablations tomorrow.")
elif min(all_acc) < 0.60:
    print("[!] Some configs near random. Check feature extraction.")
else:
    print("[OK] Accuracies in expected range (70-99%). Ready for full evaluation.")
print("=" * 60)
