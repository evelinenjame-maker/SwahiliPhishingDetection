"""
Day 4 - Step 2: Feature extraction.

Builds 5 feature configurations as specified in Proposal Section 3.4:

  1. word_keep      - Word TF-IDF (unigrams + bigrams) + 12 engineered features
  2. word_remove    - Same as #1, but with Swahili stopwords removed
  3. char           - Character n-grams (3-5) + 12 engineered features
  4. combined_keep  - Word TF-IDF + char n-grams + engineered (no SW removal)
  5. combined_remove- Word TF-IDF + char n-grams + engineered (SW removed)

Critical methodology: ALL TF-IDF and char-ngram vectorizers are fitted
on the TRAINING SET ONLY. They are then used to transform val/test/
bongoscam_test. This avoids data leakage.

Outputs (saved to data/processed/features/):
  config_<name>/
    X_train.npz, y_train.npy
    X_val.npz,   y_val.npy
    X_test.npz,  y_test.npy
    X_bongoscam_test.npz, y_bongoscam_test.npy
    feature_info.json  (vocab size, dimensions, config description)

Run from project root:
    python notebooks\\09_feature_extraction.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# =====================================================================
# Load splits
# =====================================================================
print("=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)

train = pd.read_csv(SPLITS_DIR / "train.csv")
val = pd.read_csv(SPLITS_DIR / "val.csv")
test = pd.read_csv(SPLITS_DIR / "test.csv")
bongoscam_test = pd.read_csv(SPLITS_DIR / "bongoscam_test.csv")

splits = {"train": train, "val": val, "test": test,
          "bongoscam_test": bongoscam_test}

for name, split in splits.items():
    print(f"  {name:18s}: {len(split):>5d} rows")

# Load Swahili stopwords
with open(PROCESSED_DIR / "swahili_stopwords.txt", "r", encoding="utf-8") as f:
    SWAHILI_STOPWORDS = [w.strip() for w in f if w.strip()]
print(f"\n[OK] Loaded {len(SWAHILI_STOPWORDS)} Swahili stopwords")

# Engineered feature columns
ENGINEERED_COLS = [
    "char_count", "word_count", "uppercase_ratio", "digit_ratio",
    "phone_count", "url_count", "money_count", "exclaim_count",
    "question_count", "urgency_word_count", "money_word_count",
    "all_caps_words",
]

# Make sure clean_text is string
for split in splits.values():
    split["clean_text"] = split["clean_text"].fillna("").astype(str)


# =====================================================================
# Helper: build one feature configuration
# =====================================================================
def build_config(config_name, *, use_word_tfidf, use_char_ngrams,
                 remove_stopwords):
    """Build and save one feature configuration."""
    print("\n" + "-" * 60)
    print(f"Building config: '{config_name}'")
    print(f"  word_tfidf={use_word_tfidf}, char_ngrams={use_char_ngrams}, "
          f"remove_stopwords={remove_stopwords}")
    print("-" * 60)

    config_dir = FEATURES_DIR / f"config_{config_name}"
    config_dir.mkdir(exist_ok=True)

    info = {
        "config": config_name,
        "use_word_tfidf": use_word_tfidf,
        "use_char_ngrams": use_char_ngrams,
        "remove_stopwords": remove_stopwords,
        "components": {},
    }

    matrices = {name: [] for name in splits}

    # --- 1. Word TF-IDF ---
    if use_word_tfidf:
        sw = SWAHILI_STOPWORDS if remove_stopwords else None
        word_vec = TfidfVectorizer(
            ngram_range=(1, 2),    # unigrams + bigrams
            min_df=2,              # word must appear in >=2 documents
            max_df=0.95,           # ignore words in >95% of docs
            stop_words=sw,
            sublinear_tf=True,     # log-scaled tf
            lowercase=False,       # already lowercased in preprocessing
        )
        word_train = word_vec.fit_transform(train["clean_text"])
        print(f"  [OK] Word TF-IDF: vocab size = {len(word_vec.vocabulary_)}, "
              f"shape = {word_train.shape}")
        info["components"]["word_tfidf"] = {
            "vocab_size": len(word_vec.vocabulary_),
            "ngram_range": [1, 2],
            "stopwords_removed": remove_stopwords,
        }
        matrices["train"].append(word_train)
        matrices["val"].append(word_vec.transform(val["clean_text"]))
        matrices["test"].append(word_vec.transform(test["clean_text"]))
        matrices["bongoscam_test"].append(
            word_vec.transform(bongoscam_test["clean_text"])
        )

    # --- 2. Character n-grams ---
    if use_char_ngrams:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",   # word-boundary aware
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=False,
        )
        char_train = char_vec.fit_transform(train["clean_text"])
        print(f"  [OK] Char n-grams: vocab size = {len(char_vec.vocabulary_)}, "
              f"shape = {char_train.shape}")
        info["components"]["char_ngrams"] = {
            "vocab_size": len(char_vec.vocabulary_),
            "ngram_range": [3, 5],
        }
        matrices["train"].append(char_train)
        matrices["val"].append(char_vec.transform(val["clean_text"]))
        matrices["test"].append(char_vec.transform(test["clean_text"]))
        matrices["bongoscam_test"].append(
            char_vec.transform(bongoscam_test["clean_text"])
        )

    # --- 3. Engineered features (always included) ---
    scaler = StandardScaler()
    eng_train_raw = train[ENGINEERED_COLS].fillna(0).values
    eng_train = scaler.fit_transform(eng_train_raw)
    eng_val = scaler.transform(val[ENGINEERED_COLS].fillna(0).values)
    eng_test = scaler.transform(test[ENGINEERED_COLS].fillna(0).values)
    eng_bongo = scaler.transform(
        bongoscam_test[ENGINEERED_COLS].fillna(0).values
    )
    print(f"  [OK] Engineered features: {eng_train.shape}")
    info["components"]["engineered"] = {
        "feature_count": len(ENGINEERED_COLS),
        "feature_names": ENGINEERED_COLS,
        "standardized": True,
    }
    # Convert to sparse before stacking with TF-IDF
    matrices["train"].append(sparse.csr_matrix(eng_train))
    matrices["val"].append(sparse.csr_matrix(eng_val))
    matrices["test"].append(sparse.csr_matrix(eng_test))
    matrices["bongoscam_test"].append(sparse.csr_matrix(eng_bongo))

    # --- 4. Stack all components horizontally ---
    final = {}
    for name in splits:
        final[name] = sparse.hstack(matrices[name]).tocsr()

    print(f"\n  Final shapes:")
    for name, X in final.items():
        print(f"    {name:18s}: {X.shape}")
    info["final_shape_train"] = list(final["train"].shape)

    # --- 5. Save matrices and labels ---
    for name in splits:
        sparse.save_npz(config_dir / f"X_{name}.npz", final[name])
        y = (splits[name]["label"] == "phishing").astype(int).values
        np.save(config_dir / f"y_{name}.npy", y)

    # Save info
    with open(config_dir / "feature_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  [OK] Saved to {config_dir.relative_to(PROJECT_ROOT)}/")

    return info


# =====================================================================
# Build all 5 configurations
# =====================================================================
configs = [
    ("word_keep",        dict(use_word_tfidf=True,  use_char_ngrams=False, remove_stopwords=False)),
    ("word_remove",      dict(use_word_tfidf=True,  use_char_ngrams=False, remove_stopwords=True)),
    ("char",             dict(use_word_tfidf=False, use_char_ngrams=True,  remove_stopwords=False)),
    ("combined_keep",    dict(use_word_tfidf=True,  use_char_ngrams=True,  remove_stopwords=False)),
    ("combined_remove",  dict(use_word_tfidf=True,  use_char_ngrams=True,  remove_stopwords=True)),
]

all_info = []
for name, kwargs in configs:
    info = build_config(name, **kwargs)
    all_info.append(info)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY OF ALL CONFIGURATIONS")
print("=" * 60)
print(f"{'Config':<20} {'Word vocab':<12} {'Char vocab':<12} {'Final dims':<12}")
print("-" * 60)
for info in all_info:
    word = info["components"].get("word_tfidf", {}).get("vocab_size", "—")
    char = info["components"].get("char_ngrams", {}).get("vocab_size", "—")
    final = info["final_shape_train"][1]
    print(f"{info['config']:<20} {str(word):<12} {str(char):<12} {final:<12}")

# Save master index
with open(FEATURES_DIR / "all_configs.json", "w") as f:
    json.dump(all_info, f, indent=2)
print(f"\n[OK] Master index saved: {FEATURES_DIR / 'all_configs.json'}")
print("\n" + "=" * 60)
print("Next step: run notebooks/10_baseline_check.py")
print("=" * 60)
