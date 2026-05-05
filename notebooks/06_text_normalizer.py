"""
Day 3 - Step 2: Text normalization pipeline.

Implements the preprocessing steps specified in Proposal Section 3.3:
  - Cleaning: removing duplicates, irrelevant content, noise
  - Tokenization: splitting text into tokens
  - Normalization: lowercasing, punctuation, Swahili-specific characters
  - Feature Preparation: converting to structured format

Engineering decisions (documented in Chapter 4 methodology):
  1. Engineered features are extracted BEFORE lowercasing/normalization
     because case (uppercase ratio), digit count, etc. are phishing signals.
  2. Phone numbers, URLs, and money amounts are replaced with placeholder
     tokens (<PHONE>, <URL>, <MONEY>) rather than deleted, to preserve
     "presence-of-X" signal without overfitting on specific values.
  3. Mobile operators are normalized (m-pesa, m_pesa, mpesa -> mpesa) so
     the model sees them as one feature.
  4. English words (e.g., 'me', 'need') are preserved - they reflect
     authentic Tanzanian SMS register with English borrowings, consistent
     with the proposal scope of Swahili-language online content.

Output:
  data/processed/normalized_dataset.csv  - with engineered features and clean text
  src/preprocess.py                       - reusable module for later stages
"""

from pathlib import Path
import re
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SRC_DIR = PROJECT_ROOT / "src"
SRC_DIR.mkdir(exist_ok=True)


# =====================================================================
# REGEX PATTERNS - compiled once, reused
# =====================================================================
# Tanzanian phone number patterns:
#  - 0XXXXXXXXX (10 digits starting with 0)
#  - +255XXXXXXXXX
#  - 255XXXXXXXXX
PHONE_PATTERN = re.compile(
    r"(?:\+?255|0)\d{8,9}"
)

# URLs and links
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+",
    re.IGNORECASE,
)

# Money amounts: TSh 50000, Tsh.50000, 50,000/=, 1,000,000 etc.
MONEY_PATTERN = re.compile(
    r"(?:tsh|tzs|usd|sh)\.?\s*\d[\d,]*|"
    r"\d{4,}(?:[,.]\d{3})*(?:/=|/-)?|"
    r"\d+[,.]\d{3}(?:[,.]\d{3})*",
    re.IGNORECASE,
)

# Email addresses
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

# Multiple whitespace -> single space
WHITESPACE_PATTERN = re.compile(r"\s+")

# Punctuation (preserve some semantically meaningful chars)
PUNCT_PATTERN = re.compile(r"[^\w\s<>]")  # keep word chars, whitespace, < >

# Repeated characters (e.g., "haraaaaka" -> "haraaka")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


# Mobile operator normalization: order matters!
# We map ALL variants to canonical, but apply LONGEST patterns first
# across ALL operators (not within each), to avoid e.g. "voda" matching
# inside "vodacom".
OPERATOR_VARIANTS = {
    "mpesa": ["m-pesa", "m pesa", "m_pesa", "mpesa"],
    "tigopesa": ["tigo-pesa", "tigo pesa", "tigopesa", "tigo_pesa"],
    "airtelmoney": ["airtel-money", "airtel money", "airtelmoney"],
    "halopesa": ["halo-pesa", "halo pesa", "halopesa", "halotel pesa"],
    "vodacom": ["vodacom"],
    "airtel": ["airtel"],
    "tigo": ["tigo"],
    "halotel": ["halotel"],
    "ttcl": ["ttcl"],
}


def normalize_operators(text: str) -> str:
    """Standardize mobile operator name variants. Longest-first to avoid
    partial-overlap bugs (e.g. 'voda' inside 'vodacom')."""
    text_lower = text.lower()
    # Build flat list (canonical, variant) sorted by variant length desc
    pairs = [(canonical, variant)
             for canonical, variants in OPERATOR_VARIANTS.items()
             for variant in variants]
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    for canonical, variant in pairs:
        # Use word-boundary regex so 'tigo' doesn't match inside 'tigopesa'
        # (tigopesa already replaced first because it's longer).
        text_lower = re.sub(r"\b" + re.escape(variant) + r"\b",
                           canonical, text_lower)
    return text_lower


# =====================================================================
# ENGINEERED FEATURES - extracted from RAW text BEFORE normalization
# =====================================================================
URGENCY_WORDS = {
    "haraka", "sasa", "hivi", "leo", "mara", "muda", "kabla",
    "tafadhali", "saa", "hospitali", "ajali", "shida",
    "asap", "urgent", "now", "quickly",
}

MONEY_WORDS = {
    "pesa", "hela", "ela", "tsh", "tzs", "shilingi",
    "elfu", "laki", "milioni", "money",
}


def extract_features(raw_text: str) -> dict:
    """Extract numerical features from RAW (unnormalized) text."""
    text = str(raw_text)
    n_chars = len(text)
    if n_chars == 0:
        return {
            "char_count": 0, "word_count": 0,
            "uppercase_ratio": 0.0, "digit_ratio": 0.0,
            "phone_count": 0, "url_count": 0, "money_count": 0,
            "exclaim_count": 0, "question_count": 0,
            "urgency_word_count": 0, "money_word_count": 0,
            "all_caps_words": 0,
        }

    words = text.split()
    text_lower = text.lower()

    return {
        "char_count": n_chars,
        "word_count": len(words),
        "uppercase_ratio": sum(c.isupper() for c in text) / n_chars,
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "phone_count": len(PHONE_PATTERN.findall(text)),
        "url_count": len(URL_PATTERN.findall(text)),
        "money_count": len(MONEY_PATTERN.findall(text)),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "urgency_word_count": sum(1 for w in words if w.lower() in URGENCY_WORDS),
        "money_word_count": sum(1 for w in words if w.lower() in MONEY_WORDS),
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
    }


# =====================================================================
# NORMALIZATION PIPELINE
# =====================================================================
def normalize_text(raw_text: str) -> str:
    """Apply the full normalization pipeline.

    Steps:
      1. Replace URLs with <URL>
      2. Replace emails with <EMAIL>
      3. Replace phone numbers with <PHONE>
      4. Replace money amounts with <MONEY>
      5. Lowercase
      6. Normalize mobile operator variants
      7. Collapse repeated characters (haraaaaka -> haraaka)
      8. Strip remaining punctuation
      9. Collapse whitespace
    """
    text = str(raw_text)

    # Order matters: URLs first (may contain digits), then emails,
    # then phones (10-digit), then money. Phone before money so we
    # don't tag phone digits as money amounts.
    text = URL_PATTERN.sub(" <URL> ", text)
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = PHONE_PATTERN.sub(" <PHONE> ", text)
    text = MONEY_PATTERN.sub(" <MONEY> ", text)

    text = text.lower()
    text = normalize_operators(text)

    # Strip punctuation BEFORE collapsing repeats to preserve digit sequences
    # like "666" which are phishing signals (Freemason etc.)
    text = PUNCT_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# =====================================================================
# REUSABLE MODULE EXPORT
# =====================================================================
MODULE_CODE = '''"""
Reusable preprocessing module for the Swahili Phishing Detection project.
Imported by feature-extraction and model-training scripts.

Usage:
    from src.preprocess import normalize_text, extract_features, load_stopwords
    df["clean"] = df["message"].apply(normalize_text)
"""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOPWORD_PATH = PROJECT_ROOT / "data" / "processed" / "swahili_stopwords.txt"

PHONE_PATTERN = re.compile(r"(?:\\+?255|0)\\d{8,9}")
URL_PATTERN = re.compile(r"https?://\\S+|www\\.\\S+|bit\\.ly/\\S+|tinyurl\\.com/\\S+", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"(?:tsh|tzs|usd|sh)\\.?\\s*\\d[\\d,]*|\\d{4,}(?:[,.]\\d{3})*(?:/=|/-)?|\\d+[,.]\\d{3}(?:[,.]\\d{3})*", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
WHITESPACE_PATTERN = re.compile(r"\\s+")
PUNCT_PATTERN = re.compile(r"[^\\w\\s<>]")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\\1{2,}")

OPERATOR_VARIANTS = {
    "mpesa": ["m-pesa", "m pesa", "m_pesa", "mpesa"],
    "tigopesa": ["tigo-pesa", "tigo pesa", "tigopesa"],
    "airtelmoney": ["airtel-money", "airtel money", "airtelmoney"],
    "halopesa": ["halo-pesa", "halo pesa", "halopesa", "halotel pesa"],
    "vodacom": ["vodacom"],
    "airtel": ["airtel"], "tigo": ["tigo"], "halotel": ["halotel"], "ttcl": ["ttcl"],
}

URGENCY_WORDS = {"haraka", "sasa", "hivi", "leo", "mara", "muda", "kabla",
                 "tafadhali", "saa", "hospitali", "ajali", "shida",
                 "asap", "urgent", "now", "quickly"}
MONEY_WORDS = {"pesa", "hela", "ela", "tsh", "tzs", "shilingi",
               "elfu", "laki", "milioni", "money"}


def normalize_operators(text):
    text_lower = text.lower()
    pairs = [(canonical, variant)
             for canonical, variants in OPERATOR_VARIANTS.items()
             for variant in variants]
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    for canonical, variant in pairs:
        text_lower = re.sub(r"\\b" + re.escape(variant) + r"\\b",
                           canonical, text_lower)
    return text_lower


def normalize_text(raw_text):
    text = str(raw_text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = PHONE_PATTERN.sub(" <PHONE> ", text)
    text = MONEY_PATTERN.sub(" <MONEY> ", text)
    text = text.lower()
    text = normalize_operators(text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_features(raw_text):
    text = str(raw_text)
    n_chars = len(text)
    if n_chars == 0:
        return {k: 0 for k in ["char_count", "word_count", "uppercase_ratio",
                "digit_ratio", "phone_count", "url_count", "money_count",
                "exclaim_count", "question_count", "urgency_word_count",
                "money_word_count", "all_caps_words"]}
    words = text.split()
    return {
        "char_count": n_chars,
        "word_count": len(words),
        "uppercase_ratio": sum(c.isupper() for c in text) / n_chars,
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "phone_count": len(PHONE_PATTERN.findall(text)),
        "url_count": len(URL_PATTERN.findall(text)),
        "money_count": len(MONEY_PATTERN.findall(text)),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "urgency_word_count": sum(1 for w in words if w.lower() in URGENCY_WORDS),
        "money_word_count": sum(1 for w in words if w.lower() in MONEY_WORDS),
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
    }


def load_stopwords():
    if not STOPWORD_PATH.exists():
        return set()
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize(text, remove_stopwords=False):
    text = normalize_text(text) if not text.startswith(" ") else text
    tokens = text.split()
    if remove_stopwords:
        sw = load_stopwords()
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    return tokens
'''


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 60)
    print("TEXT NORMALIZATION PIPELINE")
    print("=" * 60)

    # Save reusable module
    module_path = SRC_DIR / "preprocess.py"
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(MODULE_CODE)
    # Create __init__.py so it's importable
    init_path = SRC_DIR / "__init__.py"
    init_path.touch()
    print(f"[OK] Saved reusable module: {module_path}")

    # Load combined dataset
    df = pd.read_csv(PROCESSED_DIR / "combined_dataset.csv")
    print(f"\n[OK] Loaded {len(df)} messages")

    print("\n[INFO] Extracting engineered features (from raw text)...")
    feats = df["message"].apply(extract_features).tolist()
    feats_df = pd.DataFrame(feats)
    print(f"[OK] Extracted {feats_df.shape[1]} numeric features")

    print("\n[INFO] Normalizing text...")
    df["clean_text"] = df["message"].apply(normalize_text)
    print("[OK] Normalization complete")

    # Combine
    out = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # Drop rows where normalization removed too much
    before = len(out)
    out = out[out["clean_text"].str.len() >= 3].reset_index(drop=True)
    if len(out) < before:
        print(f"[WARN] Dropped {before - len(out)} rows after normalization")

    # Show before/after samples
    print("\n" + "=" * 60)
    print("BEFORE / AFTER SAMPLES")
    print("=" * 60)
    for label in ["phishing", "legitimate"]:
        print(f"\n--- {label.upper()} ---")
        sample = out[out["label"] == label].sample(2, random_state=7)
        for _, row in sample.iterrows():
            print(f"\n  Original : {row['message'][:140]}")
            print(f"  Cleaned  : {row['clean_text'][:140]}")
            print(f"  Features : char={row['char_count']}, "
                  f"upper_ratio={row['uppercase_ratio']:.2f}, "
                  f"phones={row['phone_count']}, "
                  f"urgency={row['urgency_word_count']}")

    # Feature stats per class
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS BY CLASS")
    print("=" * 60)
    feature_cols = [c for c in feats_df.columns]
    stats = out.groupby("label")[feature_cols].mean().round(3).T
    print(stats.to_string())

    # Save
    out_path = PROCESSED_DIR / "normalized_dataset.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] Saved: {out_path}")
    print(f"     Columns: {list(out.columns)}")
    print(f"     Rows: {len(out)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
