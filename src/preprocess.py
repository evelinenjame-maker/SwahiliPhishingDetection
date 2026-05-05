"""
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

PHONE_PATTERN = re.compile(r"(?:\+?255|0)\d{8,9}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"(?:tsh|tzs|usd|sh)\.?\s*\d[\d,]*|\d{4,}(?:[,.]\d{3})*(?:/=|/-)?|\d+[,.]\d{3}(?:[,.]\d{3})*", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(r"[^\w\s<>]")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

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
        text_lower = re.sub(r"\b" + re.escape(variant) + r"\b",
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
