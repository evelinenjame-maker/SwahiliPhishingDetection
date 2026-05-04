"""
Day 2 - Step 7: Fetch legitimate Swahili text from public sources.

Uses the HuggingFace Swahili News dataset (public, no auth needed).
Filters for short text similar in length to SMS messages, then samples
700 examples for the legitimate class.

Run from project root:
    python notebooks\03_fetch_legitimate_swahili.py

Output:
    data/processed/legitimate_swahili.csv

Note: This will pip-install 'datasets' on first run if needed.
"""

from pathlib import Path
import sys
import subprocess
import random
import pandas as pd

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Ensure datasets package is available
try:
    from datasets import load_dataset
except ImportError:
    print("[INFO] Installing 'datasets' package (one-time)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "datasets", "-q"])
    from datasets import load_dataset


def fetch_swahili_news(target_count=700, min_chars=20, max_chars=300):
    """Pull Swahili news, filter to SMS-length text."""
    print("[INFO] Loading 'community-datasets/swahili_news' from HuggingFace...")
    print("       (first run downloads ~30MB; subsequent runs use cache)")

    try:
        ds = load_dataset("community-datasets/swahili_news",
                          split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Primary dataset failed: {e}")
        print("[INFO] Trying alternative: 'flax-community/swahili-safi'...")
        ds = load_dataset("flax-community/swahili-safi", split="train")

    # Try to find the text column
    text_col = None
    for candidate in ["text", "content", "Sms", "headline", "title"]:
        if candidate in ds.column_names:
            text_col = candidate
            break
    if text_col is None:
        text_col = ds.column_names[0]
    print(f"[OK] Using column: '{text_col}'")
    print(f"[OK] Total available examples: {len(ds)}")

    # Slice into sentence-sized chunks
    candidates = []
    for i, row in enumerate(ds):
        text = str(row[text_col]).strip()
        if not text:
            continue
        # Split long articles into sentences for SMS-style examples
        for sentence in text.replace("\n", ". ").split(". "):
            s = sentence.strip()
            if min_chars <= len(s) <= max_chars and len(s.split()) >= 4:
                candidates.append(s)
        if len(candidates) >= target_count * 5:
            break

    print(f"[OK] Filtered candidates ({min_chars}-{max_chars} chars): "
          f"{len(candidates)}")

    # Shuffle and sample
    random.shuffle(candidates)
    selected = candidates[:target_count]

    return selected


def main():
    print("=" * 60)
    print("LEGITIMATE SWAHILI TEXT FETCHER")
    print("=" * 60)

    messages = fetch_swahili_news(target_count=700)

    df = pd.DataFrame({
        "message": messages,
        "label": "legitimate",
        "source": "swahili_news_hf",
    })
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    print(f"\nFinal count: {len(df)}")
    print(f"\nMessage length stats:")
    print(df["message"].str.len().describe().round(1).to_string())

    print(f"\n5 random samples:")
    for i, row in df.sample(5, random_state=1).iterrows():
        msg = row["message"][:200]
        print(f"  - {msg}")

    out_path = PROCESSED_DIR / "legitimate_swahili.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"[OK] Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
