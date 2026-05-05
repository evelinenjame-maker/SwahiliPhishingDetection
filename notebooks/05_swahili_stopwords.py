"""
Day 3 - Step 1: Build a Swahili stopwords list.

Hybrid approach combining:
  - Published academic Swahili stopwords (based on Mlowe & Lyimo, 2020,
    Data in Brief; supplemented with high-frequency function words)
  - Corpus-derived high-frequency words from our combined dataset

Output:
  data/processed/swahili_stopwords.txt  - one word per line
  data/processed/phishing_indicator_words.txt - protected discriminative words
"""

from pathlib import Path
from collections import Counter
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Published / canonical Swahili stopwords
PUBLISHED_STOPWORDS = {
    # Pronouns
    "mimi", "wewe", "yeye", "sisi", "nyinyi", "ninyi", "wao",
    "yangu", "yako", "yake", "yetu", "yenu", "yao",
    "wangu", "wako", "wake", "wetu", "wenu",
    "langu", "lako", "lake", "letu", "lenu", "lao",
    "changu", "chako", "chake", "chetu", "chenu", "chao",
    "vyangu", "vyako", "vyake", "vyetu", "vyenu", "vyao",
    "huyu", "huyo", "yule", "hawa", "hao", "wale",
    "hii", "hiyo", "ile", "hizi", "hizo", "zile",
    "hili", "hilo", "lile", "haya", "hayo", "yale",
    "kile", "vile", "hivyo",
    # Question words
    "nani", "nini", "wapi", "lini", "vipi", "gani", "kwanini",
    "ngapi", "kiasi", "namna",
    # Articles, prepositions, conjunctions
    "wa", "ya", "za", "la", "cha", "vya", "kwa", "katika", "kwenye",
    "na", "au", "ama", "ila", "lakini", "bali", "wala",
    "kama", "kwamba", "kuliko", "ili", "japo", "ijapo",
    "tangu", "hadi", "mpaka", "baada", "kabla",
    # Demonstratives & locatives
    "huko", "hapa", "pale", "kule", "ndani", "nje", "juu", "chini",
    "mbele", "nyuma", "kati", "pamoja",
    # Common verbs
    "ni", "si", "kuna", "hakuna", "kuwa", "kuwepo",
    "yupo", "nipo", "upo", "tupo", "mpo",
    "ana", "nina", "una", "tuna", "mna", "wana",
    "alikuwa", "nilikuwa", "ulikuwa", "tulikuwa", "mlikuwa", "walikuwa",
    "atakuwa", "nitakuwa", "utakuwa", "tutakuwa", "mtakuwa", "watakuwa",
    # Negations
    "hapana", "siyo", "sio", "haupo", "haipo",
    # Temporal / quantitative
    "leo", "jana", "kesho", "sasa", "zamani", "baadaye",
    "mara", "siku", "wiki", "mwezi", "mwaka",
    "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi",
    "kwanza", "pili", "mwisho",
    "wote", "yote", "zote", "kila", "baadhi",
    # Politeness / discourse
    "tafadhali", "asante", "karibu", "samahani", "pole",
    "habari", "salaam", "salama",
    # Adverbs / intensifiers
    "sana", "tu", "pia", "tena", "bado", "kweli", "hasa",
    "labda", "pengine", "kabisa", "mno",
    # Chat fillers
    "ok", "okay", "sawa", "vizuri", "nzuri",
    # Single letters
    "a", "e", "i", "o", "u", "n", "m", "p", "h",
}

print(f"[OK] Loaded {len(PUBLISHED_STOPWORDS)} canonical Swahili stopwords")


def basic_tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    return [t for t in text.split() if len(t) > 1]


print("\n[INFO] Loading combined dataset...")
df = pd.read_csv(PROCESSED_DIR / "combined_dataset.csv")
print(f"[OK] {len(df)} messages loaded")

print("\n[INFO] Counting word frequencies per class...")
phishing_counter = Counter()
legitimate_counter = Counter()

for _, row in df.iterrows():
    tokens = basic_tokenize(row["message"])
    if row["label"] == "phishing":
        phishing_counter.update(tokens)
    else:
        legitimate_counter.update(tokens)

total_phishing = sum(phishing_counter.values())
total_legit = sum(legitimate_counter.values())

THRESHOLD = 0.005
balanced_stopwords = set()
for word, p_count in phishing_counter.items():
    if word not in legitimate_counter:
        continue
    p_freq = p_count / total_phishing
    l_freq = legitimate_counter[word] / total_legit
    if p_freq >= THRESHOLD and l_freq >= THRESHOLD:
        ratio = max(p_freq, l_freq) / min(p_freq, l_freq)
        if ratio < 1.5:
            balanced_stopwords.add(word)

print(f"[OK] Found {len(balanced_stopwords)} corpus-balanced high-frequency words")

print("\n" + "=" * 60)
print("REVIEW: Corpus-derived candidates not in published list")
print("=" * 60)
new_candidates = balanced_stopwords - PUBLISHED_STOPWORDS
for word in sorted(new_candidates):
    p_freq = phishing_counter[word] / total_phishing * 100
    l_freq = legitimate_counter[word] / total_legit * 100
    print(f"  {word:20s}  phishing={p_freq:.2f}%  legitimate={l_freq:.2f}%")

print("\n" + "=" * 60)
print("PROTECTED WORDS: high phishing-vs-legitimate ratio (kept as features)")
print("=" * 60)
protected = set()
for word, p_count in phishing_counter.items():
    if word in PUBLISHED_STOPWORDS:
        continue
    p_freq = p_count / total_phishing
    l_freq = legitimate_counter.get(word, 1) / total_legit
    if p_count >= 30 and (p_freq / l_freq) >= 3:
        protected.add(word)

discriminative = sorted(
    protected,
    key=lambda w: (phishing_counter[w] / total_phishing) /
                  (legitimate_counter.get(w, 1) / total_legit),
    reverse=True,
)[:15]
for word in discriminative:
    p_freq = phishing_counter[word] / total_phishing * 100
    l_freq = legitimate_counter.get(word, 0) / total_legit * 100
    ratio = p_freq / max(l_freq, 0.001)
    print(f"  {word:20s}  phishing={p_freq:.2f}%  legitimate={l_freq:.2f}%  ratio={ratio:.1f}x")

final_stopwords = (PUBLISHED_STOPWORDS | balanced_stopwords) - protected

print("\n" + "=" * 60)
print("FINAL STOPWORD LIST")
print("=" * 60)
print(f"Published baseline:           {len(PUBLISHED_STOPWORDS)} words")
print(f"Corpus-balanced additions:    {len(new_candidates - protected)} words")
print(f"Protected (NOT removed):      {len(protected)} words")
print(f"Final stopword list size:     {len(final_stopwords)} words")

out_path = PROCESSED_DIR / "swahili_stopwords.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for word in sorted(final_stopwords):
        f.write(word + "\n")
print(f"\n[OK] Saved: {out_path}")

protected_path = PROCESSED_DIR / "phishing_indicator_words.txt"
with open(protected_path, "w", encoding="utf-8") as f:
    for word in sorted(protected):
        ratio = (phishing_counter[word] / total_phishing) / \
                (legitimate_counter.get(word, 1) / total_legit)
        f.write(f"{word}\t{ratio:.2f}\n")
print(f"[OK] Saved: {protected_path} (with phishing/legit ratios)")
print("=" * 60)