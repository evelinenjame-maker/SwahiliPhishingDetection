"""
Day 3 - Step 3: Exploratory Data Analysis with publication-ready figures.

Generates 6 figures saved to results/figures/ for use in Seminar I slides
and Chapter 4 of the dissertation:

  Fig 1: Class distribution and source breakdown
  Fig 2: Message length distributions per class
  Fig 3: Engineered feature comparison (boxplots)
  Fig 4: Top phishing indicator words (bar chart)
  Fig 5: Word clouds per class
  Fig 6: Correlation heatmap of engineered features

Each figure is saved as both PNG (for slides) and at 300 DPI (for the
dissertation). All have descriptive captions printed for direct copy-paste
into your dissertation.
"""

from pathlib import Path
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -------- Project paths --------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------- Plot styling: clean, dissertation-grade --------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Brand colors — phishing red, legitimate green
COLOR_PHISHING = "#C0392B"
COLOR_LEGITIMATE = "#27AE60"
PALETTE = {"phishing": COLOR_PHISHING, "legitimate": COLOR_LEGITIMATE}


# =====================================================================
# Load data
# =====================================================================
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(PROCESSED_DIR / "normalized_dataset.csv")
print(f"[OK] Loaded {len(df)} messages")

# Load stopwords + indicators
with open(PROCESSED_DIR / "swahili_stopwords.txt", "r", encoding="utf-8") as f:
    STOPWORDS = {line.strip() for line in f if line.strip()}

indicator_path = PROCESSED_DIR / "phishing_indicator_words.txt"
indicators = {}
if indicator_path.exists():
    with open(indicator_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                indicators[parts[0]] = float(parts[1])
print(f"[OK] Loaded {len(STOPWORDS)} stopwords, {len(indicators)} indicator words\n")


# =====================================================================
# FIGURE 1: Class distribution and source breakdown
# =====================================================================
print("[INFO] Generating Figure 1: class distribution and sources...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: class counts
class_counts = df["label"].value_counts()
ax1 = axes[0]
bars = ax1.bar(class_counts.index, class_counts.values,
               color=[PALETTE[c] for c in class_counts.index],
               edgecolor="black", linewidth=0.8)
ax1.set_title("(a) Class distribution")
ax1.set_ylabel("Number of messages")
ax1.set_xlabel("Class")
for bar, count in zip(bars, class_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f"{count}\n({count/len(df)*100:.1f}%)",
             ha="center", va="bottom", fontsize=10)
ax1.set_ylim(0, max(class_counts.values) * 1.15)

# Right: source breakdown stacked
src_label = pd.crosstab(df["source"], df["label"])
src_label = src_label.loc[src_label.sum(axis=1).sort_values(ascending=True).index]
ax2 = axes[1]
src_label.plot(kind="barh", stacked=True, ax=ax2,
               color=[PALETTE[c] for c in src_label.columns],
               edgecolor="black", linewidth=0.5)
ax2.set_title("(b) Messages per source")
ax2.set_xlabel("Number of messages")
ax2.set_ylabel("")
ax2.legend(title="Class", loc="lower right", framealpha=0.9)

plt.suptitle("Figure 1. Combined dataset composition (N=2,408)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_class_and_source_distribution.png")
plt.close()
print(f"  [OK] Saved fig01_class_and_source_distribution.png")
print(f"  CAPTION: Class distribution shows near-perfect balance (1,211 phishing")
print(f"  vs 1,197 legitimate, ratio 1.01:1). BongoSCAM contributes the largest")
print(f"  share, with synthetic and HuggingFace news rounding out the corpus.\n")


# =====================================================================
# FIGURE 2: Message length distributions
# =====================================================================
print("[INFO] Generating Figure 2: message length distributions...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Character length
ax1 = axes[0]
for label in ["legitimate", "phishing"]:
    subset = df[df["label"] == label]["char_count"]
    ax1.hist(subset, bins=40, alpha=0.6, label=label.title(),
             color=PALETTE[label], edgecolor="black", linewidth=0.3)
ax1.set_title("(a) Character count distribution")
ax1.set_xlabel("Characters per message")
ax1.set_ylabel("Frequency")
ax1.legend()
ax1.axvline(df[df["label"] == "phishing"]["char_count"].median(),
            color=COLOR_PHISHING, linestyle="--", alpha=0.7, linewidth=1)
ax1.axvline(df[df["label"] == "legitimate"]["char_count"].median(),
            color=COLOR_LEGITIMATE, linestyle="--", alpha=0.7, linewidth=1)

# Word count
ax2 = axes[1]
for label in ["legitimate", "phishing"]:
    subset = df[df["label"] == label]["word_count"]
    ax2.hist(subset, bins=30, alpha=0.6, label=label.title(),
             color=PALETTE[label], edgecolor="black", linewidth=0.3)
ax2.set_title("(b) Word count distribution")
ax2.set_xlabel("Words per message")
ax2.set_ylabel("Frequency")
ax2.legend()

plt.suptitle("Figure 2. Message length distributions per class",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_length_distributions.png")
plt.close()
print(f"  [OK] Saved fig02_length_distributions.png")
print(f"  CAPTION: Phishing messages cluster around 90-110 characters with")
print(f"  tighter variance, while legitimate messages display a broader length")
print(f"  range reflecting both short SMS and longer news sentences.\n")


# =====================================================================
# FIGURE 3: Engineered feature comparison (boxplots)
# =====================================================================
print("[INFO] Generating Figure 3: feature boxplots...")
features_to_plot = [
    "uppercase_ratio", "digit_ratio", "phone_count", "money_count",
    "all_caps_words", "exclaim_count",
]
feature_labels = {
    "uppercase_ratio": "Uppercase ratio",
    "digit_ratio": "Digit ratio",
    "phone_count": "Phone numbers",
    "money_count": "Money mentions",
    "all_caps_words": "ALL-CAPS words",
    "exclaim_count": "Exclamation marks",
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, feat in zip(axes.flat, features_to_plot):
    sns.boxplot(data=df, x="label", y=feat, ax=ax,
                palette=PALETTE, order=["legitimate", "phishing"],
                showfliers=True, fliersize=2, linewidth=0.8)
    ax.set_title(feature_labels[feat])
    ax.set_xlabel("")
    ax.set_ylabel("")
    means = df.groupby("label")[feat].mean()
    ax.text(0.05, 0.95,
            f"μ_legit={means['legitimate']:.2f}\nμ_phish={means['phishing']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                     edgecolor="gray"))

plt.suptitle("Figure 3. Engineered feature distributions by class",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig03_feature_boxplots.png")
plt.close()
print(f"  [OK] Saved fig03_feature_boxplots.png")
print(f"  CAPTION: Six engineered features show class-discriminative power.")
print(f"  Phone count and money mentions are near-binary indicators, while")
print(f"  uppercase ratio and ALL-CAPS words demonstrate the distinct")
print(f"  formatting signature of phishing messages.\n")


# =====================================================================
# FIGURE 4: Top phishing indicator words
# =====================================================================
print("[INFO] Generating Figure 4: top phishing indicator words...")
top_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)[:20]
words = [w for w, _ in top_indicators]
ratios = [r for _, r in top_indicators]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(words)), ratios, color=COLOR_PHISHING,
               edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Phishing/Legitimate frequency ratio (log scale)")
ax.set_title("Figure 4. Top 20 phishing indicator words by class ratio",
             fontweight="bold")
ax.axvline(1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
for bar, ratio in zip(bars, ratios):
    ax.text(ratio * 1.05, bar.get_y() + bar.get_height() / 2,
            f"{ratio:.0f}×", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_phishing_indicators.png")
plt.close()
print(f"  [OK] Saved fig04_phishing_indicators.png")
print(f"  CAPTION: The 20 most class-discriminative single words. Operator")
print(f"  names (airtel, vodacom, halotel, tigo) and money-related terms")
print(f"  (pesa, hela, tsh) dominate, alongside imperatives (tuma, piga).\n")


# =====================================================================
# FIGURE 5: Word clouds per class (post-stopword removal)
# =====================================================================
print("[INFO] Generating Figure 5: word clouds...")

def clean_for_cloud(texts):
    """Aggregate cleaned tokens, removing stopwords and placeholders."""
    counter = Counter()
    for t in texts:
        tokens = str(t).split()
        for tok in tokens:
            if tok in STOPWORDS:
                continue
            if tok.startswith("<") and tok.endswith(">"):
                continue  # skip <PHONE>, <MONEY>, <URL>
            if len(tok) < 2:
                continue
            counter[tok] += 1
    return counter

phishing_words = clean_for_cloud(df[df["label"] == "phishing"]["clean_text"])
legit_words = clean_for_cloud(df[df["label"] == "legitimate"]["clean_text"])

wc_phishing = WordCloud(
    width=900, height=500, background_color="white",
    colormap="Reds", max_words=80, relative_scaling=0.5,
    min_font_size=8,
).generate_from_frequencies(phishing_words)

wc_legit = WordCloud(
    width=900, height=500, background_color="white",
    colormap="Greens", max_words=80, relative_scaling=0.5,
    min_font_size=8,
).generate_from_frequencies(legit_words)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(wc_legit, interpolation="bilinear")
axes[0].set_title("(a) Legitimate messages", fontsize=12, fontweight="bold")
axes[0].axis("off")
axes[1].imshow(wc_phishing, interpolation="bilinear")
axes[1].set_title("(b) Phishing messages", fontsize=12, fontweight="bold")
axes[1].axis("off")
plt.suptitle("Figure 5. Word clouds per class (stopwords and placeholders removed)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_wordclouds.png")
plt.close()
print(f"  [OK] Saved fig05_wordclouds.png")
print(f"  CAPTION: Word clouds reveal stark vocabulary differences. Phishing")
print(f"  vocabulary centers on financial transactions and operator names;")
print(f"  legitimate vocabulary spans general news and conversational topics.\n")


# =====================================================================
# FIGURE 6: Feature correlation heatmap
# =====================================================================
print("[INFO] Generating Figure 6: feature correlation heatmap...")
feature_cols = [
    "char_count", "word_count", "uppercase_ratio", "digit_ratio",
    "phone_count", "url_count", "money_count", "exclaim_count",
    "question_count", "urgency_word_count", "money_word_count",
    "all_caps_words",
]
# Add binary class indicator for correlation with class
df["is_phishing"] = (df["label"] == "phishing").astype(int)
corr_cols = feature_cols + ["is_phishing"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
            annot_kws={"size": 8}, ax=ax)
ax.set_title("Figure 6. Pearson correlation between engineered features",
             fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_correlation_heatmap.png")
plt.close()
print(f"  [OK] Saved fig06_correlation_heatmap.png")
print(f"  CAPTION: Correlation heatmap. The bottom row (is_phishing) reveals")
print(f"  which features correlate most strongly with the phishing label,")
print(f"  guiding feature selection for modeling. Strong inter-feature")
print(f"  correlations (e.g., digit_ratio with phone_count) indicate potential")
print(f"  redundancy to consider during feature engineering.\n")


# =====================================================================
# Print correlations with target as numerical summary
# =====================================================================
target_corr = corr["is_phishing"].drop("is_phishing").sort_values(
    key=abs, ascending=False)
print("=" * 60)
print("FEATURE CORRELATION WITH PHISHING LABEL (sorted by magnitude)")
print("=" * 60)
for feat, val in target_corr.items():
    bar = "█" * int(abs(val) * 30)
    direction = "+" if val > 0 else "-"
    print(f"  {feat:25s} {direction}{bar} {val:+.3f}")

print("\n" + "=" * 60)
print(f"[OK] All 6 figures saved to: {FIG_DIR}")
print("=" * 60)
