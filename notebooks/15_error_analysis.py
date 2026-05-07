"""
Day 6 - Error Analysis.

Examines every misclassified message from the 3 best-config models on the
held-out test set and the BongoSCAM-only test set. Produces:

  results/error_analysis.csv      - all misclassified messages with details
  results/figures/fig12_error_matrix.png - error overlap heatmap
  results/figures/fig13_error_categories.png - errors by source/category
  results/error_analysis_report.md - summary report for dissertation

Run from project root:
    python notebooks\\15_error_analysis.py
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ALGO_LABELS = {
    "naive_bayes": "Naïve Bayes",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
}
BEST_CONFIG = "combined_keep"  # the most expressive configuration

# =====================================================================
# Load test split (with original messages and metadata)
# =====================================================================
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

test_df = pd.read_csv(SPLITS_DIR / "test.csv")
bongo_df = pd.read_csv(SPLITS_DIR / "bongoscam_test.csv")

print(f"[OK] Test set: {len(test_df)} messages")
print(f"[OK] BongoSCAM test subset: {len(bongo_df)} messages\n")

# =====================================================================
# Load predictions for each algorithm and identify errors
# =====================================================================
all_errors = []

for algo in ["naive_bayes", "linear_svm", "random_forest"]:
    pred_path = PRED_DIR / f"{algo}__{BEST_CONFIG}.npz"
    if not pred_path.exists():
        print(f"[WARN] Predictions not found for {algo}")
        continue

    data = np.load(pred_path)

    # ---- Errors on the full test set ----
    test_pred = data["test_pred"]
    test_true = data["test_true"]
    test_errors = test_pred != test_true
    n_errors_test = int(test_errors.sum())

    if n_errors_test > 0:
        for idx in np.where(test_errors)[0]:
            row = test_df.iloc[idx]
            error_type = "FN" if test_true[idx] == 1 else "FP"
            all_errors.append({
                "algorithm": algo,
                "test_set": "test",
                "row_id": int(row.get("row_id", idx)),
                "true_label": "phishing" if test_true[idx] == 1 else "legitimate",
                "predicted": "phishing" if test_pred[idx] == 1 else "legitimate",
                "error_type": error_type,
                "source": row["source"],
                "message": row["message"][:300],  # truncate very long messages
                "char_count": row.get("char_count", ""),
                "word_count": row.get("word_count", ""),
                "phone_count": row.get("phone_count", ""),
                "uppercase_ratio": row.get("uppercase_ratio", ""),
            })

    # ---- Errors on BongoSCAM-only test ----
    bongo_pred = data["bongo_pred"]
    bongo_true = data["bongo_true"]
    bongo_errors = bongo_pred != bongo_true
    n_errors_bongo = int(bongo_errors.sum())

    if n_errors_bongo > 0:
        for idx in np.where(bongo_errors)[0]:
            row = bongo_df.iloc[idx]
            error_type = "FN" if bongo_true[idx] == 1 else "FP"
            all_errors.append({
                "algorithm": algo,
                "test_set": "bongoscam",
                "row_id": int(row.get("row_id", idx)),
                "true_label": "phishing" if bongo_true[idx] == 1 else "legitimate",
                "predicted": "phishing" if bongo_pred[idx] == 1 else "legitimate",
                "error_type": error_type,
                "source": row["source"],
                "message": row["message"][:300],
                "char_count": row.get("char_count", ""),
                "word_count": row.get("word_count", ""),
                "phone_count": row.get("phone_count", ""),
                "uppercase_ratio": row.get("uppercase_ratio", ""),
            })

    print(f"  {ALGO_LABELS[algo]:15s}  test errors: {n_errors_test:>3d}  "
          f"BongoSCAM errors: {n_errors_bongo:>3d}")

# =====================================================================
# Save the error catalog
# =====================================================================
errors_df = pd.DataFrame(all_errors)
errors_csv = RESULTS_DIR / "error_analysis.csv"
errors_df.to_csv(errors_csv, index=False)
print(f"\n[OK] Saved: {errors_csv}  ({len(errors_df)} error records)")

if len(errors_df) == 0:
    print("\n[!] No errors found — every model classified every message correctly.")
    print("    This is statistically suspicious; we should investigate.")
    raise SystemExit(0)


# =====================================================================
# Error overlap analysis: do algorithms make the SAME mistakes?
# =====================================================================
print("\n" + "=" * 70)
print("ERROR OVERLAP — do different algorithms misclassify the SAME messages?")
print("=" * 70)

# For the test set, build a per-message error indicator per algorithm
overlap_data = {}
for algo in ["naive_bayes", "linear_svm", "random_forest"]:
    pred_path = PRED_DIR / f"{algo}__{BEST_CONFIG}.npz"
    if pred_path.exists():
        data = np.load(pred_path)
        overlap_data[algo] = (data["test_pred"] != data["test_true"]).astype(int)

if len(overlap_data) >= 2:
    overlap_df = pd.DataFrame(overlap_data)
    overlap_df["all_three_wrong"] = overlap_df.sum(axis=1) == 3
    overlap_df["any_wrong"] = overlap_df.sum(axis=1) > 0

    n_total = len(overlap_df)
    n_any = overlap_df["any_wrong"].sum()
    n_all = overlap_df["all_three_wrong"].sum()
    print(f"  Total test messages:       {n_total}")
    print(f"  Errors by ANY algorithm:   {n_any}  ({n_any/n_total*100:.2f}%)")
    print(f"  Errors by ALL three:       {n_all}  "
          f"(systemic difficulty — all algorithms struggle)")

    # Pairwise overlap
    print("\n  Pairwise error overlap (Jaccard similarity):")
    algos_list = list(overlap_data.keys())
    for i, a in enumerate(algos_list):
        for b in algos_list[i+1:]:
            both = ((overlap_df[a] == 1) & (overlap_df[b] == 1)).sum()
            either = ((overlap_df[a] == 1) | (overlap_df[b] == 1)).sum()
            jac = both / max(either, 1)
            print(f"    {a:15s} vs {b:15s}: both wrong on {both} of "
                  f"{either} (Jaccard = {jac:.3f})")


# =====================================================================
# Examine the actual misclassified messages
# =====================================================================
print("\n" + "=" * 70)
print("MISCLASSIFIED MESSAGES (BongoSCAM test set)")
print("=" * 70)

# Prefer the BongoSCAM subset for this view (most realistic)
bongo_errors = errors_df[errors_df["test_set"] == "bongoscam"]

if len(bongo_errors) > 0:
    # Group by error type
    for error_type in ["FN", "FP"]:
        sub = bongo_errors[bongo_errors["error_type"] == error_type]
        sub = sub.drop_duplicates(subset=["row_id"])  # show each unique message once
        if len(sub) == 0:
            continue
        full_label = "FALSE NEGATIVE (missed phishing)" if error_type == "FN" \
            else "FALSE POSITIVE (legit flagged as phishing)"
        print(f"\n--- {full_label} — {len(sub)} unique message(s) ---")
        for _, row in sub.iterrows():
            algos_that_erred = bongo_errors[
                (bongo_errors["row_id"] == row["row_id"]) &
                (bongo_errors["error_type"] == error_type)
            ]["algorithm"].tolist()
            print(f"\n  Misclassified by: {', '.join(algos_that_erred)}")
            print(f"  True label:       {row['true_label']}")
            print(f"  Source:           {row['source']}")
            print(f"  Message:          {row['message']}")
            if row['char_count'] != "":
                print(f"  Features:         chars={row['char_count']}, "
                      f"phones={row['phone_count']}, "
                      f"upper_ratio={float(row['uppercase_ratio']):.2f}")


# =====================================================================
# Figure 12: Error counts per algorithm and error type
# =====================================================================
print("\n[INFO] Generating fig12_error_counts.png...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: error counts per algorithm on each test set
counts = errors_df.groupby(["algorithm", "test_set"]).size().unstack(fill_value=0)
if not counts.empty:
    # Reorder
    algo_order = [a for a in ["naive_bayes", "linear_svm", "random_forest"]
                  if a in counts.index]
    counts = counts.reindex(algo_order)
    ax = axes[0]
    counts.plot(kind="bar", ax=ax,
                color={"test": "#3B7DD8", "bongoscam": "#27AE60"},
                edgecolor="black", linewidth=0.5)
    ax.set_title("(a) Error count by algorithm and test set")
    ax.set_xlabel("")
    ax.set_ylabel("Number of misclassifications")
    ax.set_xticklabels([ALGO_LABELS[a] for a in counts.index], rotation=0)
    ax.legend(title="Test set", loc="best")
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)

# Right: error type breakdown (FN vs FP)
err_type = errors_df.groupby(["algorithm", "error_type"]).size().unstack(fill_value=0)
if not err_type.empty:
    algo_order = [a for a in ["naive_bayes", "linear_svm", "random_forest"]
                  if a in err_type.index]
    err_type = err_type.reindex(algo_order)
    ax = axes[1]
    err_type.plot(kind="bar", ax=ax,
                  color={"FN": "#C0392B", "FP": "#F39C12"},
                  edgecolor="black", linewidth=0.5)
    ax.set_title("(b) Error breakdown by type")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels([ALGO_LABELS[a] for a in err_type.index], rotation=0)
    ax.legend(title="Error type", labels=["False Negative", "False Positive"])
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)

plt.suptitle("Figure 12. Error analysis on best feature configuration "
             f"({BEST_CONFIG})", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig12_error_counts.png")
plt.close()
print("  [OK] Saved fig12_error_counts.png")


# =====================================================================
# Figure 13: Errors by message source (where do they come from?)
# =====================================================================
print("\n[INFO] Generating fig13_error_by_source.png...")

if "source" in errors_df.columns:
    src_counts = errors_df.groupby(["source", "error_type"]).size().unstack(fill_value=0)
    if not src_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        src_counts.plot(kind="barh", stacked=True, ax=ax,
                        color={"FN": "#C0392B", "FP": "#F39C12"},
                        edgecolor="black", linewidth=0.5)
        ax.set_title("Figure 13. Errors by message source",
                     fontweight="bold")
        ax.set_xlabel("Number of misclassifications (across all algorithms)")
        ax.set_ylabel("")
        ax.legend(title="Error type", labels=["False Negative", "False Positive"])
        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig13_error_by_source.png")
        plt.close()
        print("  [OK] Saved fig13_error_by_source.png")


# =====================================================================
# Generate markdown summary report
# =====================================================================
print("\n[INFO] Generating error analysis report...")

# Aggregate stats
n_test = len(test_df)
n_bongo = len(bongo_df)
n_errors_per_algo_test = errors_df[errors_df["test_set"] == "test"].groupby("algorithm").size()
n_errors_per_algo_bongo = errors_df[errors_df["test_set"] == "bongoscam"].groupby("algorithm").size()

report_lines = [
    "# Error Analysis Report",
    "",
    f"Generated from `combined_keep` configuration predictions on:",
    f"- Held-out test set: {n_test} messages",
    f"- BongoSCAM-only subset: {n_bongo} messages",
    "",
    "## Error counts per algorithm",
    "",
    "| Algorithm | Test set errors | BongoSCAM errors | Test acc | BongoSCAM acc |",
    "|---|---|---|---|---|",
]

for algo in ["naive_bayes", "linear_svm", "random_forest"]:
    n_t = int(n_errors_per_algo_test.get(algo, 0))
    n_b = int(n_errors_per_algo_bongo.get(algo, 0))
    acc_t = (n_test - n_t) / n_test
    acc_b = (n_bongo - n_b) / n_bongo if n_bongo > 0 else 1.0
    report_lines.append(
        f"| {ALGO_LABELS[algo]} | {n_t} | {n_b} | "
        f"{acc_t:.4f} | {acc_b:.4f} |"
    )

# Error type breakdown
report_lines.extend([
    "",
    "## Error type breakdown",
    "",
    "False Negatives (FN) = phishing missed (most dangerous in deployment).",
    "False Positives (FP) = legitimate flagged as phishing (recoverable).",
    "",
])

et = errors_df.groupby(["algorithm", "error_type"]).size().unstack(fill_value=0)
if not et.empty:
    report_lines.append("| Algorithm | False Negatives | False Positives |")
    report_lines.append("|---|---|---|")
    for algo in ["naive_bayes", "linear_svm", "random_forest"]:
        if algo in et.index:
            fn = int(et.loc[algo].get("FN", 0))
            fp = int(et.loc[algo].get("FP", 0))
            report_lines.append(f"| {ALGO_LABELS[algo]} | {fn} | {fp} |")

# Examined examples
report_lines.extend([
    "",
    "## Examined examples (BongoSCAM test set)",
    "",
    "### False Negatives (missed phishing)",
    "",
])

bongo_fn = errors_df[
    (errors_df["test_set"] == "bongoscam") &
    (errors_df["error_type"] == "FN")
].drop_duplicates(subset=["row_id"])

if len(bongo_fn) == 0:
    report_lines.append("*None — all phishing messages correctly identified*")
else:
    for _, row in bongo_fn.iterrows():
        algos = errors_df[
            (errors_df["row_id"] == row["row_id"]) &
            (errors_df["error_type"] == "FN")
        ]["algorithm"].apply(lambda a: ALGO_LABELS[a]).tolist()
        report_lines.append(f"- **Missed by:** {', '.join(algos)}")
        report_lines.append(f"  - **Message:** {row['message']}")
        report_lines.append("")

report_lines.extend([
    "",
    "### False Positives (legitimate flagged as phishing)",
    "",
])

bongo_fp = errors_df[
    (errors_df["test_set"] == "bongoscam") &
    (errors_df["error_type"] == "FP")
].drop_duplicates(subset=["row_id"])

if len(bongo_fp) == 0:
    report_lines.append("*None — all legitimate messages correctly identified*")
else:
    for _, row in bongo_fp.iterrows():
        algos = errors_df[
            (errors_df["row_id"] == row["row_id"]) &
            (errors_df["error_type"] == "FP")
        ]["algorithm"].apply(lambda a: ALGO_LABELS[a]).tolist()
        report_lines.append(f"- **Flagged by:** {', '.join(algos)}")
        report_lines.append(f"  - **Message:** {row['message']}")
        report_lines.append("")

# Discussion
report_lines.extend([
    "",
    "## Discussion for dissertation",
    "",
    "Operationally, false negatives (missed phishing) are more dangerous than ",
    "false positives, since the latter can be filtered through user review. ",
    "The error rate for each algorithm gives an upper bound on real-world ",
    "deployment risk. Error overlap across algorithms suggests certain ",
    "messages are genuinely ambiguous; these warrant qualitative discussion ",
    "in the dissertation Limitations section.",
    "",
])

report_path = RESULTS_DIR / "error_analysis_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[OK] Saved: {report_path}")

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Total error records:    {len(errors_df)}")
print(f"  Unique misclassified:   {errors_df['row_id'].nunique()}")
print(f"  Outputs:")
print(f"    - {errors_csv}")
print(f"    - {report_path}")
print(f"    - {FIG_DIR / 'fig12_error_counts.png'}")
print(f"    - {FIG_DIR / 'fig13_error_by_source.png'}")
print("=" * 70)
