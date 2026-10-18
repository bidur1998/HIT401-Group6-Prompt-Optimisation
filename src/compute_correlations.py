"""
Step 4 - Pearson correlation between accuracy and each reliability metric.

Reads   results/model_metrics_all.csv, results/reliability_consistency_all.csv
Writes  results/accuracy_reliability_exact_correlations.csv

Each data point is one (model, prompt category) pair, i.e. n = 2 models x 4
prompts = 8 points. Accuracy is the mean over the 4 iterations.
"""

import argparse

import pandas as pd

import config


def strength_label(r):
    a = abs(r)
    if a >= 0.80:
        s = "Very strong"
    elif a >= 0.60:
        s = "Strong"
    elif a >= 0.40:
        s = "Moderate"
    elif a >= 0.20:
        s = "Weak"
    else:
        s = "Very weak / Negligible"
    return f"{s} {'positive' if r >= 0 else 'negative'}"


def compute_correlations(metrics_file, reliability_file, output_file):
    metrics_df = pd.read_csv(metrics_file)
    reliability_df = pd.read_csv(reliability_file)

    acc = metrics_df.groupby(["model", "prompt_category"])["accuracy_percent"].mean().reset_index()
    df = pd.merge(acc, reliability_df, on=["model", "prompt_category"])

    pairs = [
        ("reliability_percent", "Accuracy vs Reliability (%)"),
        ("consistency_percent", "Accuracy vs Consistency (%)"),
        ("average_pairwise_agreement_percent", "Accuracy vs Pairwise Agreement (%)"),
        ("average_cohens_kappa", "Accuracy vs Cohen's Kappa"),
    ]
    rows = []
    for col, label in pairs:
        r = df["accuracy_percent"].corr(df[col])  # Pearson by default
        rows.append({"Metric Pair": label,
                     "Correlation Coefficient (r)": round(r, 4),
                     "Strength": strength_label(r)})

    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False)
    print(f"Data points (model x prompt pairs): {len(df)}")
    print(out.to_string(index=False))
    print(f"\nSaved {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default=str(config.METRICS_FILE))
    p.add_argument("--reliability", default=str(config.RELIABILITY_FILE))
    p.add_argument("--output", default=str(config.CORRELATIONS_FILE))
    a = p.parse_args()
    compute_correlations(a.metrics, a.reliability, a.output)
