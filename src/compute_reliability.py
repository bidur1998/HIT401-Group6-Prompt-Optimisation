"""
Step 3 - Reliability and consistency across repeated iterations.

Reads   results/model_outputs.csv
Writes  results/reliability_consistency_all.csv (per model x prompt)
        results/overall_average_reliability.csv (mean per prompt)

Metrics (per model x prompt category, over the n iterations):
  Reliability (%)        mean over samples of (most frequent label count / n)
  Consistency (%)        share of samples whose label is identical in all n runs
  Pairwise agreement (%) mean agreement over all n(n-1)/2 iteration pairs
  Cohen's kappa          mean kappa over all iteration pairs (chance-corrected)
"""

import argparse
import csv
from collections import defaultdict
from itertools import combinations

import pandas as pd

import config


def cohens_kappa(labels1, labels2):
    if not labels1:
        return 0.0
    total = len(labels1)
    observed = sum(a == b for a, b in zip(labels1, labels2)) / total
    expected = 0.0
    for label in set(labels1) | set(labels2):
        expected += (labels1.count(label) / total) * (labels2.count(label) / total)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def compute_reliability(input_file, output_file, avg_file):
    categories = config.PROMPT_CATEGORIES
    # data[model][category][sample_id][iteration] = prediction
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    with open(input_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in categories:
                data[row["model"].strip()][c][row["id"].strip()][row["iteration"].strip()] = \
                    row[f"{c}_output"].strip().lower()

    results = []
    for model, cat_data in data.items():
        for category, samples in cat_data.items():
            iterations = sorted({it for s in samples.values() for it in s},
                                key=lambda x: int(x) if x.isdigit() else x)
            n = len(iterations)
            if n < 2:
                print(f"Skipping {model} | {category}: only {n} iteration")
                continue

            complete = [sid for sid, s in samples.items() if all(it in s for it in iterations)]
            if not complete:
                continue

            # Reliability: majority-label frequency per sample
            sample_rel, fully_consistent = [], 0
            for sid in complete:
                preds = [samples[sid][it] for it in iterations]
                counts = defaultdict(int)
                for p in preds:
                    counts[p] += 1
                sample_rel.append(max(counts.values()) / n)
                if len(set(preds)) == 1:
                    fully_consistent += 1

            reliability = sum(sample_rel) / len(sample_rel) * 100
            consistency = fully_consistent / len(complete) * 100

            # Pairwise agreement and Cohen's kappa between every pair of iterations
            agreements, kappas = [], []
            for i1, i2 in combinations(iterations, 2):
                l1 = [samples[sid][i1] for sid in complete]
                l2 = [samples[sid][i2] for sid in complete]
                agreements.append(sum(a == b for a, b in zip(l1, l2)) / len(l1) * 100)
                kappas.append(cohens_kappa(l1, l2))

            results.append({
                "model": model,
                "prompt_category": category,
                "num_iterations": n,
                "total_samples": len(complete),
                "reliability_percent": round(reliability, 2),
                "consistency_percent": round(consistency, 2),
                "average_pairwise_agreement_percent": round(sum(agreements) / len(agreements), 2),
                "average_cohens_kappa": round(sum(kappas) / len(kappas), 4),
            })

    fieldnames = ["model", "prompt_category", "num_iterations", "total_samples",
                  "reliability_percent", "consistency_percent",
                  "average_pairwise_agreement_percent", "average_cohens_kappa"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    df_rel = pd.read_csv(output_file)
    metrics = ["reliability_percent", "consistency_percent",
               "average_pairwise_agreement_percent", "average_cohens_kappa"]
    avg = df_rel.groupby("prompt_category")[metrics].mean().reset_index()
    avg.columns = ["Prompt Category", "Reliability (%)", "Consistency (%)",
                   "Pairwise Agreement (%)", "Cohen's Kappa"]
    avg.to_csv(avg_file, index=False)

    print(pd.DataFrame(results).to_string(index=False))
    print("\nAverage per prompt category:")
    print(avg.to_string(index=False))
    print(f"\nSaved {output_file}\nSaved {avg_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(config.MODEL_OUTPUTS_FILE))
    p.add_argument("--output", default=str(config.RELIABILITY_FILE))
    p.add_argument("--average", default=str(config.AVG_RELIABILITY_FILE))
    a = p.parse_args()
    compute_reliability(a.input, a.output, a.average)
