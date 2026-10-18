"""
Step 2 - Accuracy per model, iteration and prompt category.

Reads   results/model_outputs.csv
Writes  results/model_metrics_all.csv        (per model x iteration x prompt)
        results/overall_average_accuracy.csv (mean accuracy per prompt)

Accuracy = correct predictions / total predictions x 100
A prediction of "unknown" (unparsed or failed API call) counts as incorrect.
"""

import argparse
import csv
from collections import defaultdict

import pandas as pd

import config


def compute_accuracy(input_file, metrics_file, avg_file):
    categories = config.PROMPT_CATEGORIES
    metrics = defaultdict(lambda: defaultdict(
        lambda: {c: {"correct": 0, "total": 0, "errors": []} for c in categories}))

    with open(input_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "unknown_model").strip()
            iteration = row.get("iteration", "unknown").strip()
            expected = row["expected_label"].strip().lower()
            for c in categories:
                output = row[f"{c}_output"].strip().lower()
                metrics[model][iteration][c]["total"] += 1
                if output == expected:
                    metrics[model][iteration][c]["correct"] += 1
                else:
                    metrics[model][iteration][c]["errors"].append(row["id"])

    rows = []
    for model, iterations in metrics.items():
        print("\n" + "=" * 60 + f"\nMODEL: {model}\n" + "=" * 60)
        for iteration, counts in iterations.items():
            print(f"\nIteration {iteration}")
            print(f"{'Prompt':<14}{'Correct':<9}{'Total':<7}{'Accuracy':<10}Misclassified IDs")
            for c in categories:
                correct, total = counts[c]["correct"], counts[c]["total"]
                acc = round(correct / total * 100, 2) if total else 0.0
                errs = ";".join(map(str, counts[c]["errors"])) or "none"
                print(f"{c:<14}{correct:<9}{total:<7}{acc:<10}{errs}")
                rows.append({
                    "model": model, "iteration": iteration, "prompt_category": c,
                    "correct": correct, "total": total, "accuracy_percent": acc,
                    "misclassified_ids": errs,
                })

    fieldnames = ["model", "iteration", "prompt_category", "correct", "total",
                  "accuracy_percent", "misclassified_ids"]
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.read_csv(metrics_file)
    avg = df.groupby("prompt_category")["accuracy_percent"].mean().reset_index()
    avg.columns = ["Prompt Technique", "Average Accuracy (%)"]
    avg.to_csv(avg_file, index=False)

    print("\nAverage accuracy per prompt technique:")
    print(avg.to_string(index=False))
    print(f"\nSaved {metrics_file}\nSaved {avg_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(config.MODEL_OUTPUTS_FILE))
    p.add_argument("--metrics", default=str(config.METRICS_FILE))
    p.add_argument("--average", default=str(config.AVG_ACCURACY_FILE))
    a = p.parse_args()
    compute_accuracy(a.input, a.metrics, a.average)
