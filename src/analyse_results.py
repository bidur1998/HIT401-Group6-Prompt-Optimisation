"""
Runs steps 2-4 in one go (no API calls, no token needed):

    python src/analyse_results.py

Regenerates every metrics CSV in results/ from results/model_outputs.csv.
"""

import config
from compute_accuracy import compute_accuracy
from compute_correlations import compute_correlations
from compute_reliability import compute_reliability

if __name__ == "__main__":
    print("\n### STEP 2: ACCURACY ###")
    compute_accuracy(config.MODEL_OUTPUTS_FILE, config.METRICS_FILE, config.AVG_ACCURACY_FILE)
    print("\n### STEP 3: RELIABILITY & CONSISTENCY ###")
    compute_reliability(config.MODEL_OUTPUTS_FILE, config.RELIABILITY_FILE, config.AVG_RELIABILITY_FILE)
    print("\n### STEP 4: CORRELATIONS ###")
    compute_correlations(config.METRICS_FILE, config.RELIABILITY_FILE, config.CORRELATIONS_FILE)
