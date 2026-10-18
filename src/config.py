"""
Experiment configuration for the HIT401 Group 6 prompt-optimisation study.

Every parameter that affects the results lives here, so the experiment can be
re-run or varied without touching the pipeline code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DATASET_FILE = DATA_DIR / "dataset.csv"
MODEL_OUTPUTS_FILE = RESULTS_DIR / "model_outputs.csv"
METRICS_FILE = RESULTS_DIR / "model_metrics_all.csv"
AVG_ACCURACY_FILE = RESULTS_DIR / "overall_average_accuracy.csv"
RELIABILITY_FILE = RESULTS_DIR / "reliability_consistency_all.csv"
AVG_RELIABILITY_FILE = RESULTS_DIR / "overall_average_reliability.csv"
CORRELATIONS_FILE = RESULTS_DIR / "accuracy_reliability_exact_correlations.csv"

# ---------------------------------------------------------------------------
# Models (served through the Hugging Face Inference Providers router)
# ---------------------------------------------------------------------------
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
]

API_URL = "https://router.huggingface.co/v1/chat/completions"

# ---------------------------------------------------------------------------
# Generation / execution parameters
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 4          # repeated runs per model, used for reliability
TEMPERATURE = 0.1           # near-deterministic decoding
MAX_TOKENS = 300            # maximum length of each model response
REQUEST_TIMEOUT = 60        # seconds per HTTP request
MAX_RETRIES = 3             # retries on HTTP 429 / 5xx / network errors
INITIAL_RETRY_DELAY = 2     # seconds; doubled after every retry
DELAY_BETWEEN_CALLS = 0.2   # seconds; pause between requests

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
VALID_LABELS = ["positive", "negative", "neutral"]
UNKNOWN_LABEL = "unknown"   # used when no label can be parsed / API failed

# ---------------------------------------------------------------------------
# Zero-shot prompt templates (exactly as used in the experiment)
# {text} is replaced with each dataset sample.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "baseline": """Classify the sentiment of the following text as positive, negative, or neutral.
Text: {text}
Sentiment:""",

    "instruction": """You are given a short piece of text. Read the full text carefully, including
tone and any implied meaning such as sarcasm, before deciding. Classify its
overall sentiment as exactly one of: positive, negative, or neutral.
Respond with only the single label word.
Text: {text}
Sentiment:""",

    "role": """You are an experienced sentiment analysis expert who specialises in detecting
nuance, sarcasm, and subtle tone in customer feedback. Analyse the sentiment
of the text below as positive, negative, or neutral.
Text: {text}
Sentiment:""",

    "structured": """Classify the sentiment of the text below. Respond only in the following JSON
format with no other text: {{"sentiment": "positive"}}
Text: {text}""",
}

PROMPT_CATEGORIES = list(PROMPT_TEMPLATES.keys())
