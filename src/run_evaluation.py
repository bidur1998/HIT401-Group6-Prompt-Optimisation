"""
Step 1 - Run the experiment.

Sends every dataset sample, under every prompt template, to every model,
NUM_ITERATIONS times, and saves raw + parsed predictions to
results/model_outputs.csv.

Usage:
    export HF_API_TOKEN=hf_xxx          # Windows: set HF_API_TOKEN=hf_xxx
    python src/run_evaluation.py
    python src/run_evaluation.py --iterations 1 --limit 3   # quick smoke test
"""

import argparse
import json
import os
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

import config


def get_token():
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        try:  # optional: read from a local .env file if python-dotenv is installed
            from dotenv import load_dotenv
            load_dotenv(config.ROOT_DIR / ".env")
            token = os.environ.get("HF_API_TOKEN")
        except ImportError:
            pass
    if not token:
        raise ValueError(
            "HF_API_TOKEN is not set. Create a token at "
            "https://huggingface.co/settings/tokens and export it as an "
            "environment variable (or put it in a .env file)."
        )
    return token


def query_huggingface(model_name, prompt, token):
    """Send one chat-completion request. Returns the response text, or "" on failure."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
    }

    delay = config.INITIAL_RETRY_DELAY
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.post(
                config.API_URL, headers=headers, json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                tqdm.write(f"Rate limited. Retry {attempt + 1}/{config.MAX_RETRIES}")
                time.sleep(delay)
                delay *= 2
            elif response.status_code in (401, 403, 404, 422):
                tqdm.write(f"API request failed ({response.status_code}): {response.text}")
                break
            elif response.status_code >= 500:
                tqdm.write(f"Server error ({response.status_code}). Retrying...")
                time.sleep(delay)
                delay *= 2
            else:
                tqdm.write(f"API request failed ({response.status_code}): {response.text}")
                break
        except requests.RequestException as e:
            tqdm.write(f"REQUEST ERROR: {e}")
            time.sleep(delay)
            delay *= 2

    return ""


def parse_output(raw_output, category):
    """Extract a single sentiment label from free-text or JSON model output."""
    raw_output = str(raw_output).strip()
    if not raw_output:
        return config.UNKNOWN_LABEL

    if category == "structured":
        try:
            match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                sentiment = str(data.get("sentiment", "")).lower().strip()
                if sentiment in config.VALID_LABELS:
                    return sentiment
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    raw_lower = raw_output.lower().strip()
    if raw_lower in config.VALID_LABELS:
        return raw_lower

    match = re.search(r"\b(positive|negative|neutral)\b", raw_lower)
    if match:
        return match.group(1)

    return config.UNKNOWN_LABEL


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(config.DATASET_FILE))
    parser.add_argument("--output", default=str(config.MODEL_OUTPUTS_FILE))
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS)
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N samples (for testing)")
    args = parser.parse_args()

    token = get_token()
    df = pd.read_csv(args.dataset)
    if args.limit:
        df = df.head(args.limit)

    total_requests = len(config.MODELS) * args.iterations * len(df) * len(config.PROMPT_TEMPLATES)
    print("=" * 70)
    print(f"Models: {len(config.MODELS)} | Samples: {len(df)} | "
          f"Prompts: {len(config.PROMPT_TEMPLATES)} | Iterations: {args.iterations}")
    print(f"Total API requests: {total_requests}")
    print("=" * 70)

    results = []
    for model_name in config.MODELS:
        for iteration in range(1, args.iterations + 1):
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"{model_name.split('/')[-1]} | iter {iteration}/{args.iterations}"):
                text_input = str(row["text"])
                record = {
                    "id": row["id"],
                    "text": text_input,
                    "model": model_name,
                    "iteration": iteration,
                    "expected_label": str(row["expected_label"]).strip().lower(),
                }
                for category, template in config.PROMPT_TEMPLATES.items():
                    prompt = template.format(text=text_input)
                    raw_res = query_huggingface(model_name, prompt, token)
                    record[f"{category}_output"] = parse_output(raw_res, category)
                    record[f"{category}_raw_output"] = raw_res
                    time.sleep(config.DELAY_BETWEEN_CALLS)
                results.append(record)

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    failed = (out[[f"{c}_output" for c in config.PROMPT_CATEGORIES]] == config.UNKNOWN_LABEL).sum().sum()
    print(f"\nSaved {len(out)} rows to {args.output}")
    print(f"Unparsed / failed responses ('unknown'): {failed}")


if __name__ == "__main__":
    main()
