# Prompt Optimisation Strategies for Improving Accuracy and Reliability in LLMs

**HIT401 Capstone Research Project, Group 6, Charles Darwin University (2026)**
Amrit KC · Bidur Chautariya · Naresh Puri · Sagar Bhandari
Supervisor: Pritika

This repository contains the source code, dataset, raw model outputs and computed metrics for our study on how **zero-shot prompt design** affects the **accuracy** and **reliability** of Large Language Models (LLMs) on a sentiment-classification task. It is Appendix C (Data and Source Code) of our final report.

---

## Contents

- [Research questions](#research-questions)
- [Repository structure](#repository-structure)
- [Experimental setup](#experimental-setup)
- [Data sources](#data-sources)
- [Environment configuration](#environment-configuration)
- [Execution steps](#execution-steps)
- [Output files](#output-files)
- [Evaluation metrics](#evaluation-metrics)
- [Results summary](#results-summary)
- [Known issues and limitations](#known-issues-and-limitations)

---

## Research questions

- **RQ1:** How do different prompt wording strategies influence the accuracy of zero-shot LLM performance?
- **RQ2:** How does prompt design influence the reliability and consistency of zero-shot LLM outputs?

---

## Repository structure

```
hit401-group6-prompt-optimisation/
├── README.md
├── requirements.txt              # Python dependencies
├── .env.example                  # template for your Hugging Face token
├── .gitignore
├── data/
│   └── dataset.csv               # 20 labelled sentiment samples (input)
├── src/
│   ├── config.py                 # ALL experiment parameters and prompt templates
│   ├── run_evaluation.py         # Step 1: query the LLMs and save predictions
│   ├── compute_accuracy.py       # Step 2: accuracy per model/iteration/prompt
│   ├── compute_reliability.py    # Step 3: reliability, consistency, agreement, kappa
│   ├── compute_correlations.py   # Step 4: Pearson r between accuracy and reliability
│   └── analyse_results.py        # runs Steps 2-4 in one command
├── notebooks/
│   └── Model_Evaluation.ipynb    # original Google Colab notebook used for the study
└── results/                      # outputs from our experimental run
    ├── model_outputs.csv
    ├── model_metrics_all.csv
    ├── overall_average_accuracy.csv
    ├── reliability_consistency_all.csv
    ├── overall_average_reliability.csv
    └── accuracy_reliability_exact_correlations.csv
```

The `src/` scripts are a modular version of the notebook with identical logic. Running `src/analyse_results.py` on `results/model_outputs.csv` reproduces every metrics file in `results/` exactly.

---

## Experimental setup

A controlled comparative experiment. The **independent variable** is the prompt design. The **dependent variables** are accuracy and reliability. The dataset, models, generation settings and evaluation procedure stay the same across all conditions.

### Task
Three-class sentiment classification: `positive`, `negative` or `neutral`.

### Models
Both models are accessed through the Hugging Face Inference Providers router (OpenAI-compatible chat-completions endpoint `https://router.huggingface.co/v1/chat/completions`):

| Model | Hugging Face ID |
|---|---|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| DeepSeek V3.1 Terminus | `deepseek-ai/DeepSeek-V3.1-Terminus` |

### Parameter settings
All parameters are defined in [`src/config.py`](src/config.py).

| Parameter | Value | Purpose |
|---|---|---|
| Prompting mode | Zero-shot | No examples given to the model |
| `TEMPERATURE` | `0.1` | Near-deterministic decoding |
| `MAX_TOKENS` | `300` | Maximum response length |
| `NUM_ITERATIONS` | `4` | Repeated runs per model, used for reliability analysis |
| `REQUEST_TIMEOUT` | `60` s | Per-request HTTP timeout |
| `MAX_RETRIES` | `3` | Retries on HTTP 429, 5xx or network errors |
| `INITIAL_RETRY_DELAY` | `2` s | Exponential back-off (doubles after each retry) |
| `DELAY_BETWEEN_CALLS` | `0.2` s | Pause between requests to avoid rate limits |

**Scale of one full run:** 2 models × 4 iterations × 20 samples × 4 prompts = **640 API requests**, giving 160 predictions per prompt category.

### Prompt templates
`{text}` is replaced with each dataset sample.

<details>
<summary><b>1. Baseline zero-shot</b>: minimal task instruction</summary>

```text
Classify the sentiment of the following text as positive, negative, or neutral.
Text: {text}
Sentiment:
```
</details>

<details>
<summary><b>2. Instruction-enhanced</b>: detailed guidance and output constraint</summary>

```text
You are given a short piece of text. Read the full text carefully, including
tone and any implied meaning such as sarcasm, before deciding. Classify its
overall sentiment as exactly one of: positive, negative, or neutral.
Respond with only the single label word.
Text: {text}
Sentiment:
```
</details>

<details>
<summary><b>3. Role-based</b>: expert persona</summary>

```text
You are an experienced sentiment analysis expert who specialises in detecting
nuance, sarcasm, and subtle tone in customer feedback. Analyse the sentiment
of the text below as positive, negative, or neutral.
Text: {text}
Sentiment:
```
</details>

<details>
<summary><b>4. Structured output</b>: JSON response format</summary>

```text
Classify the sentiment of the text below. Respond only in the following JSON
format with no other text: {"sentiment": "positive"}
Text: {text}
```
</details>

### Output parsing
Model responses are free text, so `parse_output()` in `run_evaluation.py` extracts one label:
1. For the structured prompt, it first tries to parse the JSON object and read its `sentiment` field.
2. If the whole response is exactly a label, that label is used.
3. Otherwise, it takes the first whole-word match of `positive|negative|neutral` (regex).
4. If nothing matches, or the API call failed, the prediction is recorded as `unknown` and counts as incorrect.

---

## Data sources

**`data/dataset.csv`** holds 20 short English customer-feedback style sentences that the group wrote and labelled for this study. No external or personal data is used. The set was built to include clear positive and negative statements, factual or neutral statements, mixed or ambiguous statements, and sarcasm (e.g. *"Wow, five hours on hold, exactly what I wanted today."*).

| Column | Description |
|---|---|
| `id` | Unique sample identifier (1–20) |
| `text` | Input sentence to classify |
| `expected_label` | Ground-truth label: `positive`, `negative` or `neutral` |

Label distribution: 9 negative, 6 positive, 5 neutral.

**`results/model_outputs.csv`** is the raw experimental data: 160 rows (2 models × 4 iterations × 20 samples). For each prompt it stores the parsed label (`<prompt>_output`) and the full raw model response (`<prompt>_raw_output`).

---

## Environment configuration

### Requirements
- Python **3.10 or newer** (the study ran in Google Colab; the scripts were also tested locally on Python 3.13)
- A free **Hugging Face account** and access token with permission to *"Make calls to Inference Providers"*
- Internet access (only needed for Step 1)

### Option A: Run locally

```bash
# 1. Clone the repository
git clone https://github.com/bidur1998/HIT401-Group6-Prompt-Optimisation.git
cd HIT401-Group6-Prompt-Optimisation

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Hugging Face token (only needed for Step 1)
cp .env.example .env               # then edit .env and paste your token
#   or:  export HF_API_TOKEN=hf_xxx        (macOS/Linux)
#        set HF_API_TOKEN=hf_xxx           (Windows CMD)
#        $env:HF_API_TOKEN="hf_xxx"        (PowerShell)
```

> **Never commit your token.** `.env` is already listed in `.gitignore`.

### Option B: Google Colab (as used in the study)
1. Open `notebooks/Model_Evaluation.ipynb` in Google Colab.
2. Click the **key icon (Secrets)** in the left sidebar, add a secret named `HF_API_TOKEN` with your token, and enable notebook access.
3. Upload `data/dataset.csv` to the Colab session's file panel.
4. Run all cells. The CSV outputs are written to the Colab working directory.

---

## Execution steps

### Step 1: Generate model predictions (needs API token, about 640 requests)
```bash
python src/run_evaluation.py
```
Writes `results/model_outputs.csv`.

Quick smoke test with 3 samples and 1 iteration before a full run:
```bash
python src/run_evaluation.py --iterations 1 --limit 3 --output results/test_outputs.csv
```

> Running Step 1 again **overwrites** our original `results/model_outputs.csv`. LLM outputs are probabilistic and hosted models change over time, so a new run will not exactly match ours. To keep our data, pass `--output results/my_run.csv` and point the analysis scripts at it with `--input`.

### Steps 2–4: Compute all metrics (offline, no token needed)
```bash
python src/analyse_results.py
```
Or run each step on its own:
```bash
python src/compute_accuracy.py       # Step 2: accuracy
python src/compute_reliability.py    # Step 3: reliability and consistency
python src/compute_correlations.py   # Step 4: correlations
```

**To reproduce the figures in our report**, skip Step 1 and run `python src/analyse_results.py`. It rebuilds every metrics file from our saved `model_outputs.csv`, and the results are identical to the report.

---

## Output files

| File | Produced by | Description |
|---|---|---|
| `model_outputs.csv` | Step 1 | Raw and parsed predictions for every model, iteration, sample and prompt |
| `model_metrics_all.csv` | Step 2 | Correct, total, accuracy % and misclassified IDs per model × iteration × prompt |
| `overall_average_accuracy.csv` | Step 2 | Mean accuracy per prompt technique |
| `reliability_consistency_all.csv` | Step 3 | Reliability, consistency, pairwise agreement and Cohen's κ per model × prompt |
| `overall_average_reliability.csv` | Step 3 | The same metrics averaged across both models |
| `accuracy_reliability_exact_correlations.csv` | Step 4 | Pearson r between accuracy and each reliability metric |

---

## Evaluation metrics

Let *n* = number of iterations (4) and *N* = number of samples (20).

| Metric | Definition |
|---|---|
| **Accuracy** | correct predictions ÷ total predictions × 100 |
| **Reliability** | mean over samples of (count of the most frequent label across the *n* runs ÷ *n*) × 100 |
| **Consistency** | % of samples given the **same** label in all *n* runs |
| **Pairwise agreement** | mean % agreement over all *n(n−1)/2* = 6 pairs of iterations |
| **Cohen's kappa (κ)** | κ = (Pₒ − Pₑ) / (1 − Pₑ), averaged over the 6 iteration pairs; corrects for chance agreement |
| **Pearson's r** | linear correlation between mean accuracy and each reliability metric across the 8 model × prompt pairs |

---

## Results summary

### Accuracy (RQ1)

| Prompt | Llama-3.1-8B | DeepSeek-V3.1 | **Average** | Correct / 160 |
|---|---|---|---|---|
| Baseline | 97.50% | 85.00% | **91.25%** | 146 |
| Instruction-enhanced | 100.00% | 86.25% | **93.13%** | 149 |
| Role-based | 81.25% | 90.00% | **85.63%** | 137 |
| Structured output | 85.00% | 77.50% | **81.25%** | 130 |

### Reliability and consistency (RQ2), averaged across both models

| Prompt | Reliability | Consistency | Pairwise agreement | Cohen's κ |
|---|---|---|---|---|
| Baseline | 93.13% | 75.0% | 87.09% | 0.814 |
| Instruction-enhanced | **95.63%** | **82.5%** | **91.25%** | **0.880** |
| Role-based | 94.38% | 77.5% | 88.75% | 0.843 |
| Structured output | 94.38% | 77.5% | 88.75% | 0.832 |

### Accuracy vs reliability correlation (n = 8 model × prompt pairs)

| Metric pair | Pearson r | Strength |
|---|---|---|
| Accuracy vs Reliability | 0.446 | Moderate positive |
| Accuracy vs Consistency | 0.449 | Moderate positive |
| Accuracy vs Pairwise agreement | 0.449 | Moderate positive |
| Accuracy vs Cohen's κ | 0.473 | Moderate positive |

**Main finding:** the instruction-enhanced prompt gave both the highest accuracy and the highest stability. Role-based and structured-output prompts lowered accuracy, mainly by misreading neutral or factual statements (e.g. ID 3, 16) as positive.

---

## Known issues and limitations

- **Failed API calls are counted as errors.** 29 of the 640 predictions (4.5%) are `unknown` because the API returned an empty response. All of them came from DeepSeek-V3.1-Terminus in iterations 2 and 3 (sample IDs 1–6 in iteration 3; IDs 19–20 in iteration 2). The empty `*_raw_output` cells in `model_outputs.csv` mark these rows. They lower DeepSeek's accuracy and reliability scores (e.g. iteration 3 accuracy of 65–70%) and reflect provider availability, not model judgement.
- **Small dataset.** 20 samples is enough to compare prompts in a controlled way but too small for strong statistical claims. The correlations use only 8 data points.
- **Probabilistic outputs.** Even at temperature 0.1, hosted models are not fully deterministic, and providers may update models. Re-running Step 1 will give similar but not identical outputs.
- **Structured prompt example label.** The JSON example in the structured prompt shows `"positive"` as the sample value. This may bias the model towards that label on neutral inputs and should be considered when interpreting the structured-output results.
- **Scope.** Only zero-shot prompting was tested. Chain-of-Thought, RAG and automatic prompt optimisation were reviewed in the literature but not run.

---

## Academic use

This repository was produced for the HIT401 unit at Charles Darwin University. Please cite our report if you reuse the code or data.
