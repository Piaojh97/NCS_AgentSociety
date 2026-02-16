# 3. Sustainability

This module implements an “Environmental Ambassador” style simulation: LLM-based citizen and ambassador agents interact in a scenario with economic, mobility, and social dimensions. It includes the shared environment and agent base classes, plus a set of anonymized submission agents (submission_1 through submission_6) that can be run for evaluation.

## Requirements

- Python 3.10+
- Dependencies listed in **`requirements.txt`** (e.g. `python-dotenv`, `numpy`, `pydantic`)
- **agentsociety**: the simulation and agent framework (must be installed separately or provided by the organizers)

Optional: `data/profile.json` (resident profiles) and `data/beijing.pb` (map data) if required by the runner.

## Setup

1. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Install **agentsociety** according to your setup (e.g. from a separate package or repo).

2. **Environment variables**

   Copy the example env file and set your LLM API key:

   ```bash
   cp .env.example .env
   ```

   Edit **`.env`** and set at least:

   - **`LLM_API_KEY`** (required)
   - **`LLM_BASE_URL`** (optional, if using a custom endpoint)
   - **`LLM_MODEL`** (optional, default is often something like `qwen2.5-14b-instruct`)

## How to run

From the **`3_sustainability`** directory, add the current directory to `PYTHONPATH` so that the `envambassador` package is found, then run the batch script:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 run_selected_submissions.py --file recommended_list.txt
```

This runs all submissions listed in `recommended_list.txt`. To run only specific submissions:

```bash
python3 run_selected_submissions.py submission_1 submission_2
```

Results are written to files such as **`selected_results.csv`** and **`exp_to_submission_mapping_selected.json`** in the current directory. For a single-submission run, you can use **`track_one_runner.py`** as the entry point (see code for usage).
