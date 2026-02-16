# 3. Sustainability

Install dependencies: `pip install -r requirements.txt` (and **agentsociety** if required).

Copy `.env.example` to `.env` and set `LLM_API_KEY`.

From this directory, set Python path and run:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 run_selected_submissions.py --file recommended_list.txt
```

To run specific submissions: `python3 run_selected_submissions.py submission_1 submission_2`.
