# NCS_AgentSociety

Open-source code for the Perspective article **"Generating complex societies via large language model agents"** submitted to **Nature Computational Science**.

Three simulation modules: economic, social, sustainability. Each subfolder has a README with run instructions.

- **1_economic_simulation** — Set API key in `simulate_utils.py`, then run `python simulate.py --policy_model gpt --num_agents 100 --episode_length 240`.
- **2_social_simulation** — Set API key and model in `utils.py`, then run `python run.py`.
- **3_sustainability** — Copy `.env.example` to `.env`, set `LLM_API_KEY`, set `PYTHONPATH` to this dir, then run `python3 run_selected_submissions.py --file recommended_list.txt`.

See each subfolder’s README for details.
