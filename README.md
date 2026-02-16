# NCS_AgentSociety

Open-source code for the Perspective article **"Generating complex societies via large language model agents"** submitted to **Nature Computational Science**.

This repository contains three simulation modules that use large language model (LLM) agents to simulate different aspects of complex societies. Each module can be run independently; see the README in each subfolder for setup and run instructions.

---

## Repository structure

| Folder | Description |
|--------|-------------|
| **1_economic_simulation** | Macroeconomic simulation: LLM agents make work and consumption decisions over time, building on the AI Economist / Foundation framework. |
| **2_social_simulation** | Opinion dynamics and polarization: LLM agents interact on a social network and their standpoints evolve over multiple conversation epochs. |
| **3_sustainability** | Environmental ambassador track: LLM-based citizen and ambassador agents in a sustainability scenario, with economy, mobility, and social blocks. |

---

## Quick start

- **1_economic_simulation**: Set your OpenAI API key in `simulate_utils.py`, then run  
  `python simulate.py --policy_model gpt --num_agents 100 --episode_length 240`
- **2_social_simulation**: Set API key and model in `utils.py`, then run  
  `python run.py`
- **3_sustainability**: Install dependencies, copy `.env.example` to `.env` and set `LLM_API_KEY`, then from the module directory run  
  `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` and  
  `python3 run_selected_submissions.py --file recommended_list.txt`

For detailed requirements and options, see each subfolder’s README.
