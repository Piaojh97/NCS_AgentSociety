# 1. Economic Simulation

LLM-empowered agents simulate macroeconomic activities: each agent decides how much to work and how much to consume in each time step. The environment is based on the Foundation (AI Economist) framework.

## Requirements

- Python 3.x
- OpenAI API key

## Setup

Set your OpenAI API key in **`simulate_utils.py`** (e.g. the variable used for the OpenAI client).

## How to run

**GPT policy** — agents use an LLM to make decisions (example: 100 agents, 240 months):

```bash
python simulate.py --policy_model gpt --num_agents 100 --episode_length 240
```

**Composite policy** — uses a hand-crafted composite rule instead of the LLM:

```bash
python simulate.py --policy_model complex --num_agents 100 --episode_length 240
```

You can change `--num_agents` and `--episode_length` as needed.

## Note

If you use a model other than the one originally tested (e.g. `gpt-4o-mini` instead of `gpt-3.5-turbo`), and see many invalid decisions (high `gpt_error`), try adjusting the prompt in the code, especially the part that asks for JSON with `work` and `consumption` in the range [0, 1] in steps of 0.02.
