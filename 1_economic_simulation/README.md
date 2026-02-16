# 1. Economic Simulation

Set your OpenAI API key in `simulate_utils.py`.

Run with GPT policy (e.g. 100 agents, 240 months):

```bash
python simulate.py --policy_model gpt --num_agents 100 --episode_length 240
```

Run with composite policy:

```bash
python simulate.py --policy_model complex --num_agents 100 --episode_length 240
```
