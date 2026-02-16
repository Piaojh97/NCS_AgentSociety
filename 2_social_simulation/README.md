# 2. Social Simulation

Simulates the emergence of opinion dynamics and polarization among LLM agents on a social network. Agents discuss a topic over multiple epochs; their standpoints (e.g. negative, neutral, positive) and the network structure can be configured.

## Requirements

- Python 3.8+
- numpy, scipy, scikit-learn, matplotlib, seaborn
- openai (for the LLM API)

Recommended versions are documented in the literature; the code has been tested with Python 3.8.11 and compatible library versions.

## Setup

Edit **`utils.py`** and set:

1. Your **OpenAI API key**
2. The **LLM model** to use for the simulation (e.g. `gpt-3.5-turbo`)

You can also adjust experimental settings in the code (e.g. `datasource` for the network file, `num_epoch`, `starting_epoch`, `side_init`, output path).

## How to run

From this directory:

```bash
python run.py
```

For large networks (e.g. thousands of edges and agents), a full run can take several hours depending on the model and API tier.
