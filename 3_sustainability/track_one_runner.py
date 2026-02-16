import os
import importlib.util
import json
from typing import Dict, Any, Type

from agentsociety.configs import (
    Config,
    LLMConfig,
    EnvConfig,
    MapConfig,
    AgentsConfig,
    AgentConfig,
)
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety

from envambassador import (
    TRACK_ONE_EXPERIMENT,
    TrackOneEnvCitizen,
    BaselineEnvAmbassador,
)


def load_agent_from_file(file_path: str) -> Type:
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ValueError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"Module loader is None for {file_path}")

    spec.loader.exec_module(module)

    for item_name in dir(module):
        item = getattr(module, item_name)
        if isinstance(item, type) and "EnvAgentBase" in str(item.__bases__):
            return item
    raise ValueError(f"在文件 {file_path} 中未找到 EnvAgentBase 子类")


def load_params_from_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        params = json.load(file)
    return params


async def run(file_path: str, is_json: bool):
    """
    Run the agent simulation with the given file.

    Args:
        file_path: Path to the input file (Python or JSON)
        is_json: Whether the input file is JSON (True) or Python (False)
        profile_path: Optional path to the profile file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    # Default configurations (API Key via env LLM_API_KEY, base_url via LLM_BASE_URL)
    llm_configs = [
        LLMConfig(
            provider=LLMProviderType.VLLM,
            base_url=os.getenv("LLM_BASE_URL", "https://your-llm-endpoint/v1"),
            api_key=os.getenv("LLM_API_KEY", "YOUR_API_KEY"),
            model=os.getenv("LLM_MODEL", "qwen2.5-14b-instruct"),
            concurrency=200,
            timeout=30,
        ),
    ]

    env_config = EnvConfig.model_validate(
        {
           "db": {
                "enabled": True,
                "db_type": "sqlite",
            },
        }
    )

    map_config = MapConfig(
        file_path="./data/beijing.pb",
    )

    if is_json:
        params = load_params_from_file(file_path)
        config = Config(
            llm=llm_configs,
            env=env_config,
            map=map_config,
            agents=AgentsConfig(
                citizens=[
                    AgentConfig(
                        agent_class=TrackOneEnvCitizen,
                        memory_from_file="./data/profile.json",
                    ),
                    AgentConfig(
                        agent_class=BaselineEnvAmbassador,
                        number=1,
                        agent_params=params,
                    ),
                ],
                firms=[],
                banks=[],
                nbs=[],
                governments=[],
                others=[],
                supervisor=None,
                init_funcs=[],
            ),
            exp=TRACK_ONE_EXPERIMENT,
        )
    else:
        agent_class = load_agent_from_file(file_path)
        config = Config(
            llm=llm_configs,
            env=env_config,
            map=map_config,
            agents=AgentsConfig(
                citizens=[
                    AgentConfig(
                        agent_class=TrackOneEnvCitizen,
                        memory_from_file="./data/profile.json",
                    ),
                    AgentConfig(
                        agent_class=agent_class,
                        number=1,
                    ),
                ],
                firms=[],
                banks=[],
                nbs=[],
                governments=[],
                others=[],
                supervisor=None,
                init_funcs=[],
            ),
            exp=TRACK_ONE_EXPERIMENT,
        )

    agentsociety = AgentSociety(config)
    await agentsociety.init()
    await agentsociety.run()

    # Results
    survey_score = agentsociety.context["survey_result"]["final_score"]
    carbon_emission_score = agentsociety.context["carbon_emission_result"][
        "final_score"
    ]
    promotion_score = agentsociety.context["promotion_result"]["final_score"]
    overall_score = agentsociety.context["overall_score"]
    print(
        f"Survey score: {survey_score}, Carbon emission score: {carbon_emission_score}, Promotion score: {promotion_score}, Overall score: {overall_score}"
    )
    await agentsociety.close()
    info = {
        "survey_score": survey_score,
        "carbon_emission_score": carbon_emission_score,
        "promotion_score": promotion_score,
        "overall_score": overall_score,
    }

    return overall_score, info


if __name__ == "__main__":
    load_agent_from_file("./download.py")
