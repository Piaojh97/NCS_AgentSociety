# Due to the current limitations of the simulator's support, only NoneBlock, MessageBlock, and FindPersonBlock are available in the Dispatcher.

import random
from typing import Any, Optional


from agentsociety.agent import AgentToolbox, Block, BlockParams
from agentsociety.memory import Memory


class EnvSocialBlockParams(BlockParams):
    ...


class EnvSocialBlock(Block):
    """
    Orchestrates social interactions by dispatching to appropriate sub-blocks.
    """
    ParamsType = EnvSocialBlockParams
    name = "SocialBlock"
    description = "Orchestrates social interactions by dispatching to appropriate sub-blocks."
    actions = {
        "find_person": "Support the find person action, determine the social target.",
        "message": "Support the message action, send a message to the social target.",
        "social_none": "Support other social operations",
    }

    def __init__(
            self, 
            toolbox: AgentToolbox, 
            agent_memory: Memory, 
            block_params: Optional[EnvSocialBlockParams] = None
        ):
        super().__init__(toolbox=toolbox, agent_memory=agent_memory, block_params=block_params)

    async def forward(
        self, step: dict[str, Any], plan_context: Optional[dict] = None
    ) -> dict[str, Any]:
        """Main entry point for social interactions. Dispatches to sub-blocks based on context.

        Args:
            step: Workflow step containing intention and metadata.
            plan_context: Additional execution context.

        Returns:
            Result dict from the executed sub-block.
        """
        node_id = await self.memory.stream.add(
            topic="social", description=f"I {step['intention']}"
        )
        return {
            "success": True,
            "evaluation": f'Finished {step["intention"]}',
            "consumed_time": random.randint(1, 120),
            "node_id": node_id,
        }
