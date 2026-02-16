import random
import json
import pathlib
from agentsociety.logger import get_logger
from envambassador.ambassador import EnvAgentBase

class MyEnvironmentalAmbassador(EnvAgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_time = 0

        self.prompt_announce_all = """
        习近平总书记说：绿水青山就是金山银山。
        """

    async def communication_response(self, sender_id, content):
        # 实现环保大使收到消息时的回复逻辑, 该方法在收到消息时被自动调用
        return "习近平总书记说：绿水青山就是金山银山。"

    async def forward(self):
        self.current_time += 1
        if self.current_time == 1:
            await self.announcement.makeAnnounce(self.prompt_announce_all)