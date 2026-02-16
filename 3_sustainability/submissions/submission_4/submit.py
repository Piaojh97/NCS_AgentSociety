from envambassador.ambassador import EnvAgentBase


class MyEnvAmbassador(EnvAgentBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.budget = 100000  # 初始预算
        self.message_count = 0  # 消息计数器

    async def communication_response(self, sender_id, content):
        """处理市民消息回复"""
        # 使用正确的工具名称 self.sence
        profile = await self.sence.getCitizenProfile([sender_id])
        citizen_profile = profile.get(sender_id, {})
        age = citizen_profile.get('age', 40)  # 默认40岁

        if "汽车" in content:
            return "建议您尝试公共交通，北京地铁覆盖率达95%，既环保又快捷！"
        elif age < 30:
            return "年轻人是环保主力军！试试共享单车出行吧~"
        else:
            return "感谢您关注环保！选择绿色出行能为下一代创造更好环境。"

    async def forward(self):
        """智能体主动行为逻辑"""
        # 使用正确的工具名称 self.sence
        current_time = await self.sence.getCurrentTime()

        # 策略1：早高峰在交通枢纽发公告
        if "08:00" <= current_time <= "10:00":
            if self.budget >= 20000:
                # 获取所有AOI信息并筛选交通枢纽
                aoi_info = await self.sence.getAoIInformation()
                transport_aois = [aoi_id for aoi_id, info in aoi_info.items()
                                  if "交通枢纽" in info.get('type', '')]
                if transport_aois:
                    await self.announcement.makeAnnounce(
                        content="早高峰选择地铁出行，碳排放仅为私家车的1/4！",
                        reason="高峰时段最大范围宣传"
                    )
                    self.budget -= 20000

        # 策略2：午间在商业区贴海报
        elif "12:00" <= current_time <= "14:00":
            aoi_info = await self.sence.getAoIInformation()
            commercial_aois = [aoi_id for aoi_id, info in aoi_info.items()
                               if "商业区" in info.get('type', '')]
            if commercial_aois and self.budget >= 3000:
                # 最多选择3个商业区
                target_aois = commercial_aois[:3]
                await self.poster.putUpPoster(
                    target_aoi_ids=target_aois,
                    content="绿色消费：自带杯具享折扣，减少一次性用品",
                    reason="商业区环保宣传"
                )
                self.budget -= 3000 * len(target_aois)

        # 策略3：定向消息（限5次/轮）
        if self.message_count < 5:
            citizens = await self.sence.getCitizenProfile()
            # 寻找有车的市民
            for cid, info in citizens.items():
                if info.get('car_owner', False):
                    await self.communication.sendMessage(
                        citizen_ids=cid,
                        content="您作为车主，尝试每周少开一天车可减少10%碳排哦！"
                    )
                    self.message_count += 1
                    if self.message_count >= 5:
                        break