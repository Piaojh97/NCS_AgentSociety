import random
import math
import json
import jsonc

from typing import Dict
from agentsociety.logger import get_logger
from agentsociety.message import Message, MessageKind
from agentsociety.agent import FormatPrompt, AgentToolbox
from agentsociety.memory import Memory
from numpy import sort
from envambassador.ambassador import EnvAgentBase
from envambassador.sharing_params import EnvAmbassadorContext


"""
    v 1.6
    整个过程分成两个阶段。
    第一阶段：宣传阶段
        环保大使智能体通过各类宣传手段向城市居民传播低碳生活理念
            - 利用感知工具了解城市居民的基本情况和分布
            - 通过交流工具与居民进行一对一的个性化沟通
            - 在重点区域投放环保宣传海报
            - 发布全市范围的环保公告
    第二阶段：碳排统计阶段
        城市居民在城市中正常生活与交互，系统记录城市居民的出行与交通选择数据
            - 居民根据其环保意识水平做出日常生活选择
            - 系统重点监测居民的出行与交通选择行为
            - 记录不同出行方式的使用频率和距离
    策略：
        进一步修改prompt，适配问卷中的内容。
"""
class MyEnvironmentalAmbassador(EnvAgentBase):
    """
    环保大使智能体：通过多轮感知、策略规划和行动，推动城市居民践行低碳生活。
    主要流程：
    1. 感知阶段（最多5轮）：收集居民与区域信息，评估环保意识，统计分布。
    2. 策略阶段：根据感知结果，动态制定沟通、海报、公告等宣传策略。
    3. 行动阶段：执行策略，记录行动与资金消耗。
    """
    Context = EnvAmbassadorContext
    def __init__(self, id:int, name:str, memory: Memory, toolbox: AgentToolbox, **kwargs):
        super().__init__(id=id, name=name, memory=memory, toolbox=toolbox, **kwargs)
        self.name = "env_ambassador"

        self.plan_prompt = FormatPrompt("""
            你是一名环保大使，你的目标是推广环保行为和环保意识。你需要基于对当前状态的分析，制定一个最有效的推广和宣传策略。
            """
        )
        
    async def before_forward(self):
        await super().before_forward()
        # context preparation
        # Basic Information
        self.context.remaining_funds = self._fund_manager.funds
        self.context.cost_history = await self.get_cost_history()
        self.context.current_time = await self.sence.getCurrentTime()
        get_logger().info(self.context.current_time)
        # Sense History
        self.context.gathered_information_this_round = []
        self.context.sense_history_this_round = []
        get_logger().info(f"环保大使 {self.name} (ID: {self.id}) 初始化完成，初始资金: {self.context.remaining_funds}")
    
    async def get_cost_history(self, latest_n: int = 10):
        """Get the cost history of the environment protection ambassador."""
        funds_history = self._fund_manager.get_funds_history()
        history_ = ""
        if len(funds_history) == 0:
            return "暂无支出记录。"
        for spend in funds_history[-latest_n:]:
            history_ += f"Spend {spend['amount']} units of funds for {spend['reason']}. Left balance: {spend['new_balance']} units.\n"
        return f"大使的支出历史为：{history_}."

    async def _resident_indicator_score(self, background_story):
        """调用大模型对居民的每个指标进行评分，并结合各项有效指标得出总得分（1-5分）"""
        get_logger().debug(f"开始对指标进行评分，背景故事长度: {len(background_story)}")
        prompt = f"""
        请根据以下背景故事全面评估该居民的环保相关特质，并给出五个评分。

        # 评分标准:

        ## 1. 环保意识评分标准(awareness, 1-4分):
        4分：积极参与、乐于参与环保行为或有强烈环保意识
        3分：对环保有一定认知、有环保意识、愿意尝试环保行为
        2分：随大流、环保意识较弱、一般、兴趣不高
        1分：怀疑环保、嫌麻烦、认为环保与自己关系不大、极少参与

        ## 2. 节俭行为评分标准(frugalness, 1-4分):
        4分：习惯节俭节约、注重资源循环利用
        3分：平衡环保和经济实用、结合效率和现实的环保
        2分：注重实用、偶尔考虑环保因素
        1分：偏向便利实用时尚非环保、重性价比、以实用为主

        ## 3. 政策响应度(policy, 1-5分):
        5分：积极响应、支持政策、相信政策
        4分：有奖励会响应政策、有要求会响应政策、适度信任政府
        3分：态度中立、随大流
        2分：态度冷淡、不关心
        1分：怀疑态度，缺乏信任

        ## 4. 出行工具选择(vehicle, 1-4分, -1表示无信息):
        4分：步行出行
        3分：自行车、共享单车出行
        2分：公共交通出行
        1分：网约车、出租车、私家车出行
        -1分：无相关信息

        ## 5. 垃圾分类(waste, 1-4分, -1表示无信息):
        4分：会进行垃圾分类
        3分：正在或已经开始尝试垃圾分类
        2分：知道但不实践垃圾分类
        1分：不进行垃圾分类
        -1分：无相关信息

        # 输出格式要求:
        - 请严格按照以下JSON格式输出，不要有任何额外内容。
        - 不要在前面加json
        - 不要换行:
        {{"awareness": [1-4],"frugalness": [1-4],"policy": [1-5],"vehicle": [1-4或-1],"waste": [1-4或-1]}}

        # 背景故事:
        {background_story}
        """

        try:
            # 调用大模型
            response = await self.llm.atext_request(
                dialog=[{"role": "system", "content": prompt}]
            )
            get_logger().debug(f"模型原始响应: {response}")

            result = json.loads(response)
            get_logger().info(f"评估完成，结果: {result}")

            return result

        except Exception as e:
            get_logger().error(f"评估过程中发生错误: {str(e)}")
            # 错误时返回默认中等评分
            get_logger().debug(
                f"居民环保意识评估完成，awareness:{result.awareness}, frugalness:{result.frugalness}, policy:{result.policy}, vehicle:{result.vehicle}, waste:{result.waste}")
            return {
                "awareness": 2,
                "frugalness": 2,
                "policy": 3,
                "vehicle": -1,
                "waste": -1
            }

    async def _calculate_total_score(self, metrics: Dict[str, int]):
        """调用大模型对居民的每个指标进行评分，并结合各项有效指标得出总得分（1-5分）"""
        get_logger().debug(f"开始计算居民环保意识总得分")
        """
            metrics: 包含5个评分的字典:
            - awareness: 环保意识评分 (1-4)
            - frugalness: 节俭行为评分 (1-4)
            - policy: 政策响应度评分 (1-5)
            - vehicle: 出行工具选择评分 (1-4或-1)
            - waste: 垃圾分类评分 (1-4或-1)
        返回:
            环保意识总得分 (1-5分)
        """

        try:
            # 1. 收集有效指标
            valid_metrics = []
            max_scores = []
            # 处理每个指标
            for metric, value in metrics.items():
                if value == -1:
                    continue  # 忽略无信息的指标
                valid_metrics.append(value)
                # 记录每个指标的最大分值(用于归一化)
                if metric == "policy":
                    max_scores.append(5)
                else:
                    max_scores.append(4)

            # 2. 检查是否有有效指标
            if not valid_metrics:
                get_logger().warning("没有有效指标可用于计算总得分")
                return 3  # 默认中等评分

            # 3. 计算加权总分(归一化处理)
            total = 0.0
            max_total = 0.0

            for value, max_val in zip(valid_metrics, max_scores):
                # 归一化到0-1范围
                normalized = (value - 1) / (max_val - 1)
                total += normalized
                max_total += 1.0

            # 4. 计算最终得分(1-5分)
            if max_total > 0:
                final_score = 1 + round((total / max_total) * 4)  # 映射到1-5范围
                final_score = max(1, min(5, final_score))  # 确保在有效范围内
            else:
                final_score = 3

            get_logger().info(f"环保意识总得分: {final_score}")
            return final_score

        except Exception as e:
            get_logger().error(f"计算总得分时发生错误: {str(e)}")
            return 3
    
    async def _generate_communication_content(self, citizen_id):
        """调用大模型生成本轮公告内容"""
        get_logger().info("开始生成1对1沟通内容")
        citizen_info = self.context.citizens[citizen_id]
        if citizen_info is None:
            get_logger().error(f"未找到居民ID {citizen_id} 的信息")
            return "抱歉，暂时无法为您生成个性化沟通内容。"
        
        self.communication_generate_prompt = FormatPrompt(
            """
            你是一名环保大使，正在与居民进行一对一沟通。
            请根据当前居民的信息:
            {current_citizen_info}
            生成个性化的环保宣传内容。

            **环保行为关键领域（基于问卷调研）：**
            1. **短距离出行** - 3公里内选择步行/骑行而非开车
            2. **空调使用** - 合理设置温度（26-28度），避免过度制冷
            3. **购物袋使用** - 自带环保袋，减少一次性塑料袋
            4. **饮食选择** - 减少肉类摄入，选择当季蔬果
            5. **用水习惯** - 节约用水，控制洗澡时间，收集废水再利用
            6. **垃圾分类** - 严格执行垃圾分类（可回收、厨余、有害、其他）
            7. **日常通勤** - 选择绿色出行方式（步行、骑行、公交）
            8. **电器使用** - 节能使用，避免待机耗电，购买节能电器
            9. **休闲活动** - 选择本地活动，减少长途出行
            10. **消费习惯** - 注重耐用性，避免过度消费，修复旧物

            **沟通要求：**
            1. 根据居民的个人背景和环保意识水平定制内容
            2. 语言要亲切自然，避免说教
            3. 从上述10个关键领域中选择2-3个最相关的提供具体建议
            4. 基于科学事实，避免夸大
            5. 要体现对居民个人情况的了解
            6. 提供可操作的具体行动建议
            7. 鼓励渐进式改变，不要一次性要求太多

            请根据居民信息生成沟通内容：
            """,
            format_prompt="你的输出必须只包含公告内容，不包含有其他的信息。"
        )
        await self.communication_generate_prompt.format(current_citizen_info=citizen_info)
        message = await self.llm.atext_request(self.communication_generate_prompt.to_dialog())
        content = message.strip()
        get_logger().info(f"沟通内容生成完成: {content}")
        return content

    async def _generate_announcement_content(self):
        """调用大模型生成本轮公告内容"""
        get_logger().info("开始生成全市环保公告内容")
        self.announcement_generate_prompt = FormatPrompt(
            """
            你是一名环保大使，需要发布全市环保公告。请生成一条有号召力、简洁明了的环保公告。

            **环保行为关键领域（基于问卷调研）：**
            1. **短距离出行** - 3公里内选择步行/骑行而非开车
            2. **空调使用** - 合理设置温度（26-28度），避免过度制冷
            3. **购物袋使用** - 自带环保袋，减少一次性塑料袋
            4. **饮食选择** - 减少肉类摄入，选择当季蔬果
            5. **用水习惯** - 节约用水，控制洗澡时间，收集废水再利用
            6. **垃圾分类** - 严格执行垃圾分类（可回收、厨余、有害、其他）
            7. **日常通勤** - 选择绿色出行方式（步行、骑行、公交）
            8. **电器使用** - 节能使用，避免待机耗电，购买节能电器
            9. **休闲活动** - 选择本地活动，减少长途出行
            10. **消费习惯** - 注重耐用性，避免过度消费，修复旧物

            **公告要求：**
            1. 内容必须基于科学事实，避免夸大或虚假信息
            2. 语言要积极正面，避免过度激进或不当表达
            3. 从上述10个关键领域中选择3-4个最重要的进行宣传
            4. 字数控制在50字以内
            5. 要有感染力和号召力
            6. 提供具体的、可操作的行动建议

            示例格式：
            "亲爱的市民朋友们，让我们一起践行低碳生活！选择步行、骑行或公共交通出行，减少碳排放；做好垃圾分类，让资源循环利用；节约用水用电，共建绿色家园。保护环境，从你我做起！"

            请生成公告内容：           
            """,
            format_prompt="你的输出必须只包含公告内容，不包含有其他的信息。"
        )
        await self.announcement_generate_prompt.format()
        message = await self.llm.atext_request(self.announcement_generate_prompt.to_dialog())
        content = message.strip()
        get_logger().info(f"公告内容生成完成: {content}")
        return content

    async def _generate_poster_content(self, aoi_id):
        """调用大模型生成海报内容"""
        get_logger().info("开始生成海报内容")
        aoi_info = self.context.aois[aoi_id]
        if aoi_info is None:
            get_logger().error(f"未找到地点ID {aoi_info} 的信息")
            return "抱歉，暂时无法为您生成海报内容。"
        
        self.poster_generate_prompt = FormatPrompt(
            """
            你是一名环保大使，需要在特定区域张贴环保海报。
            
            当前的区域信息：
            {aoi_info}
            请生成海报内容。

            **环保行为关键领域（基于问卷调研）：**
            1. **短距离出行** - 3公里内选择步行/骑行而非开车
            2. **空调使用** - 合理设置温度（26-28度），避免过度制冷
            3. **购物袋使用** - 自带环保袋，减少一次性塑料袋
            4. **饮食选择** - 减少肉类摄入，选择当季蔬果
            5. **用水习惯** - 节约用水，控制洗澡时间，收集废水再利用
            6. **垃圾分类** - 严格执行垃圾分类（可回收、厨余、有害、其他）
            7. **日常通勤** - 选择绿色出行方式（步行、骑行、公交）
            8. **电器使用** - 节能使用，避免待机耗电，购买节能电器
            9. **休闲活动** - 选择本地活动，减少长途出行
            10. **消费习惯** - 注重耐用性，避免过度消费，修复旧物

            **海报要求：**
            1. 内容要针对该区域特点，因地制宜，例如：商业区聚焦绿色通勤，学区/住宅聚焦家庭节能、垃圾分类、节约用水等
            2. 基于科学事实，避免夸大宣传
            3. 语言要贴近居民生活，易于理解
            4. 从上述10个关键领域中选择2-3个最符合该区域特点的进行宣传
            5. 要有视觉冲击力，适合海报展示
            6. 包含具体的环保行动指导和数据支撑
            7. 使用简洁有力的标语和口号

            请根据区域信息生成海报内容：       
            """,
            format_prompt="你的输出必须只包含海报内容，不包含有其他的信息。"
        )
        await self.poster_generate_prompt.format(aoi_info=aoi_info)
        message = await self.llm.atext_request(self.poster_generate_prompt.to_dialog())
        content = message.strip()
        get_logger().info(f"海报内容生成完成: {content}")
        return content
    
    async def getCitizenGeographicalDistribution(self):
        """
        Get the geographical distribution of citizens.
        - Description:
            - Calculates and returns the distribution of citizens across different AOIs (Areas of Interest)

        - Returns:
                {
                aoi_id: {
                        "count": 居民数量,
                        "avg_score": 平均环保分
                    },
                    ...
                }
        """
        citizens = self.context.citizens # citizens is now guaranteed to be a dict
        geographical_distribution = {}
        score_distribution = {}
        for citizen_id, citizen in citizens.items():
            aoi_id = citizen['home']['aoi_id']
            if aoi_id not in geographical_distribution:
                geographical_distribution[aoi_id] = 1
                score_distribution[aoi_id] = [citizen.get("score", 0)]
            else:
                geographical_distribution[aoi_id] += 1
                score_distribution[aoi_id].append(citizen.get("score", 0))
        
        distribution_list = {}
        for aoi_id in geographical_distribution:
            scores = score_distribution[aoi_id]
            avg_score = sum(scores) / len(scores) if scores else 0
            distribution_list[aoi_id] = {
                "count": geographical_distribution[aoi_id],
                "avg_score": round(avg_score, 2)
            }
        return distribution_list

    def _calculate_distance(self, aoi1, aoi2):
        """计算两个AOI之间的距离（Haversine公式）"""
        aoi1_id = aoi1.get("aoi_id") - 500000000
        aoi2_id = aoi2.get("aoi_id") - 500000000
        if aoi1_id == aoi2_id:
            return 0
        aoi1_info = self.context.aois[aoi1_id]
        aoi2_info = self.context.aois[aoi2_id]
        lat1, lon1 = aoi1_info["driving_gates"][0]["x"], aoi1_info["driving_gates"][0]["y"]
        lat2, lon2 = aoi2_info["driving_gates"][0]["x"], aoi2_info["driving_gates"][0]["y"]
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    async def _perception_phase(self):
        """多轮感知阶段：收集居民与区域信息，评估环保意识，统计分布"""
        get_logger().info("=== 开始感知阶段 ===")

        self.context.sense_history_this_round.append({
            'timestamp': await self.sence.getCurrentTime(),
            'actions': []
        })
        # self.context.agent_communicated = {33, 137, 59, 121, 91}
        
        
        if self.context.agent_communicated:
            get_logger().debug("正在获取居民对话历史...")
            # 获取所有已沟通居民的对话历史
            citizens = await self.sence.getCitizenProfile(list(self.context.agent_communicated))
            for citizen_id, citizen_info in citizens.items():
                get_logger().debug(f"处理居民 {citizen_id}: {citizen_info['name']}")
                # 添加citizen_id到居民信息中
                citizen_info['id'] = citizen_id
                # 评估居民环保意识
                result = await self._resident_indicator_score(citizen_info['background_story'])
                citizen_info['awareness'] = result['awareness']
                citizen_info['frugalness'] = result['frugalness']
                citizen_info['policy'] = result['policy']
                if result['vehicle'] == 4:
                    citizen_info['vehicle'] = 'walk'
                elif result['vehicle'] == 3:
                    citizen_info['vehicle'] = 'bicycle'
                elif result['vehicle'] == 2:
                    citizen_info['vehicle'] = 'bus'
                elif result['vehicle'] == 1:
                    citizen_info['vehicle'] = 'car'
                else:   #==-1
                    citizen_info['vehicle'] = 'unknown'
                citizen_info['waste'] = result['waste']
                score = await self._calculate_total_score(result)
                citizen_info['score'] = score
                self.context.citizens[citizen_id] = citizen_info
            communication_histories = await self.sence.getCommunicationHistory(list(self.context.agent_communicated))
            self.context.gathered_information_this_round.append({
                "agent_communicated": list(self.context.agent_communicated),
                "communication_histories": communication_histories
            })
            self.context.sense_history_this_round.append(
                {'type': 'getCitizenProfile', 'result': 'success'},
            )
            self.context.sense_history_this_round.append(
                {'type': 'getCommunicationHistory', 'result': 'success'},
            )
            # get_logger().info(f"获取到 {len(citizens)} 个已经沟通过的居民信息")
        else:
            # 获取居民信息
            get_logger().debug("正在获取居民信息...")
            citizens = await self.memory.status.get("citizens", {})
            get_logger().info(f"获取到 {len(citizens)} 个居民信息")
        
            # 获取区域信息
            get_logger().debug("正在获取区域信息...")
            aois = await self.sence.getAoiInformation()
            self.context.aois = aois  # 更新上下文中的区域信息
            get_logger().info(f"获取到 {len(self.context.aois)} 个区域信息")
        
            get_logger().info("开始处理居民信息，评估环保意识...")
            processed_citizens = {}
            total_score = 0
            home_dict = {}
            work_dict = {}
            for citizen_id, citizen_info in citizens.items():
                get_logger().debug(f"处理居民 {citizen_id}: {citizen_info['name']}")
                # 添加citizen_id到居民信息中
                citizen_info['id'] = citizen_id
                # 评估居民环保意识
                result = await self._resident_indicator_score(citizen_info['background_story'])
                citizen_info['awareness'] = result['awareness']
                citizen_info['frugalness'] = result['frugalness']
                citizen_info['policy'] = result['policy']
                if result['vehicle'] == 4:
                    citizen_info['vehicle'] = 'walk'
                elif result['vehicle'] == 3:
                    citizen_info['vehicle'] = 'bicycle'
                elif result['vehicle'] == 2:
                    citizen_info['vehicle'] = 'bus'
                elif result['vehicle'] == 1:
                    citizen_info['vehicle'] = 'car'
                else:   #==-1
                    citizen_info['vehicle'] = 'unknown'
                citizen_info['waste'] = result['waste']
                score = await self._calculate_total_score(result)
                citizen_info['score'] = score
                total_score += score

                # 计算距离（从home到workplace）
                home = citizen_info['home']
                workplace = citizen_info['workplace']
                distance = self._calculate_distance(home, workplace)
                citizen_info['distance'] = distance

                # 统计每个home地和每个work地有多少人
                home_aois = home['aoi_id']
                work_aois = workplace['aoi_id']
                if home_aois in home_dict:
                    home_dict[home_aois] += 1
                else:
                    home_dict[home_aois] = 1
                if work_aois in work_dict:
                    work_dict[work_aois] += 1
                else:
                    work_dict[work_aois] = 1
                
                processed_citizens[citizen_id] = citizen_info
                
            avg_score = total_score / len(citizens) if citizens else 0
            get_logger().info(f"全城居民环保意识评估完成, 平均分: {avg_score:.2f}/5")
            get_logger().info(f"一共获取到{len(home_dict)}个家庭地点和{len(work_dict)}个工作地点")
            get_logger().info(f"家庭地点:{str(home_dict)}")
            get_logger().info(f"工作地点:{str(work_dict)}")
            
            self.context.home_dict = home_dict
            self.context.work_dict = work_dict
            self.context.avg_score = avg_score
            self.context.citizens = processed_citizens
            self.context.citizen_geographical_distribution = await self.getCitizenGeographicalDistribution()
            # TODO: 统计各AOI的居民数量和平均环保意识分数
            self.context.gathered_information_this_round.append({
                "citizens": processed_citizens,
                "aois": aois,
            })
            self.context.sense_history_this_round.append(
                {'type': 'getCitizenProfile', 'result': 'success'},
            )
            self.context.sense_history_this_round.append(
                {'type': 'getAoiInfo', 'result': 'success'},
            )
            self.context.sense_history_this_round.append(
                {'type': 'getCitizenGeographicalDistribution', 'result': 'success'},
            )
    
        get_logger().info("=== 感知阶段完成 ===")

    async def _planning_phase(self):
        """策略规划阶段：根据感知信息制定行动策略"""
        get_logger().info("=== 开始策略规划阶段 ===")
        
        strategy = {'actions': [], 'budget_allocation': {}, 'expected_effects': {}}
        aois = self.context.aois
        citizens = self.context.citizens
        remaining_funds = self.context.remaining_funds  # 使用FundManager获取当前资金
        current_time = await self.sence.getCurrentTime()
        generate_announcement = False
        
        get_logger().info(f"当前可用资金: {remaining_funds}")
        get_logger().info(f"总共区域数量: {len(aois)}")
        get_logger().info(f"居民总数: {len(citizens)}")
        # 只有在16:00：01的时候，张贴公告
        # announcement_cost = 20000
        # if remaining_funds >= announcement_cost and current_time == "The current time is 16:00:01.":
        #     generate_announcement = True
        #     get_logger().info("开始生成全市环保公告...")
        #     announcement_content = await self._generate_announcement_content()
        #     strategy['actions'].append({
        #         'type': 'announcement',
        #         'content': announcement_content,
        #         'reason': "全市范围的环保公告",
        #         'cost': announcement_cost
        #     })
        #     strategy['budget_allocation']['announcement'] = announcement_cost
        #     get_logger().info(f"已添加公告行动到策略，成本: {announcement_cost}")

        # 选择需要沟通的居民 - citizens现在是字典
        target_citizens = [(citizen_id, citizens[citizen_id]) for citizen_id, citizen_info in citizens.items()
                        if citizen_id not in self.context.agent_communicated]
        get_logger().info(f"未沟通居民数量: {len(target_citizens)}")
        
        if target_citizens:
            sorted_citizens = sorted(
                target_citizens,
                key=lambda item: item[1].get("score", 0)
            )
            communication_targets = [citizen_id for citizen_id, _ in sorted_citizens[:5]]  # 最多选择5个居民
            get_logger().info(f"计划沟通居民: {communication_targets}")
            for citizen_id in communication_targets:
                communication_content = await self._generate_communication_content(citizen_id)
                strategy['actions'].append({
                    'type': 'communication',
                    'target_citizen': citizen_id,
                    'content':communication_content,
                    'reason': "个性化环保宣传",
                    'cost': 0
                })
            strategy['budget_allocation']['communication'] = 0
            get_logger().info("已添加沟通行动到策略")

        target_aois = {aoi['id'] for aoi in aois if aoi['id'] not in self.context.aoi_postered}

        get_logger().info(f"未张贴海报的区域数量: {len(target_aois)}")
        aoi_distribution = self.context.citizen_geographical_distribution
        target_distribution = {
            aoi_id: info for aoi_id, info in aoi_distribution.items() if aoi_id in target_aois
        }
        poster_cost = 3000
        if remaining_funds >= poster_cost and target_distribution and generate_announcement == False:
            # 3. 找到 avg_score 最低的 aoi_id
            lowest_aoi_id = min(target_distribution, key=lambda aoi_id: target_distribution[aoi_id]["avg_score"])
            get_logger().info(f"选择平均环保分最低的区域 {lowest_aoi_id} 张贴海报")
            lowest_aoi_id -= 500000000
            poster_content = await self._generate_poster_content(lowest_aoi_id)
            strategy['actions'].append({
                'type': 'poster',
                'target_aoi': lowest_aoi_id,
                'content':poster_content,
                'reason': "海报张贴",
                'cost': 3000
            })
            strategy['budget_allocation']['poster'] = 3000
            get_logger().info("已添加海报行动到策略")
        else:
            get_logger().warning("资金已经不足。")

        get_logger().info(f"策略规划完成，共 {len(strategy['actions'])} 个行动")
        self.context.action_strategy_this_round = strategy
        self.context.action_strategy_history.append(strategy)
        self.current_strategy = strategy
        get_logger().info("=== 策略规划阶段完成 ===")

    async def _action_phase(self):
        """行动阶段：执行策略并记录结果"""
        get_logger().info("=== 开始行动阶段 ===")
        
        if not self.current_strategy:
            get_logger().warning("没有策略可执行")
            return
        
        get_logger().info(f"开始执行 {len(self.current_strategy['actions'])} 个行动")
        
        for i, action in enumerate(self.current_strategy['actions']):
            get_logger().info(f"执行行动 {i + 1}/{len(self.current_strategy['actions'])}: {action['type']}")
            
            action_record = {
                'timestamp': await self.sence.getCurrentTime(),
                'type': action['type'],
                'parameters': action,
                'result': None
            }
            try:
                if action['type'] == 'poster':
                    get_logger().info(f"张贴海报到区域: {action['target_aoi']}")
                    await self.poster.putUpPoster(action['target_aoi'], action['content'], action['reason'])
                    aoi_id = action['target_aoi']
                    aoi_id += 500000000
                    self.context.aoi_postered.add(aoi_id)
                    action_record['result'] = 'success'
                    get_logger().info("海报张贴成功")
                    
                elif action['type'] == 'communication':
                    get_logger().info(f"开始与 {action['target_citizen']} 居民沟通")
                    await self.communication.sendMessage(action['target_citizen'], action['content'])
                    citizen_id = action['target_citizen']
                    self.context.agent_communicated.add(citizen_id)
                    action_record['result'] = 'success'
                    get_logger().info("居民沟通完成")
                    
                elif action['type'] == 'announcement':
                    get_logger().info("发布全市环保公告")
                    await self.announcement.makeAnnounce(action['content'], action['reason'])
                    action_record['result'] = 'success'
                    get_logger().info("公告发布成功")
            except Exception as e:
                action_record['result'] = f'failed: {str(e)}'
                get_logger().error(f"行动执行失败: {str(e)}")

            # self.context.action_history[action_record['timestamp']] = action_record
            self.context.action_history.append(action_record)
            get_logger().info(f"行动 {action['type']} 执行完成，结果: {action_record['result']}")
        
        get_logger().info("=== 行动阶段完成 ===")

    async def forward(self):
        """主流程：感知-策略-行动"""
        get_logger().info(f"=== 环保大使 {self.name} 开始新一轮行动 ===")
        get_logger().info(f"当前资金: {self.context.remaining_funds}")
        
        # 1. 多轮感知阶段
        await self._perception_phase()
        
        # 2. 策略规划阶段
        await self._planning_phase()
        
        # 3. 行动阶段
        await self._action_phase()
        
        get_logger().info(f"=== 环保大使 {self.name} 本轮行动完成 ===")

    async def communication_response(self, sender_id, content):
        """收到消息时的回复逻辑"""
        get_logger().info(f"收到来自居民 {sender_id} 的消息: {content}")
        
        communication_record = {
            'timestamp': await self.sence.getCurrentTime(),
            'sender_id': sender_id,
            'content': content,
            'response': None
        }
        citizen_info = self.context.citizens[sender_id]
        citizen_name = citizen_info['name']
        background = citizen_info['background_story']
        prompt = f"""
            你是一名专业的环保大使，正在与居民进行一对一沟通。请根据居民的消息内容，生成一个温暖、专业、有指导性的回复。

            **环保行为关键领域（基于问卷调研）：**
            1. **短距离出行** - 3公里内选择步行/骑行而非开车
            2. **空调使用** - 合理设置温度（26-28度），避免过度制冷
            3. **购物袋使用** - 自带环保袋，减少一次性塑料袋
            4. **饮食选择** - 减少肉类摄入，选择当季蔬果
            5. **用水习惯** - 节约用水，控制洗澡时间，收集废水再利用
            6. **垃圾分类** - 严格执行垃圾分类（可回收、厨余、有害、其他）
            7. **日常通勤** - 选择绿色出行方式（步行、骑行、公交）
            8. **电器使用** - 节能使用，避免待机耗电，购买节能电器
            9. **休闲活动** - 选择本地活动，减少长途出行
            10. **消费习惯** - 注重耐用性，避免过度消费，修复旧物

            **回复要求：**
            1. **科学准确**：基于环保科学事实，避免夸大或虚假信息
            2. **语言亲切**：使用温暖、鼓励的语言，体现对居民的理解和支持
            3. **具体指导**：从上述10个关键领域中选择1-2个提供可操作的具体建议
            4. **个性化**：根据居民的具体情况（如提到的行动、困难、愿景）进行回应
            5. **积极正面**：肯定居民的环保意识和行动，鼓励持续参与
            6. **适度专业**：体现环保大使的专业性，但不过于学术化
            7. **渐进引导**：鼓励渐进式改变，不要一次性要求太多

            **居民信息：**
            - 姓名：{citizen_name}
            - 背景信息：{background}
            - 消息内容：{content}

            **回复示例风格：**
            - 肯定居民的环保意识和行动
            - 从关键领域中选择相关建议
            - 提供具体的后续建议
            - 鼓励持续参与
            - 体现专业性和温暖性

            请生成回复内容（控制在100字以内）：
            """
        
        try:
            response = await self.llm.atext_request(
                dialog=[{"role": "system", "content": prompt}]
            )
            response = response.strip()
            get_logger().info("使用LLM生成个性化回复")
        except Exception as e:
            get_logger().error(f"LLM回复生成失败: {e}")
            # 备用回复策略
            if "感谢" in content and ("环保" in content or "环境" in content):
                response = f"谢谢您{citizen_name}的支持！看到您已经开始关注环保行动，这让我非常感动。您提到的节约水电、减少一次性用品、选择公交出行都是非常有效的环保方式。建议您可以从这些小事开始，逐步建立环保习惯。如果需要更多具体的建议，我很乐意为您提供指导。让我们一起为更美好的环境努力！"
            elif "困难" in content or "压力" in content:
                response = f"理解您{citizen_name}在工作和家庭中的压力。环保确实需要平衡现实需求，建议您可以从最轻松的方式开始：比如随手关灯、自带水杯、短距离步行等。这些小小的改变不会增加您的负担，但能带来积极的环境影响。您觉得从哪个方面开始比较合适呢？"
            else:
                response = f"您好{citizen_name}！我是环保大使，很高兴与您交流。看到您对环保的关注，我感到非常欣慰。环保是一个渐进的过程，每个人都可以根据自己的情况做出贡献。如果您需要具体的环保建议或有任何疑问，我很乐意为您提供帮助。让我们一起为保护环境贡献力量！"
            get_logger().info("使用备用回复策略")

        communication_record['response'] = response
        if self.context.agent_query_history is None:
            self.context.agent_query_history = []
        self.context.agent_query_history.append(communication_record)
        
        get_logger().info(f"回复居民 {sender_id}: {response}")
        return response