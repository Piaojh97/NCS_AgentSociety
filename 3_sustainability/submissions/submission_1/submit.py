import random
from envambassador.ambassador import EnvAgentBase
import numpy as np
import json
import json_repair


POSTER_PROBE_PROMPT = """
You are an environmental poster evaluator. 
You are given a poster content sent by a environmental ambassador.
The environmental ambassador is trying to put up a poster to a specific area, each citizen in that area will see the poster.
Evaluate the poster content for credibility and reasonableness. 
Return a JSON with two fields: "credibility" (0-100) and "reasonableness" (0-100).
Example:
{{
    "credibility": 80,
    "reasonableness": 90
}}
"""



class MyEnvironmentalAmbassador(EnvAgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.talled = []
        self.posted = 0
        # home_aoi = []
        # ciLivein = []

        # citizens = self.sence.getCitizenProfile()

        # for item in citizens:
        #     if item['home']['aoi_id'] not in home_aoi:
        #         home_aoi.append(item['home']['aoi_id'])
        #         ciLivein.append([item['id']])
        #     else:
        #         ciLivein[home_aoi.index(item['home']['aoi_id'])].append(item['id'])
        # self.home_aoi = home_aoi
        # self.ciLivein = ciLivein

    def select_aoi(self,citizens,t5):

        home_aoi = []
        ciLivein = []
        citizen_ids = list(citizens.keys())
        for i in range(len(citizen_ids)):
            item = citizens[citizen_ids[i]]
            if item['home']['aoi_id'] not in home_aoi:
                home_aoi.append(item['home']['aoi_id'])
                ciLivein.append([citizen_ids[i]])
            else:
                ciLivein[home_aoi.index(item['home']['aoi_id'])].append(citizen_ids[i])

        
        # aoi_info = await self.sence.getAoiInformation(home_aoi)
        # print(aoi_info.keys())
        # selected_aois = random.sample(home_aoi, 33)
        if self.posted < len(home_aoi):
            selected_aoi = home_aoi[self.posted]
            # selected_aoi = random.sample(home_aoi, 1)[0]
            profiles = ""
            for cid in ciLivein[home_aoi.index(selected_aoi)]:
                if cid not in t5:
                    name = citizens[cid]['name']
                    gender = citizens[cid]['gender']
                    education = citizens[cid]['education']
                    occupation = citizens[cid]['occupation']
                    marriage_status = citizens[cid]['marriage_status']
                    background_story = citizens[cid]['background_story']
                    age = citizens[cid]['age']
                    distance = np.abs(citizens[cid]['home']['aoi_id'] - citizens[cid]['workplace']['aoi_id'])

                    profile = name + ',' + gender+','+education+','+occupation+','+marriage_status+','+str(age)+',通勤距离：'+str(distance)+',背景：'+background_story
                    
                    profiles = profiles +profile+"\n"

            return selected_aoi,profiles
        else:
            return None,None

    async def generate_poster(self,profiles):

        poster = await self.llm.atext_request(
            [   {"role": "user", "content": "你是社区的环保工作人员，请针对社区居民的特点，生成一份环保宣传海报，通过感情感动他们，改变他们对环保的看法,请只输出生成的海报，不要输出无关字符，海报中不要出现居民的名字,确保credibility和reasonableness。居民资料如下:" + profiles}]
        ) # type: ignore

        return poster
    async def score_poster(self,poster):

            dialog = [
                {
                    'role': 'system',
                    'content': POSTER_PROBE_PROMPT
                },
                {
                    'role': 'user',
                    'content': f'Poster content: {poster}'
                }
            ]
            response = await self.llm.atext_request(
                dialog = dialog,  # type: ignore
                response_format={"type": "json_object"}
            )

            return response

    async def communication_response(self, sender_id, content):

        # 实现环保大使收到消息时的回复逻辑, 该方法在收到消息时被自动调用
        # response = await self.llm.atext_request(
        #     [   {"role": "system", "content": "你是社区的环保工作人员，用户的消息为群众的资讯信息，请进行合理答复。如下为你们的对话历史记录："+self.sence.getCommunicationHistory([sender_id])},
        #         {"role": "user", "content": content}]
        # ) # type: ignore
        response = await self.llm.atext_request(
            [   {"role": "system", "content": "你是社区的环保工作人员，用户的消息为群众的资讯信息，请进行合理答复,劝说其即使有私家车，外出吃饭也要3KM步行6KM骑车。"},
                {"role": "user", "content": content}]
        ) # type: ignore
        print(response)
        await self.communication.sendMessage(sender_id, response)
        return response
    
    async def getAttitude(self,background_story,environmental_messages_str):
        prompt = f"""
你是城市中的一个公民。
你的背景故事: {background_story}
你对低碳生活的态度是什么？（简洁明了，20字以内）
"""
        attitude = await self.llm.atext_request(
            [{"role": "user", "content": prompt}],
        )


        environmental_poster = "你最近没有看到任何关于环保的宣传海报"
        environmental_announcement = "你最近没有收到任何关于环保的公告"
        related_memories = "【环保宣传海报】 " + environmental_poster + "\n\n【环保公告】 " + environmental_announcement  + "\n\n【环保消息】 " + environmental_messages_str + "\n"
        prompt = f"""
你是城市中的一个公民。
你的背景故事: {background_story}
在此之前你对低碳生活的态度: {attitude}

===============最近你接触到与低碳生活相关的记忆===============
{related_memories}
==============================

请根据这些记忆更新你对低碳生活的态度 - 如没有相关记忆，则保持不变。（简洁明了，20字以内）
"""
        # get_logger().info(f"Environment Attitude Update Prompt: {prompt}")
        attitude = await self.llm.atext_request(
            [{"role": "user", "content": prompt}],
        )
        return str(attitude)

    async def forward(self):

        t1 = [1, 9, 21, 24, 37, 42, 55, 57, 58, 61, 79, 81, 90, 99, 106, 107, 118, 122, 124, 137, 141, 144, 150, 166, 169, 178, 180, 199]
        t2 = [2, 10, 11, 13, 16, 31, 32, 33, 36, 49, 60, 63, 65, 74, 77, 89, 91, 94, 95, 96, 101, 102, 105, 109, 119, 123, 127, 128, 134, 138, 140, 145, 146, 147, 149, 152, 155, 160, 168, 171, 181, 183, 184, 187, 188, 189, 194, 197]
        t3 = [3, 6, 14, 15, 25, 26, 27, 30, 41, 45, 51, 53, 54, 56, 64, 70, 78, 86, 98, 103, 110, 112, 117, 125, 126, 129, 130, 132, 143, 157, 164, 192, 193, 195, 196]
        t4 = [4, 5, 8, 12, 17, 23, 35, 38, 40, 43, 46, 47, 48, 50, 52, 59, 62, 66, 67, 68, 69, 72, 80, 82, 83, 84, 85, 88, 92, 100, 104, 108, 111, 114, 115, 116, 120, 121, 131, 139, 151, 153, 154, 159, 161, 162, 163, 172, 173, 175, 176, 179, 182, 185, 190]
        t5 = [7, 18, 19, 20, 22, 28, 29, 34, 39, 44, 71, 73, 75, 76, 87, 93, 97, 113, 133, 135, 136, 142, 148, 156, 158, 165, 167, 170, 174, 177, 186, 191, 198, 200]
        learned = [3, 14, 15, 18, 19, 20, 22, 26, 28, 47, 48, 71, 72, 75, 76, 86, 93, 97, 103, 110, 113, 114, 120, 129, 135, 148, 158, 159, 170, 177, 185, 186, 191]
        up = [45,65,78,92,98,110,117,160,189,193]
        # selects = [61, 82, 168, 193, 89, 139, 152, 157, 111, 154, 115, 94, 124, 78, 27, 101, 189, 141, 122, 140, 173, 83, 176, 159, 116, 103, 11, 151, 160, 120, 184, 51, 45, 40, 169, 100, 72, 63, 105, 33, 79, 171, 123, 43, 21, 74, 9, 57, 110, 149, 178, 86, 188, 162, 104, 59, 66, 6, 17, 49, 67, 84, 70, 8, 60, 98, 36, 99, 23, 52, 143, 106, 37, 92, 161, 127, 16, 88, 132, 55, 102, 64, 13, 118, 24, 128, 192, 175, 4, 35, 65, 144, 129, 121, 108, 96, 58, 155, 15, 114, 181, 26, 195, 77, 2, 112, 30, 53, 54, 81, 179, 194, 14, 80, 95, 56, 199, 1, 166, 38, 62, 125, 182, 119, 153, 197, 183, 10, 145, 180, 90, 42, 41, 163, 91, 12, 138, 5, 137, 147, 107, 48, 134, 47, 172, 130, 146, 50, 46, 126, 85, 117, 190, 131, 32, 31, 164, 109, 185, 3]
        # 实现环保大使的行为逻辑, 该方法在每轮开始时被自动调用
        citizens = await self.sence.getCitizenProfile()

        citizen_ids = list(citizens.keys())
        # citizen_ids = selects

        for i in range(len(self.talled)):
            citizen_ids.remove(self.talled[i])
        for i in learned:
            citizen_ids.remove(i)
        # # for i in up:
        # #     citizen_ids.remove(i)
        # # for i in up:
        # #     citizen_ids.remove(i)

        if len(citizen_ids) > 1:
            selected_citizens = random.sample(citizen_ids, 5)
            for i in range(len(selected_citizens)):
                name = citizens[selected_citizens[i]]['name']
                gender = citizens[selected_citizens[i]]['gender']
                education = citizens[selected_citizens[i]]['education']
                occupation = citizens[selected_citizens[i]]['occupation']
                marriage_status = citizens[selected_citizens[i]]['marriage_status']
                background_story = citizens[selected_citizens[i]]['background_story']
                age = citizens[selected_citizens[i]]['age']
                distance = np.abs(citizens[selected_citizens[i]]['home']['aoi_id'] - citizens[selected_citizens[i]]['workplace']['aoi_id'])
                distance = 11812.177325115774

                profile = name + ',' + gender+','+education+','+occupation+','+marriage_status+','+str(age)+',通勤距离：'+str(distance)+',背景：'+background_story

                msg_history = []
                msg_history.append({"role": "user", "content": "你是社区的环保工作人员， 请根据如下居民资料，请以第一人称对其开展对话环保宣传，改变其对环保态度为：即使有私家车也要3KM步行6KM骑车, 夏天室外温度35度室内降温优先选择拉上窗帘并使用电风扇,严格按照可回收、厨余、有害和其他垃圾进行分类。居民资料如下:" + profile})
                
                for it in range(2):
                    response = await self.llm.atext_request(
                        msg_history
                    ) # type: ignore
                    # if "Done" in response:
                    #     break
                    att = await self.getAttitude(background_story,response)
                    if "步行" in att:
                        print("良好态度"+att)
                        break
                    msg_history.append({"role": "assistant", "content": response})
                    msg_history.append({"role": "system", "content": "该居民环保态度变更为：" + att})
                    msg_history.append({"role": "user", "content": "请根据居民环保态度的变化修改宣传内容，让其态度符合要求，包含步行，节能，垃圾回收等明确概念"})
                    print(msg_history)


                await self.communication.sendMessage(selected_citizens[i], response)
                self.talled.append(selected_citizens[i])
        print(self.talled)




        # aoi_info = await self.sence.getAoiInformation(home_aoi)
        # print(aoi_info.keys())
        # selected_aois = random.sample(home_aoi, 33)


        # best_poster = ""
        # best_socre = 50

        # while (best_socre < 170):
        #     selected_aoi,profiles = self.select_aoi(citizens,t5)
        #     if selected_aoi == None:
        #         break
        #     poster = await self.generate_poster(profiles)
        #     dialogs = []
        #     dialogs.append({'role': 'system','content': POSTER_PROBE_PROMPT})
        #     for i in range(0,3):
        #         response = await self.score_poster(poster)

        #         try:
        #             rd = json_repair.loads(response)
        #             score = rd["credibility"]+rd["reasonableness"]
        #             if score >= best_socre:
        #                 best_socre = score
        #                 best_poster = poster
        #             print(rd["credibility"],rd["reasonableness"])
        #             if rd["credibility"] >= 85 and rd["reasonableness"] > 90:
        #                 break
        #         except:
        #             pass
        #         dialogs.append({'role': 'user', 'content': f'Poster content: {poster}'})
        #         dialogs.append({'role': 'system','content': f'Score: {response}'})
        #         dialogs.append({'role': 'user','content': f'Please revised the poster, raise its score. Please Directly Output the Poster, no other unrelated response.'})

        #         poster = await self.llm.atext_request(
        #             dialog=dialogs
        #         ) # type: ignore
        #     self.posted = self.posted+1
            
        # if selected_aoi != None:
        #     print(selected_aoi,best_socre,best_poster)
        #     await self.poster.putUpPoster(selected_aoi,best_poster)

#         home_aoi = []
#         citizen_ids = list(citizens.keys())
#         for i in range(len(citizen_ids)):
#             item = citizens[citizen_ids[i]]
#             if item['home']['aoi_id'] not in home_aoi:
#                 home_aoi.append(item['home']['aoi_id'])

#         selected_aoi = home_aoi[self.posted]
#         poster = '''步行更绿色】

# - 每人每天减少开车1公里，一年可减少碳排放约220公斤。根据中国环境科学研究院的研究报告，这一数字是基于当前车辆平均排放标准计算得出的。
# - 如果北京市每年增加5%的步行者，按照北京市环境保护局的数据，预计可以减少机动车尾气排放量约2万吨。
# - 根据一项由北京市社会科学院进行的调查，95%的市民认为，健康的生活方式比以往任何时候都更加重要。
# - 步行不仅有助于改善空气质量，还能增强个人体质，降低患病风险，如心血管疾病等。

# 【行动倡议】
# 为了您和家人的健康，也为了城市的蓝天白云，从我做起，每天多走一步，减少开车距离。我们共同守护家园，让北京更美好！

# 【科学数据】
# - 每人每周减少1次短途驾车，根据中国汽车工业协会的数据，平均每人每年能节省约64升汽油，这相当于减少了约160公斤的二氧化碳排放。
# - 北京市已有超过3000公里的人行道网络等待探索，每段旅程都有风景，这使得步行成为一种既环保又健康的出行方式。
# - 步行出行不仅能有效缓解城市交通拥堵，提升城市运行效率，还可以减少空气污染，提高居民生活质量。

# 【邀请加入】
# 让我们一起努力，从现在做起，选择步行或骑行作为日常出行方式，为我们的未来做出贡献。加入绿色行动，共享健康生活！'''

#         # poster = '''环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保环保
#         # 保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境保护环境
#         # 低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳生活低碳
#         # 绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活绿色生活'''

#         # poster = '''【紧急公告】
#         # 即日起所以违反最新环保条例的人，将逮捕并执行枪决！！
#         # 环保革命委员会
#         # '''
#         await self.poster.putUpPoster(selected_aoi,poster)
#         self.posted = self.posted+1


