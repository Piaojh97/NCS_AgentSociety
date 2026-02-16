#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行指定的提交并记录exp_id和submit_file的映射关系

使用方法:
    # 方式1: 命令行参数指定提交列表
    python3 run_selected_submissions.py submission_1 submission_2

    # 方式2: 使用推荐列表文件（一行一个提交文件夹名）
    python3 run_selected_submissions.py --file recommended_list.txt

    # 方式3: 使用JSON配置文件
    python3 run_selected_submissions.py --json submissions_list.json
"""

import sys
import os
import json
import asyncio
import traceback
import logging
import ast
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# 导入运行器（需要先修改track_one_runner.py来返回exp_id）
import importlib.util
spec = importlib.util.spec_from_file_location("track_one_runner", "./track_one_runner.py")
track_one_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(track_one_runner)

load_dotenv()

# 配置参数
INPUT_DIR = "./submissions"
RESULT_FILE = "selected_results.csv"  # 结果文件路径（包含exp_id）
MAPPING_FILE = "exp_to_submission_mapping_selected.json"  # 映射文件
DOWNLOAD_FILE_PATH = "./tmp_download.py"

async def run_with_exp_id(file_path: str, is_json: bool, submission_folder: str):
    """
    运行提交并返回exp_id和得分
    
    由于AgentSociety在运行时会自动保存到agentsociety_data/exps/<exp_id>/artifacts.json，
    我们需要通过查找最新创建的exp目录来获取exp_id
    """
    
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
    
    # 获取运行前的exp目录列表和时间戳
    exp_dir = Path("agentsociety_data/exps")
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = datetime.now().timestamp()
    
    before_exp_ids = set()
    if exp_dir.exists():
        before_exp_ids = {d.name for d in exp_dir.iterdir() if d.is_dir()}
    
    # 默认配置（API Key 请通过环境变量 LLM_API_KEY 配置，base_url 通过 LLM_BASE_URL 配置）
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
            "home_dir": "./agentsociety_data_new"
        }
    )
    
    map_config = MapConfig(
        file_path="./data/beijing.pb",
    )
    
    # 加载agent
    if is_json:
        params = track_one_runner.load_params_from_file(file_path)
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
        agent_class = track_one_runner.load_agent_from_file(file_path)
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
    
    # 运行仿真
    # 使用 submission_folder 作为 tenant_id 的一部分，确保每次运行都有独立的 exp_id
    tenant_id = f"SUBMISSION_{submission_folder.replace('/', '_')}"
    agentsociety = AgentSociety(config, tenant_id=tenant_id)
    await agentsociety.init()
    await agentsociety.run()
    
    # 获取结果
    survey_score = agentsociety.context["survey_result"]["final_score"]
    carbon_emission_score = agentsociety.context["carbon_emission_result"]["final_score"]
    promotion_score = agentsociety.context["promotion_result"]["final_score"]
    overall_score = agentsociety.context["overall_score"]
    
    await agentsociety.close()
    
    # 等待一下确保文件已写入
    await asyncio.sleep(1)
    
    # 获取新创建的exp_id（通过对比运行前后的目录和时间戳）
    after_exp_ids = set()
    if exp_dir.exists():
        after_exp_ids = {d.name for d in exp_dir.iterdir() if d.is_dir()}
    
    # 找到在运行开始后创建的exp目录
    new_exp_ids = []
    for exp_id in after_exp_ids - before_exp_ids:
        exp_path = exp_dir / exp_id
        if exp_path.exists():
            mtime = exp_path.stat().st_mtime
            if mtime >= run_start_time:
                new_exp_ids.append((exp_id, mtime))
    
    if len(new_exp_ids) == 1:
        exp_id = new_exp_ids[0][0]
    elif len(new_exp_ids) > 1:
        # 如果多个，选择最新的（通过mtime）
        new_exp_ids.sort(key=lambda x: x[1], reverse=True)
        exp_id = new_exp_ids[0][0]
        logging.warning(f"发现多个新的exp_id: {[e[0] for e in new_exp_ids]}，选择最新的: {exp_id}")
    else:
        # 如果没找到新的，尝试使用运行后最新创建的目录
        if after_exp_ids:
            exp_with_mtime = [(eid, (exp_dir / eid).stat().st_mtime) for eid in after_exp_ids]
            exp_with_mtime.sort(key=lambda x: x[1], reverse=True)
            # 使用在运行开始后创建的
            for eid, mtime in exp_with_mtime:
                if mtime >= run_start_time:
                    exp_id = eid
                    logging.warning(f"未找到明确的新exp_id，使用最近创建的: {exp_id}")
                    break
            else:
                exp_id = exp_with_mtime[0][0] if exp_with_mtime else None
                logging.warning(f"使用最新的exp目录: {exp_id}")
        else:
            exp_id = None
            logging.error(f"无法找到exp_id")
    
    info = {
        "survey_score": survey_score,
        "carbon_emission_score": carbon_emission_score,
        "promotion_score": promotion_score,
        "overall_score": overall_score,
    }
    
    return exp_id, overall_score, info


def parse_arguments():
    """解析命令行参数"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    submissions = []
    
    if sys.argv[1] == "--file":
        # 从文件读取（每行一个提交名）
        if len(sys.argv) < 3:
            print("错误: --file 需要指定文件路径")
            sys.exit(1)
        file_path = sys.argv[2]
        with open(file_path, 'r', encoding='utf-8') as f:
            submissions = [line.strip() for line in f if line.strip()]
    elif sys.argv[1] == "--json":
        # 从JSON文件读取
        if len(sys.argv) < 3:
            print("错误: --json 需要指定JSON文件路径")
            sys.exit(1)
        json_path = sys.argv[2]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                submissions = data
            elif isinstance(data, dict) and 'submissions' in data:
                submissions = data['submissions']
    else:
        # 命令行参数直接指定
        submissions = sys.argv[1:]
    
    return submissions


def init_result_file():
    """初始化结果文件"""
    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['提交文件夹名', 'exp_id', '得分', '错误信息', '运行时间'])
        logging.info(f"创建结果文件: {RESULT_FILE}")


def write_result_to_csv(submission_folder: str, exp_id: Optional[str], score: Optional[float], 
                        error_msg: str, run_time: str):
    """将结果写入CSV文件"""
    with open(RESULT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        score_str = str(score) if score is not None else ""
        exp_id_str = exp_id if exp_id else ""
        writer.writerow([submission_folder, exp_id_str, score_str, error_msg, run_time])
    logging.info(f"结果已写入CSV: {submission_folder}, exp_id: {exp_id_str}, 得分: {score_str}")


def save_mapping(submission_folder: str, exp_id: Optional[str], score: Optional[float], 
                 run_time: str, error_msg: str = ""):
    """保存exp_id和submission_folder的映射到JSON文件"""
    mapping_file = Path(MAPPING_FILE)
    
    # 读取现有映射
    mappings = []
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
    
    # 添加新映射
    mapping_entry = {
        "submission_folder": submission_folder,
        "exp_id": exp_id,
        "score": score,
        "run_time": run_time,
        "error": error_msg,
        "artifacts_path": f"agentsociety_data/exps/{exp_id}/artifacts.json" if exp_id else None
    }
    
    mappings.append(mapping_entry)
    
    # 保存回文件
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    logging.info(f"映射已保存: {submission_folder} -> {exp_id}")


def cleanup_temp_file():
    """清理临时文件"""
    try:
        if os.path.exists(DOWNLOAD_FILE_PATH):
            os.remove(DOWNLOAD_FILE_PATH)
    except Exception as e:
        logging.warning(f"清理临时文件失败: {str(e)}")


async def process_submission(submission_folder: str):
    """处理单个提交"""
    submission_path = Path(INPUT_DIR) / submission_folder
    
    if not submission_path.exists():
        error_msg = f"提交文件夹不存在: {submission_path}"
        logging.error(error_msg)
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_result_to_csv(submission_folder, None, None, error_msg, run_time)
        save_mapping(submission_folder, None, None, run_time, error_msg)
        return
    
    # 查找submit文件
    submit_py = submission_path / "submit.py"
    submit_json = submission_path / "submit.json"
    
    if submit_json.exists():
        file_path = str(submit_json)
        is_json = True
    elif submit_py.exists():
        file_path = str(submit_py)
        is_json = False
    else:
        error_msg = f"未找到submit.py或submit.json"
        logging.error(error_msg)
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_result_to_csv(submission_folder, None, None, error_msg, run_time)
        save_mapping(submission_folder, None, None, run_time, error_msg)
        return
    
    # 复制到临时位置
    try:
        import shutil
        shutil.copy2(file_path, DOWNLOAD_FILE_PATH)
    except Exception as e:
        error_msg = f"复制文件失败: {str(e)}"
        logging.error(error_msg)
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_result_to_csv(submission_folder, None, None, error_msg, run_time)
        save_mapping(submission_folder, None, None, run_time, error_msg)
        return
    
    # 运行评测
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"开始运行提交: {submission_folder}")
    try:
        # 直接运行，不设置超时限制
        exp_id, score, info = await run_with_exp_id(DOWNLOAD_FILE_PATH, is_json, submission_folder)
        logging.info(f"运行完成: {submission_folder}, exp_id: {exp_id}, 得分: {score}")
        write_result_to_csv(submission_folder, exp_id, score, "", run_time)
        save_mapping(submission_folder, exp_id, score, run_time, "")
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"运行错误: {str(e)}"
        logging.error(f"运行失败: {error_traceback}")
        write_result_to_csv(submission_folder, None, None, error_msg, run_time)
        save_mapping(submission_folder, None, None, run_time, error_msg)
    finally:
        cleanup_temp_file()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(filename)s %(funcName)s:%(lineno)d %(message)s'
    )
    
    # 解析参数
    submissions = parse_arguments()
    
    if not submissions:
        logging.error("未指定要运行的提交")
        sys.exit(1)
    
    logging.info(f"准备运行 {len(submissions)} 个提交: {submissions}")
    
    # 初始化结果文件
    init_result_file()
    
    # 处理每个提交
    for i, submission in enumerate(submissions, 1):
        logging.info(f"处理进度: {i}/{len(submissions)} - {submission}")
        await process_submission(submission)
        logging.info(f"完成: {i}/{len(submissions)} - {submission}")
    
    logging.info("所有提交处理完成")
    print(f"\n结果已保存到:")
    print(f"  - CSV文件: {RESULT_FILE}")
    print(f"  - 映射文件: {MAPPING_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

