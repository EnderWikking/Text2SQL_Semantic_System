import sqlite3
import pandas as pd
import json
import os
import time
from openai import OpenAI

# 强制使用兼容模式
client = OpenAI(
    api_key="sk-90a75b57806f4794bdbb96273df856a3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_db_profile(db_path):
    """提取单个数据库的完整物理画像，控制采样防止内存溢出"""
    conn = sqlite3.connect(db_path)
    tables_df = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )

    db_profile = {}
    for table in tables_df["name"]:
        try:
            # 工业级防崩：如果表太大，绝不全量读取，只采样子集获取 Schema
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 50000", conn)
            table_info = {"row_count": len(df), "columns": {}}

            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                distinct_count = int(df[col].nunique())
                # 提取高频样本，转换为字符串防止 JSON 序列化报错
                top_samples = [
                    str(x) for x in df[col].value_counts().head(5).index.tolist()
                ]

                table_info["columns"][col] = {
                    "nulls": null_count,
                    "distinct": distinct_count,
                    "samples": top_samples,
                    "type": str(df[col].dtype),
                }
            db_profile[table] = table_info
        except Exception as e:
            print(f"  ⚠️ 警告：跳过表 {table}，原因: {e}")

    conn.close()
    return db_profile


def enhance_semantics_with_retry(db_name, table_name, table_profile, max_retries=3):
    """带重试机制的 LLM API 调用，专治网络波动和限流"""
    prompt = (
        f"作为数据库专家，请分析数据库 {db_name} 中的表 {table_name} 的物理画像。\n"
        f"画像数据：{json.dumps(table_profile, ensure_ascii=False)}\n"
        "请推断每个字段的业务含义，并简要说明在 Text2SQL 中写 WHERE 条件时的约束（如是否带引号、保留前导零等）。直接输出精简的描述文本。"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严谨的 Text-to-SQL 数据分析师。",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=45,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    ⏳ API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2**attempt)  # 指数退避策略：1s, 2s, 4s...

    return "LLM_ENHANCEMENT_FAILED"


def run_industrial_pipeline(base_dir):
    """执行全局批量处理，并建立 Checkpoint 机制"""
    output_dir = os.path.join("data", "metadata_cache")
    os.makedirs(output_dir, exist_ok=True)

    # 扫描所有子文件夹（即每个数据库）
    db_folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]
    print(f"🚀 侦测到 {len(db_folders)} 个数据库，开始执行离线流水线...")

    for db_name in db_folders:
        db_path = os.path.join(base_dir, db_name, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            continue

        print(f"\n📁 正在处理数据库: [{db_name}]")

        # 建立断点续传：如果已经存在处理完毕的 JSON，则直接跳过
        cache_file = os.path.join(output_dir, f"{db_name}_enhanced.json")
        if os.path.exists(cache_file):
            print(f"  ⏭️ 检测到缓存，跳过...")
            continue

        # 1. 物理提取
        print("  - 阶段 1: 物理特征全量提取中...")
        raw_profile = get_db_profile(db_path)

        # 2. 语义增强 (分表调用，避免单次 Token 超限)
        enhanced_profile = {}
        for table, info in raw_profile.items():
            print(f"  - 阶段 2: 正在请求 LLM 增强表 {table} ...")
            enhanced_desc = enhance_semantics_with_retry(db_name, table, info)
            enhanced_profile[table] = {
                "physical_stats": info,
                "semantic_description": enhanced_desc,
            }
            time.sleep(1)  # 保护 API 额度，防止高并发限流

        # 3. 落地落盘
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_profile, f, ensure_ascii=False, indent=2)
        print(f"  ✅ [{db_name}] 处理完毕，已持久化至 {cache_file}")


if __name__ == "__main__":
    dev_path = os.path.join("data", "dev_databases")
    run_industrial_pipeline(dev_path)
