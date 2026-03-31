import json
import os
import sqlite3
import time

import pandas as pd
from provider_clients import get_chat_client, get_chat_model


def quote_identifier(name):
    # SQLite 标识符转义，避免关键字/特殊字符导致 SQL 失败
    return '"' + str(name).replace('"', '""') + '"'


def get_db_profile(db_path):
    """提取单个数据库的完整物理画像，控制采样防止内存溢出。"""
    conn = sqlite3.connect(db_path)
    tables_df = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )

    db_profile = {}
    for table in tables_df["name"]:
        try:
            quoted_table = quote_identifier(table)
            # 工业级防崩：如果表太大，绝不全量读取，只采样子集获取 Schema
            df = pd.read_sql_query(f"SELECT * FROM {quoted_table} LIMIT 50000", conn)

            # 提取主外键信息（对 Text2SQL JOIN 至关重要）
            pk_info = pd.read_sql_query(f"PRAGMA table_info({quoted_table})", conn)
            primary_keys = pk_info[pk_info["pk"] > 0]["name"].tolist()

            fk_info = pd.read_sql_query(f"PRAGMA foreign_key_list({quoted_table})", conn)
            foreign_keys = []
            for _, row in fk_info.iterrows():
                foreign_keys.append(
                    {
                        "from_column": row["from"],
                        "to_table": row["table"],
                        "to_column": row["to"],
                    }
                )

            row_count_df = pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM {quoted_table}", conn)
            row_count = int(row_count_df.iloc[0]["cnt"])

            table_info = {
                "row_count": row_count,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "columns": {},
            }

            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                distinct_count = int(df[col].nunique())
                # 提取高频样本，转换为字符串防止 JSON 序列化报错
                top_samples = [str(x) for x in df[col].value_counts().head(5).index.tolist()]

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
    """带重试机制的 LLM API 调用，专治网络波动和限流，强制返回 JSON 结构化元数据。"""
    prompt = (
        f"作为顶级数据库架构师，请深度分析数据库 {db_name} 中 {table_name} 表的物理画像。\n"
        f"画像数据（含空值率、主外键等）：{json.dumps(table_profile, ensure_ascii=False)}\n\n"
        "任务要求：\n"
        "1. 详细推衍每个字段的业务语义，补充在 Text2SQL 中生成 WHERE 或 JOIN 时的具体约束口径。\n"
        "2. 你必须完全返回合法的 JSON 对象。请严格遵守以下 JSON 结构：\n"
        "{\n"
        '  "table_summary": "一句话概括该表的业务核心用途",\n'
        '  "join_hints": "若存在外键关系，详细说明在跨表关联（JOIN）或嵌套查询时的关键注意事项。若无，留空。",\n'
        '  "columns_semantic": {\n'
        '    "这里填列名": {\n'
        '      "description": "详细且准确的业务含义解释",\n'
        '      "sql_format_constraints": "在生成 SQL 时，如何正确处理数值/文本的格式？是否需要加单引号或特定类型转换？"\n'
        "    }\n"
        "  }\n"
        "}"
    )

    for attempt in range(max_retries):
        try:
            response = get_chat_client().chat.completions.create(
                model=get_chat_model(),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个极其严谨的 Text-to-SQL 数据分析专家，你只输出完全合法且无多余说明的 JSON 格式。",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=45,
            )
            result_str = response.choices[0].message.content
            return json.loads(result_str)  # 强制校验并解析为 dict
        except Exception as e:
            print(f"    ⏳ API调用或JSON解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2**attempt)  # 指数退避策略：1s, 2s, 4s...

    return {"error": "LLM_ENHANCEMENT_FAILED"}


def run_industrial_pipeline(base_dir):
    """执行全局批量处理，并建立 Checkpoint 机制。"""
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
            print("  ⏭️ 检测到缓存，跳过...")
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
