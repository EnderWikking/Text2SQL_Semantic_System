import sqlite3
import os
import pandas as pd
from openai import OpenAI

# 1. 初始化千问大模型客户端
client = OpenAI(
    api_key="sk-90a75b57806f4794bdbb96273df856a3",  # ⚠️ 请务必替换为你的真实 Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_table_profile(db_path, table_name):
    """提取指定表的物理画像，并拼装成文本"""
    print(f"🔍 正在提取物理画像: {table_name} 表...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    profile_lines = [f"表名: {table_name}", f"总行数: {len(df)}"]

    # 为了演示效果和节省 token，我们先只取前 5 列进行画像
    for col in df.columns[:5]:
        null_count = df[col].isnull().sum()
        distinct_count = df[col].nunique()
        top_samples = df[col].value_counts().head(3).index.tolist()

        col_profile = (
            f"\n字段: {col}\n"
            f" - 空值: {null_count}\n"
            f" - 唯一值: {distinct_count}\n"
            f" - 高频样本: {top_samples}"
        )
        profile_lines.append(col_profile)

    return "\n".join(profile_lines)


def stream_semantic_enhancement(table_profile):
    """调用大模型，使用流式输出生成语义描述"""
    print("\n🧠 正在召唤 AI 专家进行语义翻译（流式输出）...\n")
    print("-" * 50)

    system_prompt = (
        "你是一个顶级的数据库架构师。我正在构建一个 Text2SQL 系统。"
        "请根据我提供的数据库物理画像，推测每个字段的具体业务含义。"
        "输出格式要求：请为每个字段提供一段'Short Description'（简短总结），"
        "说明它的业务用途、格式，以及大模型写 SQL WHERE 条件时必须注意的数据类型约束。"
    )

    try:
        # 开启 stream=True，实现打字机效果
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"这是提取到的物理画像：\n{table_profile}"},
            ],
            stream=True,  # 🌟 关键：开启流式输出
            timeout=30,
        )

        # 实时打印出来的字符
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n" + "-" * 50)
        print("✅ 语义增强解析完成！")

    except Exception as e:
        print(f"\n❌ 请求失败: {e}")


if __name__ == "__main__":
    # 指向咱们的加州学校数据库
    db_file = os.path.join(
        "data", "dev_databases", "california_schools", "california_schools.sqlite"
    )

    if os.path.exists(db_file):
        # 1. 自动提取画像文本
        profile_text = get_table_profile(db_file, "frpm")

        # 2. 流式输入给大模型
        stream_semantic_enhancement(profile_text)
    else:
        print(f"❌ 找不到文件: {db_file}")
