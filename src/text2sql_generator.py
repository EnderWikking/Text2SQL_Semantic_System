import json
import os
from openai import OpenAI

# 1. 初始化千问大模型
client = OpenAI(
    api_key="sk-90a75b57806f4794bdbb96273df856a3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def link_schema_by_literal(question, index_path):
    """
    核心黑科技：通过用户的自然语言提问，去字典里反查它命中了哪些数据库列。
    """
    print("🔍 [Schema Linking] 正在通过字面量字典匹配相关列...")
    with open(index_path, "r", encoding="utf-8") as f:
        literal_index = json.load(f)

    matched_columns = set()
    # 遍历字典，如果某个字面量（长度>3防噪音）出现在问题中，就把对应的列加进来
    for val_str, columns in literal_index.items():
        if len(val_str) > 3 and val_str.lower() in question.lower():
            print(f"  🎯 命中字面量: '{val_str}' -> 可能的列: {columns}")
            matched_columns.update(columns)

    return list(matched_columns)


def generate_sql(question, matched_columns):
    """
    将用户问题和精准筛选后的列信息喂给大模型，生成最终的 SQL。
    """
    print("\n🧠 正在召唤大模型生成 SQL...\n")

    # 构建 Schema 提示词（这里为了演示，我们假设这些是 enhancer 提炼出的语义）
    schema_context = f"用户问题命中了以下数据库列：{matched_columns}。\n"
    schema_context += "请只使用这些列和 california_schools 数据库中的表（frpm, schools, satscores）来编写 SQL。"

    system_prompt = (
        "你是一个顶级的 Text-to-SQL 专家。请根据用户的问题和提供的数据库列信息，"
        "编写出完全正确、可以在 SQLite 中运行的 SQL 查询语句。"
        "请直接输出 SQL，不需要任何多余的解释，并用 ```sql 和 ``` 包裹。"
    )

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"上下文信息：\n{schema_context}\n\n用户提问：{question}",
            },
        ],
        timeout=30,
    )

    sql_output = response.choices[0].message.content
    print("-" * 50)
    print("✨ 生成的终极 SQL 如下：")
    print(sql_output)
    print("-" * 50)


if __name__ == "__main__":
    index_file = os.path.join("data", "literal_index.json")

    # 导师可能会考你的一个经典的复杂 BIRD 问题：
    user_question = "请帮我查一下，属于 Los Angeles Unified 学区（District Name），并且提供了 9-12 年级教育（GSoffered）的学校有多少所？"

    print(f"👤 用户提问: {user_question}\n")

    if os.path.exists(index_file):
        # 1. 第一步：模式链接（Schema Linking）
        relevant_cols = link_schema_by_literal(user_question, index_file)

        # 2. 第二步：生成 SQL
        if relevant_cols:
            generate_sql(user_question, relevant_cols)
        else:
            print(
                "❌ 没有从字典中匹配到相关列，可能需要引入向量检索（FAISS）作为补充。"
            )
    else:
        print(f"❌ 找不到字典文件: {index_file}")
