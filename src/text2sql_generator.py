import json
import os
import numpy as np
from provider_clients import (
    get_chat_client,
    get_chat_model,
    get_embedding_client,
    get_embedding_model,
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


def link_schema_by_vector(question, vector_path="data/vector_index.npz", top_k=5):
    """
    核心黑科技 2.0：密集检索（Dense Retrieval）。
    将自然语言问题转为向量，通过余弦相似度召回最贴切的数据库列。
    解决了纯字面量匹配中“同义词、缩写无法命中”的痛点。
    """
    if not os.path.exists(vector_path):
        return []
        
    print("🧠 [Schema Linking] 正在通过大模型向量空间进行语义余弦相似度(Dense)匹配...")
    
    try:
        # 1. 在线将用户问题向量化
        response = get_embedding_client().embeddings.create(
            model=get_embedding_model(),
            input=[question]
        )
        query_vector = np.array(response.data[0].embedding, dtype=np.float32)
        
        # 2. 毫秒级加载本地 Numpy 高维知识库
        data = np.load(vector_path, allow_pickle=True)
        embeddings_matrix = data["embeddings"]
        keys = data["keys"]
        
        # 3. 矩阵点乘，极速计算所有列与提问的余弦相似度
        norm_q = np.linalg.norm(query_vector)
        norms_e = np.linalg.norm(embeddings_matrix, axis=1)
        similarities = np.dot(embeddings_matrix, query_vector) / (norm_q * norms_e + 1e-10)
        
        # 4. 获取 Top-K 最相关的物理列
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matched_columns = []
        for idx in top_indices:
            col_key = keys[idx]
            score = similarities[idx]
            # 引入置信度阈值机制，防止胡乱召回
            if score > 0.35:
                print(f"  🌌 命中相似高维向量: '{col_key}' (相似度: {score:.3f})")
                parts = col_key.split(".")
                if len(parts) >= 2:
                    matched_columns.append(f"{parts[-2]}.{parts[-1]}")
                    
        return matched_columns
    except Exception as e:
        print(f"❌ 向量召回失败: {e}")
        return []


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

    response = get_chat_client().chat.completions.create(
        model=get_chat_model(),
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
        # 1. 传统架构：精确匹配保下限
        exact_cols = link_schema_by_literal(user_question, index_file)

        # 2. 现代架构：向量召回提上限 (Hybrid Retrieval)
        vector_index_file = os.path.join("data", "vector_index.npz")
        vector_cols = link_schema_by_vector(user_question, vector_index_file)
        
        # 3. 双路融合，合并作为最终提示词 Context
        final_cols = list(set(exact_cols + vector_cols))

        if final_cols:
            generate_sql(user_question, final_cols)
        else:
            print("❌ 无论是字典还是向量均未命中列，请检查模型知识库情况。")
    else:
        print(f"❌ 找不到字典文件: {index_file}")
