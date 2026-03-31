import json
import os
import time

import numpy as np
from provider_clients import get_embedding_client, get_embedding_model


def build_vector_index(metadata_dir="data/metadata_cache", output_path="data/vector_index.npz"):
    print("🚀 开始根据【强化版 JSON Metadata】构建高维向量知识库...")

    if not os.path.exists(metadata_dir):
        print(f"❌ 找不到元数据目录: {metadata_dir}")
        return

    keys = []
    texts = []

    # 1. 组装富文本 Context
    for filename in os.listdir(metadata_dir):
        if not filename.endswith("_enhanced.json"):
            continue

        db_name = filename.replace("_enhanced.json", "")
        with open(os.path.join(metadata_dir, filename), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for table_name, table_info in metadata.items():
            semantic_data = table_info.get("semantic_description", {})
            if not isinstance(semantic_data, dict):
                continue

            columns_semantic = semantic_data.get("columns_semantic", {})
            for col_name, col_info in columns_semantic.items():
                desc = col_info.get("description", "")
                fmt = col_info.get("sql_format_constraints", "")

                # 这一段就是浓缩给 Embedding 模型去映射向量的精选文本
                rich_text = f"表名:{table_name} 字段名:{col_name} 核心语义:{desc} 格式约束:{fmt}"

                # key 记录身份: 数据库.表.列
                keys.append(f"{db_name}.{table_name}.{col_name}")
                texts.append(rich_text)

    if not texts:
        print("⚠️ 未发现任何可被向量化的文本，请检查 metadata_cache 格式是否为最新的 JSON。")
        return

    print(f"📦 共计装填了 {len(texts)} 个核心字段的高密度语义，准备调用 API 向量化...")

    # 2. 分批调用 Embedding API
    # ⚠️ 阿里云 DashScope `text-embedding-v3` 单次批处理通常不宜过大
    batch_size = 8
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(f"  - 正在请求嵌入模型计算向量余弦矩阵进度 ({i}/{len(texts)})...")

        # 增加重试机制抗波动
        for attempt in range(3):
            try:
                response = get_embedding_client().embeddings.create(
                    model=get_embedding_model(),
                    input=batch_texts,
                )
                batch_emb = [item.embedding for item in response.data]
                all_embeddings.extend(batch_emb)
                time.sleep(0.5)
                break
            except Exception as e:
                print(f"    ⏳ 向量化 API 异常 (尝试 {attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            print("❌ 多次重试均失败，中止向量化。")
            return

    # 3. 结果固化为 numpy 矩阵
    if len(all_embeddings) == len(texts):
        emb_matrix = np.array(all_embeddings, dtype=np.float32)
        np.savez(output_path, embeddings=emb_matrix, keys=np.array(keys))
        print("\n✅ Dual-Recall 双路召回底层基建完毕！")
        print(f"💾 向量库已成功落盘于 {output_path}。大模型这下有真正的“千里眼”了。")
    else:
        print("⚠️ 向量数组长度不匹配，数据可能部分丢失。")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_vector_index()
