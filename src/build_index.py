import sqlite3
import pandas as pd
import json
import os


def build_literal_index(db_path, output_json_path, max_values_per_col=1000):
    """
    扫描数据库，提取文本字段的唯一值，构建【值 -> 字段名】的映射字典。
    对应 AT&T 论文的预处理步骤：为 Schema Linking 建立字面量检索基础。
    """
    print(f"🗂️ 开始为数据库构建字面量索引: {db_path}...")
    conn = sqlite3.connect(db_path)
    tables_df = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )

    value_to_columns = {}

    for table in tables_df["name"]:
        print(f"  - 扫描表: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        for col in df.columns:
            # 我们只对文本类型的值建立索引（数字和日期通常不需要精准的倒排索引）
            if df[col].dtype == "object":
                # 获取该列去重后的值，剔除空值
                unique_values = df[col].dropna().unique()

                # 限制提取数量，防止字典过大（论文中提到 BIRD benchmark 使用了 N=10000）
                for val in unique_values[:max_values_per_col]:
                    val_str = str(val).strip()
                    if not val_str:
                        continue

                    col_identifier = f"{table}.{col}"

                    if val_str not in value_to_columns:
                        value_to_columns[val_str] = []
                    if col_identifier not in value_to_columns[val_str]:
                        value_to_columns[val_str].append(col_identifier)

    conn.close()

    # 将建好的“字典”保存到本地
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(value_to_columns, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 索引构建完成！共收录了 {len(value_to_columns)} 个独立字面量。")
    print(f"📂 字典已保存至: {output_json_path}")


if __name__ == "__main__":
    db_file = os.path.join(
        "data", "dev_databases", "california_schools", "california_schools.sqlite"
    )
    # 我们把生成的字典也存放在 data 目录下
    out_file = os.path.join("data", "literal_index.json")

    if os.path.exists(db_file):
        build_literal_index(db_file, out_file)
    else:
        print(f"❌ 找不到数据库文件: {db_file}")
