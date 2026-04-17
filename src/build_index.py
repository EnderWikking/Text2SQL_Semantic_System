import json
import os
import sqlite3

from knowledge_base_paths import describe_active_kb, get_literal_index_path


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def is_text_like(declared_type):
    t = (declared_type or "").upper()
    return any(k in t for k in ("CHAR", "CLOB", "TEXT", "JSON"))


def build_literal_index(db_path, output_json_path, max_values_per_col=1000):
    """
    扫描数据库，提取文本字段的唯一值，构建【值 -> 字段名】映射字典。
    采用按列 DISTINCT + LIMIT 方式，避免一次性读取整表导致内存压力。
    """
    print(f"🗂️ 开始为数据库构建字面量索引: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_rows = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    value_to_columns = {}

    for (table,) in table_rows:
        print(f"  - 扫描表: {table}")
        quoted_table = quote_identifier(table)
        col_rows = cursor.execute(f"PRAGMA table_info({quoted_table})").fetchall()

        for col_row in col_rows:
            col_name = col_row[1]
            declared_type = col_row[2]
            if not is_text_like(declared_type):
                continue

            quoted_col = quote_identifier(col_name)
            sql = (
                f"SELECT DISTINCT {quoted_col} FROM {quoted_table} "
                f"WHERE {quoted_col} IS NOT NULL LIMIT ?"
            )
            values = cursor.execute(sql, (max_values_per_col,)).fetchall()

            for (val,) in values:
                val_str = str(val).strip()
                if not val_str:
                    continue

                col_identifier = f"{table}.{col_name}"
                if val_str not in value_to_columns:
                    value_to_columns[val_str] = []
                if col_identifier not in value_to_columns[val_str]:
                    value_to_columns[val_str].append(col_identifier)

    conn.close()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(value_to_columns, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 索引构建完成！共收录了 {len(value_to_columns)} 个独立字面量。")
    print(f"📂 字典已保存至: {output_json_path}")


if __name__ == "__main__":
    db_file = os.path.join(
        "data", "dev_databases", "california_schools", "california_schools.sqlite"
    )
    out_file = get_literal_index_path()
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"🧱 当前知识库命名空间: {describe_active_kb()} -> {out_file}")

    if os.path.exists(db_file):
        build_literal_index(db_file, out_file)
    else:
        print(f"❌ 找不到数据库文件: {db_file}")
