import sqlite3
import os
import pandas as pd


def basic_profiling(db_path):
    print(f"🚀 开始对数据库进行基础画像 (Basic Profiling): {db_path}\n")

    # 连接数据库
    conn = sqlite3.connect(db_path)

    # 提取所有表名
    tables_df = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )

    for table_name in tables_df["name"]:
        print(f"📊 正在分析表: 【{table_name}】")

        # 读取整张表
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"  - 总行数: {len(df)}")

        # 遍历每一列，提取论文要求的物理特征
        for col in df.columns:
            null_count = df[col].isnull().sum()
            distinct_count = df[col].nunique()

            # 提取论文中强调的 Top-3 高频样本 (Top-k field values)
            top_samples = df[col].value_counts().head(3).index.tolist()

            print(f"  🔹 字段 [{col}]:")
            print(f"      - 空值数量: {null_count}")
            print(f"      - 唯一值数量: {distinct_count}")
            print(f"      - 高频样本: {top_samples}")
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    # 精准指向你截图里的 california_schools 数据库
    # 注意：BIRD 数据库的后缀通常是 .sqlite
    db_file = os.path.join(
        "data", "dev_databases", "california_schools", "california_schools.sqlite"
    )

    if os.path.exists(db_file):
        basic_profiling(db_file)
    else:
        print(f"❌ 找不到文件: {db_file}。请确认里面数据库的后缀是 .sqlite 还是 .db？")
