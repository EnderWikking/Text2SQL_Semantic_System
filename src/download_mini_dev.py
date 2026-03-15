import json
import os
from datasets import load_dataset


def download_and_process_mini_dev():
    print("🚀 正在从 Hugging Face 云端拉取 BIRD Mini-Dev (迷你验证集)...")
    print("这可能需要半分钟左右的时间，请耐心等待⏳\n")

    # 自动下载并加载官方数据集
    dataset = load_dataset("birdsql/bird_mini_dev")

    # 我们只需要 SQLite 版本的 500 道题
    sqlite_data = dataset["mini_dev_sqlite"]

    mini_dev_json = []
    mini_dev_sql = []

    for item in sqlite_data:
        # 保存完整题目信息
        mini_dev_json.append(item)
        # 为官方阅卷机提取纯净版的标准答案 (格式为: SQL语句 + \t + db_id)
        # 把换行符去掉，防止阅卷脚本解析报错
        clean_sql = item["SQL"].replace("\n", " ").replace("\r", " ")
        mini_dev_sql.append(f"{clean_sql}\t{item['db_id']}")

    # 确保存放的路径正确
    os.makedirs("data", exist_ok=True)

    # 1. 落地为 JSON 考卷
    with open("data/mini_dev.json", "w", encoding="utf-8") as f:
        json.dump(mini_dev_json, f, ensure_ascii=False, indent=4)

    # 2. 落地为纯净的 SQL 标准答案本
    with open("data/mini_dev.sql", "w", encoding="utf-8") as f:
        for line in mini_dev_sql:
            f.write(line + "\n")

    print("-" * 50)
    print("✅ Mini-Dev 下载并处理成功！你获得了：")
    print(f"📄 data/mini_dev.json (共 {len(mini_dev_json)} 道浓缩精华题)")
    print(f"📄 data/mini_dev.sql (配套的 500 个绝对正确的黄金标准 SQL)")


if __name__ == "__main__":
    download_and_process_mini_dev()
