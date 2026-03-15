import os

# 🛡️ 护甲 1：依然保留物理超度“幽灵代理”，防止本地网络环境干扰
proxy_keys = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]
for key in proxy_keys:
    if key in os.environ:
        del os.environ[key]

import json
import sqlite3
import re
import time
from openai import OpenAI

# 🌟 核心修改 1：切换到 DeepSeek 专属接口
client = OpenAI(
    api_key="sk-7106955771c249b59a10831ab07b71b9",  # ⚠️ 填入你的 DeepSeek Key
    base_url="https://api.deepseek.com",  # ⚠️ 指向 DeepSeek 的服务器
)

# 🌟 核心修改 2：解除封印！把 TEST_LIMIT 改成 None，准备一次性跑完 500 题！
# 如果你想再测两道题定定心，可以先改成 2，跑通后再改成 None
TEST_LIMIT = None


def extract_sql(text):
    """从大模型回复中剥离纯净 SQL"""
    match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().replace("\n", " ")


# ... (前面的 import 和 client 初始化保持不变) ...


def run_batch_evaluation():
    print("🚀 [全自动批量阅卷机 - 断点续传版] 启动！准备冲击 EX 分数...")

    with open("data/mini_dev.json", "r", encoding="utf-8") as f:
        mini_dev_data = json.load(f)

    if TEST_LIMIT:
        mini_dev_data = mini_dev_data[:TEST_LIMIT]

    # 🌟 核心修改：断点续传逻辑
    results_log = []
    start_index = 0
    correct_count = 0

    archive_path = "data/evaluation_results.json"
    if os.path.exists(archive_path):
        try:
            with open(archive_path, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            start_index = len(results_log)  # 看看已经存了多少题
            # 重新计算已经答对的题目数
            correct_count = sum(1 for item in results_log if item["is_correct"])
            print(
                f"📦 发现历史存档！已完成 {start_index} 题，答对 {correct_count} 题。"
            )
            print(
                f"⏭️ 将跳过已完成的题目，直接从第 {start_index + 1} 题开始断点续传！\n"
            )
        except Exception as e:
            print("⚠️ 读取存档失败，将从头开始...")

    total_processed = start_index

    # 🌟 核心修改：从 start_index 切片开始循环
    for index, test_case in enumerate(mini_dev_data[start_index:], start=start_index):
        db_id = test_case["db_id"]
        # ... (接下来的代码完全保持不变) ...
        question = test_case["question"]
        gold_sql = test_case["SQL"]

        print(f"\n[{index + 1}/{len(mini_dev_data)}] 正在处理数据库: {db_id}")

        cache_path = os.path.join("data", "metadata_cache", f"{db_id}_enhanced.json")
        if not os.path.exists(cache_path):
            print(f"  ⏭️ 警告: 找不到 {db_id} 的增强知识库，跳过该题。")
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            schema_knowledge = f.read()

        prompt = f"""
你是一个精通 Text-to-SQL 的数据库专家。请根据以下数据库的物理与语义画像，为用户的提问编写正确的 SQL。
只输出 SQL，不要解释，用 ```sql 包裹。

【⚠️极其重要的严厉警告⚠️】：
1. 请严格、仅查询提问中明确要求的字段！
2. 绝对不要输出任何多余的列（例如不要把中间计算结果、不要把总量也 SELECT 出来）。
3. 如果题目只问了一个值（比如比例、最大值、名字），你的 SELECT 后面只能有一个字段！

【增强知识库】:
{schema_knowledge}

【用户提问】:
{question}
"""
        predicted_sql = "ERROR"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # 🌟 核心修改 3：使用 DeepSeek 模型 (deepseek-chat 也就是最新的 V3 模型，极其便宜且代码能力强)
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个严谨的 Text-to-SQL 数据分析师。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    timeout=60,
                )
                predicted_sql = extract_sql(response.choices[0].message.content)

                # 🌟 核心修改 4：删除了原先 3 秒的强制休眠，直接全速推进！
                break

            except Exception as e:
                print(f"  ⚠️ API 调用波动或超时: {e}")
                if attempt == max_retries - 1:
                    print("  ❌ 3次尝试均失败，最终放弃该题。")
                else:
                    print("  🔄 触发重试防御机制，冷却 2 秒后重新发起请求...")
                    time.sleep(2)  # 失败了稍微等一下再试

        if predicted_sql == "ERROR":
            continue

        db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
        is_correct = False
        error_msg = ""

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(gold_sql)
            gold_result = cursor.fetchall()

            cursor.execute(predicted_sql)
            pred_result = cursor.fetchall()
            conn.close()

            if set(gold_result) == set(pred_result):
                is_correct = True
                correct_count += 1
                print("  🎉 结果: 【匹配成功】 (得分 +1)")
            else:
                print("  💔 结果: 【匹配失败】 (查询结果不一致)")

        except Exception as e:
            error_msg = str(e)
            print(f"  💥 结果: 【执行报错】 ({error_msg})")

        total_processed += 1
        current_acc = (correct_count / total_processed) * 100
        print(
            f"  📊 当前实时准确率 (EX): {current_acc:.2f}% ({correct_count}/{total_processed})"
        )

        results_log.append(
            {
                "question_id": index + 1,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "is_correct": is_correct,
                "error_msg": error_msg,
            }
        )

        with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results_log, f, ensure_ascii=False, indent=2)

    if total_processed > 0:
        final_acc = (correct_count / total_processed) * 100
        print("\n" + "=" * 50)
        print(f"🏆 跑分结束！")
        print(f"📈 最终执行准确率 (Execution Accuracy): {final_acc:.2f}%")
        print(f"✅ 答对题目: {correct_count} / {total_processed}")
        print("=" * 50)


if __name__ == "__main__":
    run_batch_evaluation()

# import os

# # 🛡️ 护甲 1：物理超度“幽灵代理”，彻底清空终端残留的 Clash 环境变量
# proxy_keys = [
#     "HTTP_PROXY",
#     "HTTPS_PROXY",
#     "ALL_PROXY",
#     "http_proxy",
#     "https_proxy",
#     "all_proxy",
# ]
# for key in proxy_keys:
#     if key in os.environ:
#         del os.environ[key]

# import json
# import sqlite3
# import re
# import time
# from openai import OpenAI

# # 🛡️ 护甲 2：初始化大模型客户端
# client = OpenAI(
#     api_key="sk-90a75b57806f4794bdbb96273df856a3",  # ⚠️ 记得填入你阿里云的 Key，千万别传到 GitHub 上
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# # 🌟 测试规模控制：咱们先跑 3 题试试水，确认稳定后再改成 None 跑全量！
# TEST_LIMIT = 3


# def extract_sql(text):
#     """从大模型回复中剥离纯净 SQL"""
#     match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return text.strip().replace("\n", " ")


# def run_batch_evaluation():
#     print("🚀 [全自动批量阅卷机 - 终极装甲版] 启动！准备冲击 EX 分数...")

#     with open("data/mini_dev.json", "r", encoding="utf-8") as f:
#         mini_dev_data = json.load(f)

#     if TEST_LIMIT:
#         mini_dev_data = mini_dev_data[:TEST_LIMIT]
#         print(f"⚠️ 当前处于快速测试模式，只抽取前 {TEST_LIMIT} 道题进行跑分。")

#     correct_count = 0
#     total_processed = 0
#     results_log = []

#     for index, test_case in enumerate(mini_dev_data):
#         db_id = test_case["db_id"]
#         question = test_case["question"]
#         gold_sql = test_case["SQL"]

#         print(f"\n[{index + 1}/{len(mini_dev_data)}] 正在处理数据库: {db_id}")

#         # 1. 加载增强知识库（如果找不到就跳过）
#         cache_path = os.path.join("data", "metadata_cache", f"{db_id}_enhanced.json")
#         if not os.path.exists(cache_path):
#             print(f"  ⏭️ 警告: 找不到 {db_id} 的增强知识库，跳过该题。")
#             continue

#         with open(cache_path, "r", encoding="utf-8") as f:
#             schema_knowledge = f.read()

#         # 2. 组装 Prompt
#         prompt = f"""
# 你是一个精通 Text-to-SQL 的数据库专家。请根据以下数据库的物理与语义画像，为用户的提问编写正确的 SQL。
# 只输出 SQL，不要解释，用 ```sql 包裹。

# 【增强知识库】:
# {schema_knowledge}

# 【用户提问】:
# {question}
# """
#         predicted_sql = "ERROR"
#         max_retries = 3  # 🛡️ 护甲 3：防网络波动的三次重试机制

#         for attempt in range(max_retries):
#             try:
#                 print(
#                     f"  ⏳ 正在请求千问大模型思考中... (尝试 {attempt + 1}/{max_retries})"
#                 )
#                 response = client.chat.completions.create(
#                     model="qwen3.5-plus",
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": "你是一个严谨的 Text-to-SQL 数据分析师。",
#                         },
#                         {"role": "user", "content": prompt},
#                     ],
#                     timeout=60,  # 放宽到 60 秒，容忍跨国高延迟
#                 )
#                 predicted_sql = extract_sql(response.choices[0].message.content)

#                 # 🛡️ 护甲 4：防限流休眠。成功后强制休息 3 秒，不给阿里云服务器压力
#                 print("  🛏️ 答题成功，强制休息 3 秒，保护 API 免费额度...")
#                 time.sleep(3)
#                 break  # 成功获取结果，跳出重试循环

#             except Exception as e:
#                 print(f"  ⚠️ API 调用波动或超时: {e}")
#                 if attempt == max_retries - 1:
#                     print("  ❌ 3次尝试均失败，最终放弃该题。")
#                 else:
#                     print("  🔄 触发重试防御机制，冷却 5 秒后重新发起请求...")
#                     time.sleep(5)  # 失败了休息更久一点再试

#         # 如果3次都失败了，跳过这题的数据库执行
#         if predicted_sql == "ERROR":
#             continue

#         # 3. 真实数据库双轨执行与比对
#         db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
#         is_correct = False
#         error_msg = ""

#         try:
#             conn = sqlite3.connect(db_path)
#             cursor = conn.cursor()

#             # 执行官方 SQL
#             cursor.execute(gold_sql)
#             gold_result = cursor.fetchall()

#             # 执行预测 SQL
#             cursor.execute(predicted_sql)
#             pred_result = cursor.fetchall()
#             conn.close()

#             if set(gold_result) == set(pred_result):
#                 is_correct = True
#                 correct_count += 1
#                 print("  🎉 结果: 【匹配成功】 (得分 +1)")
#             else:
#                 print("  💔 结果: 【匹配失败】 (查询结果不一致)")

#         except Exception as e:
#             error_msg = str(e)
#             print(f"  💥 结果: 【执行报错】 ({error_msg})")

#         total_processed += 1
#         current_acc = (correct_count / total_processed) * 100
#         print(
#             f"  📊 当前实时准确率 (EX): {current_acc:.2f}% ({correct_count}/{total_processed})"
#         )

#         # 4. 实时存档，防断电
#         results_log.append(
#             {
#                 "question_id": index + 1,
#                 "db_id": db_id,
#                 "question": question,
#                 "gold_sql": gold_sql,
#                 "predicted_sql": predicted_sql,
#                 "is_correct": is_correct,
#                 "error_msg": error_msg,
#             }
#         )

#         with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
#             json.dump(results_log, f, ensure_ascii=False, indent=2)

#     if total_processed > 0:
#         final_acc = (correct_count / total_processed) * 100
#         print("\n" + "=" * 50)
#         print(f"🏆 跑分结束！")
#         print(f"📈 最终执行准确率 (Execution Accuracy): {final_acc:.2f}%")
#         print(f"✅ 答对题目: {correct_count} / {total_processed}")
#         print("=" * 50)


# if __name__ == "__main__":
#     run_batch_evaluation()
