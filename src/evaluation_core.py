import os
import json
import sqlite3
import re
import time
from openai import OpenAI
import concurrent.futures

# 环境清理防干扰
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(key, None)

client = OpenAI(
    api_key="sk-7106955771c249b59a10831ab07b71b9",
    base_url="https://api.deepseek.com",
)

def extract_sql(text):
    # Retrieve the last purely sql-marked block
    matches = re.findall(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback to any code block
    blocks = re.findall(r"```(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
        
    return text.strip().replace("\n", " ")

def process_single_case(args):
    index, test_case, mode, official_knowledge = args
    db_id = test_case["db_id"]
    question = test_case["question"]
    gold_sql = test_case["SQL"]
    evidence = test_case.get("evidence", "").strip()

    db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(db_path)
        raw_ddl = "\n".join([row[0] for row in conn.cursor().execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall() if row[0]])
        conn.close()
    except Exception as e:
        return {"question_id": index + 1, "db_id": db_id, "question": question, "gold_sql": gold_sql, "predicted_sql": "ERROR", "is_correct": False, "error_msg": str(e)}

    external_context = ""
    system_content = "你是一个严谨的 Text-to-SQL 数据分析师。"
    
    # 通用 evidence 注入段（498/500题都自带这个关键提示）
    evidence_block = ""
    if evidence:
        evidence_block = f"\n\n【关键提示 (Evidence)】：\n{evidence}"
    
    if mode == "baseline":
        external_context = f"【数据库建表语句 DDL】:\n{raw_ddl}{evidence_block}"
        
    elif mode == "official":
        external_context = f"【数据库建表语句 DDL】:\n{raw_ddl}{evidence_block}"
        
    elif mode == "enhanced":
        cache_path = os.path.join("data", "metadata_cache", f"{db_id}_enhanced.json")
        schema_knowledge = ""
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                full_meta = json.load(f)
            # 只提取语义描述，剔除冗余的 physical_stats（省 ~60% Token）
            slim_parts = []
            for table_name, info in full_meta.items():
                sem = info.get("semantic_description", {})
                if sem:
                    slim_parts.append(f"表 {table_name}: {json.dumps(sem, ensure_ascii=False)}")
            schema_knowledge = "\n".join(slim_parts)
        
        external_context = (
            "【SQL 生成核心规则】：\n"
            "1. 严格字段映射：SELECT 只放问题要求的目标列。\n"
            "2. 除法/比例分子转 CAST(A AS REAL)。\n"
            "3. 极值用 ORDER BY ... LIMIT 1。\n"
            "4. 不要输出任何解释文字，直接给出 ```sql 代码块。\n\n"
            f"【数据库建表语句 DDL】:\n{raw_ddl}{evidence_block}\n\n"
            f"【增强语义知识库】:\n{schema_knowledge}"
        )

    # == 缝合最终 Prompt ==
    prompt = f"你是一顶尖 SQL 专家。请仔细分析问题需要哪些表和列、如何 JOIN，但不要输出分析过程，直接且仅返回可在 SQLite 运行的 ```sql 代码块。\n\n{external_context}\n\n【用户提问】:\n{question}"
    
    # 🌟 [单轮生成 + 1次纠错] 🌟
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
    predicted_sql = "ERROR"
    final_error_msg = ""
    is_correct = False
    
    for attempt in range(2):  # 最多1次纠错重试
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                timeout=90,
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            predicted_sql = extract_sql(response_content)
        except Exception as e:
            final_error_msg = str(e)
            break
        
        if predicted_sql == "ERROR":
            break
            
        # 沙盒执行验证语法
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(predicted_sql)
            pred_res = cur.fetchall()
            conn.close()
            final_error_msg = ""
            break  # 执行成功，跳出
        except Exception as exec_e:
            final_error_msg = str(exec_e)
            if attempt == 0:  # 仅允许1次纠错
                feedback = f"你生成的 SQL 执行报错：{final_error_msg}\n请修复语法错误，直接给出修正后的 ```sql 代码块。"
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": feedback})
                continue
            
    # 对比金标准
    if predicted_sql != "ERROR" and not final_error_msg:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(predicted_sql)
            pred_res = cur.fetchall()
            cur.execute(gold_sql)
            gold_res = cur.fetchall()
            conn.close()
            
            if set(gold_res) == set(pred_res):
                is_correct = True
        except Exception as e:
            final_error_msg = str(e)

    print(f"[{index + 1}] {'✅' if is_correct else '❌'}")
    return {
        "question_id": index + 1, "db_id": db_id, "question": question,
        "gold_sql": gold_sql, "predicted_sql": predicted_sql,
        "is_correct": is_correct, "error_msg": final_error_msg
    }

def run_unified_evaluation(mode, dataset_path="data/mini_dev.json", test_limit=None):
    print(f"\n🚀 [终极评测引擎] 雷霆出击！所有历史成绩全量作废，开启【沙盒自修正 (Reflexion)】进行最为纯净的从零评估！ 当前挂载模式: 【{mode.upper()}】")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if test_limit:
        data = data[:test_limit]

    official_knowledge = {}
    if mode == "official" and os.path.exists("data/dev_tied_append.json"):
        with open("data/dev_tied_append.json", "r", encoding="utf-8") as f:
            official_knowledge = json.load(f)

    # 准备多线程参数
    args_list = [(i, tc, mode, official_knowledge) for i, tc in enumerate(data)]
    
    results_log = []
    correct_count = 0
    total_processed = len(data)

    print(f"📦 开始通过并发池高速测试全量 {len(args_list)} 个查询 (支持执行态自修复)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_single_case, args): args[0] for args in args_list}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results_log.append(res)
            if res["is_correct"]:
                correct_count += 1
                
    # 按原题目顺序排序，不然看起来很混乱！
    results_log.sort(key=lambda x: x["question_id"])

    # 统一存盘
    save_path = f"data/evaluation_results_{mode}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    if total_processed > 0:
        final_acc = (correct_count / total_processed) * 100
        print("\n" + "=" * 60)
        print(f"🏆 【{mode.upper()} 组】学术消融实验收官战报！")
        print(f"📈 最终 EX 准确率 (Execution Acc): {final_acc:.2f}%")
        print(f"✅ 成功跑通数目: {correct_count} / {total_processed}")
        print("=" * 60)
