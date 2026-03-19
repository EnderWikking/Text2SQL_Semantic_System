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

    db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(db_path)
        raw_ddl = "\n".join([row[0] for row in conn.cursor().execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall() if row[0]])
        conn.close()
    except Exception as e:
        return {"question_id": index + 1, "db_id": db_id, "question": question, "gold_sql": gold_sql, "predicted_sql": "ERROR", "is_correct": False, "error_msg": str(e)}

    external_context = ""
    system_content = "你是一个严谨的 Text-to-SQL 数据分析师。"
    
    if mode == "baseline":
        external_context = f"【数据库建表语句 DDL】:\n{raw_ddl}"
        
    elif mode == "official":
        evidence = "No additional evidence."
        if isinstance(official_knowledge, list) and index < len(official_knowledge):
            item = official_knowledge[index]
            evidence = item.get("evidence", str(item)) if isinstance(item, dict) else str(item)
        elif isinstance(official_knowledge, dict):
            evidence = official_knowledge.get(str(index), "No additional evidence.")
        external_context = f"【数据库建表语句 DDL】:\n{raw_ddl}\n\n【官方提供辅助知识 (Official Evidence)】:\n{evidence}"
        
    elif mode == "enhanced":
        cache_path = os.path.join("data", "metadata_cache", f"{db_id}_enhanced.json")
        schema_knowledge = ""
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                schema_knowledge = f.read()
        
        # 极度强化的语义约束，采用最高级的零容忍法则
        external_context = (
            "【⚠️强制思维链与斩获高分的绝对法则⚠️】：\n"
            "0. 终极奥义【严格字段映射】：仔细阅读问题要什么！如果问“Who/Name/which person”，最终 SELECT 绝对只能有名字字段，多一个列（比如顺带把金额 SELECT 出来）就直接零分！若需聚合或排序请只在 ORDER BY 里完成，SELECT一律干干净净只放目标列。\n"
            "1. 你必须首先输出一段 <think> 解析过程 </think>，在这里面分析：它需要什么表？主键外键如何 JOIN？如果用了隐式 FROM A, B，立刻在草稿中重写为明确的 JOIN。\n"
            "2. 针对聚合函数的日期：注意年份直接用 LIKE '2012%' 或者 strftime('%Y', Date)。平均每月注意需要除以 12 等。\n"
            "3. 遇到极值或先后单词（如 least, most, top, newest），必须用 `ORDER BY ... LIMIT 1`。\n"
            "4. 字符安全：表如果报错没找到，检查是否加了双引号；字符串值（如地点、人名）比较必须用单引号转义。\n"
            "5. 当题目要比例/占比时，所有的除法必须将分子强制转为浮点 `CAST(A AS REAL) / B`。\n\n"
            f"【数据库原生建表语句 (Hard DDL)】:\n{raw_ddl}\n\n"
            f"【您挖掘出的最高阶增强语义知识库 (LLM Enhanced Metadata)】:\n{schema_knowledge}"
        )

    # == 缝合最终 Prompt ==
    prompt = f"你是一顶尖 SQL 专家。请结合以下上下文线索，编写能在 SQLite 中稳定运行的代码。你必须先在 <think> 标签内进行精细的思维链推理，推理结束后再给出最终的 ```sql 代码块。\n\n{external_context}\n\n【用户实体提问】:\n{question}"
    
    # 🌟 [Agent自我修正核心逻辑 (Reflexion)] 🌟
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
    predicted_sql = "ERROR"
    final_error_msg = ""
    is_correct = False
    
    for attempt in range(3): # 最多允许自我反思修复 3 次
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                timeout=90,
                temperature=0.2
            )
            response_content = response.choices[0].message.content
            predicted_sql = extract_sql(response_content)
        except Exception as e:
            time.sleep(1)
            continue
            
        if predicted_sql == "ERROR":
            continue
            
        # 尝试沙盒执行，若报错则反馈给大模型让他自己修
        try:
            conn = sqlite3.connect(db_path)
            # 配置 3秒 watchdog 防死机
            exec_start_time = time.time()
            def progress_handler():
                if time.time() - exec_start_time > 3.0:
                    return 1
                return 0
            conn.set_progress_handler(progress_handler, 1000)
            
            cur = conn.cursor()
            cur.execute(predicted_sql)
            pred_res = cur.fetchall()
            conn.close()
            
            # 若没报错，说明语法完全OK，跳出自我反思循环，去对比金标准
            final_error_msg = ""
            break
            
        except Exception as exec_e:
            final_error_msg = str(exec_e)
            # 有语法错误！把错误信息抛回给它
            feedback = f"你上次生成的 SQL 在 SQLite 中执行失败了，报错信息是：\n{final_error_msg}\n请你重新思考语法结构，修复该错误，并再次给我一个完整的包含 ```sql 的答复。"
            messages.append({"role": "assistant", "content": response_content})
            messages.append({"role": "user", "content": feedback})
            continue

    # 全量重考验证金标准
    if not final_error_msg:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(gold_sql)
            gold_res = cur.fetchall()
            
            cur.execute(predicted_sql)
            pred_res = cur.fetchall()
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
