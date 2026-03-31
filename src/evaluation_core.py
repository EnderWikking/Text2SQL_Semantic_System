import os
import json
import sqlite3
import re
import time
import concurrent.futures
import threading
from collections import Counter, defaultdict
from provider_clients import get_chat_client, get_chat_model
DEFAULT_MAX_WORKERS = max(1, int(os.getenv("EVAL_MAX_WORKERS", "8")))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "4")))
LLM_API_RETRY_MAX = max(1, int(os.getenv("LLM_API_RETRY_MAX", "3")))
LLM_API_RETRY_BASE_SECONDS = max(0.1, float(os.getenv("LLM_API_RETRY_BASE_SECONDS", "1.5")))
EVAL_CANDIDATE_COUNT = max(1, int(os.getenv("EVAL_CANDIDATE_COUNT", "3")))
_API_CALL_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)

def is_retryable_llm_error(exc):
    text = str(exc).lower()
    retry_markers = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "timeout",
        "timed out",
        "temporar",
        "connection",
        "network",
        "overload",
    ]
    return any(marker in text for marker in retry_markers)

def call_chat_completion_with_retry(messages, temperature=0.0):
    last_error = None
    for attempt in range(LLM_API_RETRY_MAX):
        try:
            with _API_CALL_SEMAPHORE:
                response = get_chat_client().chat.completions.create(
                    model=get_chat_model(),
                    messages=messages,
                    timeout=90,
                    temperature=temperature
                )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt == LLM_API_RETRY_MAX - 1 or not is_retryable_llm_error(e):
                raise
            sleep_seconds = LLM_API_RETRY_BASE_SECONDS * (2 ** attempt)
            time.sleep(sleep_seconds)
    raise last_error

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

def get_candidate_variant_hint(candidate_idx):
    hints = [
        "优先保证筛选条件中 literal 与题面严格一致。",
        "优先先确定聚合/分组，再补 JOIN 与过滤条件。",
        "优先先写清 JOIN 关系，再补聚合与排序。",
    ]
    return hints[candidate_idx % len(hints)]

def multiset_key(rows):
    row_counter = Counter(rows or [])
    return tuple(sorted((repr(k), v) for k, v in row_counter.items()))

def normalize_lookup_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()

def normalize_question_id(question_id):
    if question_id in (None, ""):
        return None
    return str(question_id)

def build_official_knowledge_index(raw_knowledge):
    index = {
        "by_question_id": {},
        "by_db_and_question": {},
    }

    entries = []
    if isinstance(raw_knowledge, list):
        entries = raw_knowledge
    elif isinstance(raw_knowledge, dict):
        for key, value in raw_knowledge.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        entry = dict(item)
                        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                            entry.setdefault("question_id", key)
                        elif isinstance(key, str):
                            entry.setdefault("db_id", key)
                        entries.append(entry)
            elif isinstance(value, dict):
                entry = dict(value)
                if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                    entry.setdefault("question_id", key)
                elif isinstance(key, str):
                    entry.setdefault("db_id", key)
                entries.append(entry)

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        normalized_entry = {
            "question_id": normalize_question_id(entry.get("question_id")),
            "db_id": entry.get("db_id"),
            "question": normalize_lookup_text(entry.get("question", "")),
            "evidence": (entry.get("evidence") or "").strip(),
        }

        if normalized_entry["question_id"]:
            index["by_question_id"][normalized_entry["question_id"]] = normalized_entry
        if normalized_entry["db_id"] and normalized_entry["question"]:
            key = (normalized_entry["db_id"], normalized_entry["question"])
            index["by_db_and_question"][key] = normalized_entry

    return index

def resolve_official_knowledge(test_case, official_knowledge_index):
    if not official_knowledge_index:
        return None

    db_id = test_case.get("db_id")
    question = normalize_lookup_text(test_case.get("question", ""))
    question_id = normalize_question_id(test_case.get("question_id"))

    match = None
    if question_id:
        candidate = official_knowledge_index["by_question_id"].get(question_id)
        if candidate:
            same_db = not db_id or not candidate["db_id"] or candidate["db_id"] == db_id
            if same_db:
                match = {**candidate, "match_type": "question_id"}

    if not match and db_id and question:
        candidate = official_knowledge_index["by_db_and_question"].get((db_id, question))
        if candidate:
            match = {**candidate, "match_type": "db_id+question"}

    return match

def process_single_case(args):
    index, test_case, mode, official_evidence, include_dataset_evidence, candidate_count = args
    question_id = test_case.get("question_id", index)
    db_id = test_case["db_id"]
    question = test_case["question"]
    gold_sql = test_case["SQL"]
    evidence = test_case.get("evidence", "").strip() if include_dataset_evidence else ""

    db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
    try:
        with sqlite3.connect(db_path) as conn:
            raw_ddl = "\n".join(
                [
                    row[0]
                    for row in conn.cursor()
                    .execute("SELECT sql FROM sqlite_master WHERE type='table'")
                    .fetchall()
                    if row[0]
                ]
            )
    except Exception as e:
        return {
            "order_index": index,
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": "ERROR",
            "is_correct": False,
            "error_msg": str(e),
        }

    external_context = ""
    system_content = "你是一个严谨的 Text-to-SQL 数据分析师。"

    context_parts = [f"【数据库建表语句 DDL】:\n{raw_ddl}"]
    if evidence:
        context_parts.append(f"【关键提示 (Evidence)】：\n{evidence}")
    if mode in ("official", "fused") and official_evidence:
        context_parts.append(f"【外部官方知识 Evidence [1]】：\n{official_evidence}")
    base_context = "\n\n".join(context_parts)
    
    if mode == "baseline":
        external_context = base_context
        
    elif mode == "official":
        external_context = base_context
        
    elif mode in ("enhanced", "fused"):
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
            f"{base_context}\n\n"
            f"【增强语义知识库】:\n{schema_knowledge}"
        )

    # == 多候选生成 + 投票 ==
    candidate_outputs = []
    for candidate_idx in range(max(1, candidate_count)):
        variant_hint = get_candidate_variant_hint(candidate_idx)
        prompt = (
            "你是一顶尖 SQL 专家。请仔细分析问题需要哪些表和列、如何 JOIN，但不要输出分析过程，"
            "直接且仅返回可在 SQLite 运行的 ```sql 代码块。\n\n"
            f"{external_context}\n\n"
            f"【候选生成偏好】:\n{variant_hint}\n\n"
            f"【用户提问】:\n{question}"
        )
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
        predicted_sql = "ERROR"
        final_error_msg = ""
        pred_res = None

        for attempt in range(2):  # 单候选：首轮 + 1次纠错
            try:
                temperature = 0.0 if candidate_idx == 0 else 0.2
                response_content = call_chat_completion_with_retry(messages, temperature=temperature)
                predicted_sql = extract_sql(response_content)
            except Exception as e:
                final_error_msg = str(e)
                break

            if predicted_sql == "ERROR":
                break

            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(predicted_sql)
                    pred_res = cur.fetchall()
                final_error_msg = ""
                break
            except Exception as exec_e:
                final_error_msg = str(exec_e)
                if attempt == 0:
                    feedback = (
                        f"你生成的 SQL 执行报错：{final_error_msg}\n"
                        "请修复语法错误，直接给出修正后的 ```sql 代码块。"
                    )
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({"role": "user", "content": feedback})
                    continue

        candidate_outputs.append(
            {
                "candidate_idx": candidate_idx,
                "sql": predicted_sql,
                "error_msg": final_error_msg,
                "pred_res": pred_res,
                "valid": predicted_sql != "ERROR" and not final_error_msg and pred_res is not None,
            }
        )

    valid_candidates = [c for c in candidate_outputs if c["valid"]]
    selected = candidate_outputs[0]
    if valid_candidates:
        vote_groups = defaultdict(list)
        for c in valid_candidates:
            vote_groups[multiset_key(c["pred_res"])].append(c)
        best_group = max(vote_groups.values(), key=lambda g: len(g))
        # 同票时倾向更短 SQL
        best_group.sort(key=lambda item: len(item["sql"] or ""))
        selected = best_group[0]
    else:
        # 所有候选都失败时，返回最后一个候选，便于排错
        selected = candidate_outputs[-1]

    predicted_sql = selected["sql"]
    final_error_msg = selected["error_msg"]
    pred_res = selected["pred_res"]
    is_correct = False

    if predicted_sql != "ERROR" and not final_error_msg and pred_res is not None:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(gold_sql)
                gold_res = cur.fetchall()
            if Counter(gold_res) == Counter(pred_res):
                is_correct = True
        except Exception as e:
            final_error_msg = str(e)

    print(f"[{index + 1}] {'✅' if is_correct else '❌'}")
    return {
        "order_index": index, "question_id": question_id, "db_id": db_id, "question": question,
        "gold_sql": gold_sql, "predicted_sql": predicted_sql,
        "is_correct": is_correct, "error_msg": final_error_msg
    }

def run_unified_evaluation(
    mode,
    dataset_path="data/mini_dev.json",
    test_limit=None,
    include_dataset_evidence=True,
    candidate_count=EVAL_CANDIDATE_COUNT,
    save_path=None,
):
    print(f"\n🚀 [终极评测引擎] 雷霆出击！所有历史成绩全量作废，开启【沙盒自修正 (Reflexion)】进行最为纯净的从零评估！ 当前挂载模式: 【{mode.upper()}】")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if test_limit:
        data = data[:test_limit]

    # fail-fast：在启动并发前先验证关键配置
    try:
        get_chat_client()
    except Exception as e:
        print(f"❌ 模型客户端初始化失败，评测已中止：{e}")
        return

    official_knowledge_index = {}
    if mode in ("official", "fused") and os.path.exists("data/dev_tied_append.json"):
        with open("data/dev_tied_append.json", "r", encoding="utf-8") as f:
            official_knowledge_index = build_official_knowledge_index(json.load(f))

    # 准备多线程参数
    args_list = []
    official_match_count = 0
    official_evidence_count = 0
    official_match_types = defaultdict(int)
    for i, tc in enumerate(data):
        official_match = resolve_official_knowledge(tc, official_knowledge_index)
        official_evidence = ""
        if official_match:
            official_match_count += 1
            official_match_types[official_match["match_type"]] += 1
            official_evidence = official_match["evidence"]
            if official_evidence:
                official_evidence_count += 1
        args_list.append((i, tc, mode, official_evidence, include_dataset_evidence, candidate_count))

    if mode in ("official", "fused") and official_knowledge_index:
        match_summary = ", ".join(
            f"{match_type}={count}" for match_type, count in sorted(official_match_types.items())
        ) or "无"
        print(
            f"📚 官方知识命中 {official_match_count} / {len(data)} 题，"
            f"其中成功注入非空 Evidence [1] {official_evidence_count} 题。"
        )
        print(f"🧭 匹配来源分布：{match_summary}")
    
    results_log = []
    correct_count = 0
    total_processed = len(data)

    print(
        f"📦 开始通过并发池高速测试全量 {len(args_list)} 个查询 "
        f"(候选数={max(1, candidate_count)}, hints={'ON' if include_dataset_evidence else 'OFF'})..."
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_case, args): args for args in args_list}
        
        for future in concurrent.futures.as_completed(futures):
            index, test_case, _, _ = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "order_index": index,
                    "question_id": test_case.get("question_id", index),
                    "db_id": test_case.get("db_id", ""),
                    "question": test_case.get("question", ""),
                    "gold_sql": test_case.get("SQL", ""),
                    "predicted_sql": "ERROR",
                    "is_correct": False,
                    "error_msg": f"UNCAUGHT_WORKER_EXCEPTION: {e}",
                }
            results_log.append(res)
            if res["is_correct"]:
                correct_count += 1
                
    # 按原题目顺序排序，不然看起来很混乱！
    results_log.sort(key=lambda x: x["order_index"])
    output_log = []
    for item in results_log:
        row = dict(item)
        row.pop("order_index", None)
        output_log.append(row)

    # 统一存盘
    if save_path is None:
        save_path = f"data/evaluation_results_{mode}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_log, f, ensure_ascii=False, indent=2)

    summary = {
        "mode": mode,
        "dataset_path": dataset_path,
        "include_dataset_evidence": bool(include_dataset_evidence),
        "candidate_count": int(max(1, candidate_count)),
        "total": int(total_processed),
        "correct": int(correct_count),
        "accuracy": float((correct_count / total_processed) * 100) if total_processed else 0.0,
        "save_path": save_path,
    }

    if total_processed > 0:
        final_acc = (correct_count / total_processed) * 100
        print("\n" + "=" * 60)
        print(f"🏆 【{mode.upper()} 组】学术消融实验收官战报！")
        print(f"📈 最终 EX 准确率 (Execution Acc): {final_acc:.2f}%")
        print(f"✅ 成功跑通数目: {correct_count} / {total_processed}")
        print("=" * 60)
    return summary
