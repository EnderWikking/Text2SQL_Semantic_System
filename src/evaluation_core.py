import os
import json
import sqlite3
import re
import time
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from knowledge_base_paths import describe_active_kb, get_metadata_cache_dir
from provider_clients import get_chat_client, get_chat_model
DEFAULT_MAX_WORKERS = max(1, int(os.getenv("EVAL_MAX_WORKERS", "8")))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "4")))
LLM_API_RETRY_MAX = max(1, int(os.getenv("LLM_API_RETRY_MAX", "3")))
LLM_API_RETRY_BASE_SECONDS = max(0.1, float(os.getenv("LLM_API_RETRY_BASE_SECONDS", "1.5")))
EVAL_CANDIDATE_COUNT = max(1, int(os.getenv("EVAL_CANDIDATE_COUNT", "1")))
EVAL_INCLUDE_LONG_PROFILE = os.getenv("EVAL_INCLUDE_LONG_PROFILE", "0").strip() == "1"
_API_CALL_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)
VALID_SAVE_STRATEGIES = {"timestamp", "increment", "overwrite"}


def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text):
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _next_incremental_path(path):
    path = Path(path)
    if not path.exists():
        return str(path)
    for idx in range(2, 100000):
        candidate = path.with_name(f"{path.stem}_v{idx}{path.suffix}")
        if not candidate.exists():
            return str(candidate)
    raise RuntimeError(f"无法为输出文件分配唯一名称: {path}")


def _safe_percent(numerator, denominator):
    return float((numerator / denominator) * 100.0) if denominator else 0.0


def _summary_path_for_result(result_path):
    p = Path(result_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}.summary{p.suffix}"))
    return str(p.with_name(f"{p.name}.summary.json"))


def _write_json_file(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_jsonl(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def persist_evaluation_summary(summary):
    runs_root = Path(os.getenv("EVAL_RUNS_DIR", "data/evaluation_runs"))
    runs_root.mkdir(parents=True, exist_ok=True)
    _append_jsonl(runs_root / "history.jsonl", summary)

    mode_slug = _slug(summary.get("mode", "unknown"))
    _write_json_file(runs_root / "latest.json", summary)
    _write_json_file(runs_root / f"latest_{mode_slug}.json", summary)


def resolve_evaluation_save_path(
    mode,
    dataset_path,
    test_limit,
    include_dataset_evidence,
    candidate_count,
    save_path=None,
    run_dir=None,
    run_tag=None,
):
    save_strategy = os.getenv("EVAL_SAVE_STRATEGY", "timestamp").strip().lower()
    if save_strategy not in VALID_SAVE_STRATEGIES:
        save_strategy = "timestamp"

    tag = run_tag or _now_tag()
    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if save_strategy == "overwrite":
            return str(target), save_strategy
        if save_strategy == "increment":
            return _next_incremental_path(target), save_strategy
        return str(target.with_name(f"{target.stem}_{tag}{target.suffix or '.json'}")), save_strategy

    runs_root = Path(run_dir) if run_dir else Path(os.getenv("EVAL_RUNS_DIR", "data/evaluation_runs"))
    runs_root.mkdir(parents=True, exist_ok=True)

    dataset_tag = _slug(Path(dataset_path).stem)
    mode_tag = _slug(mode)
    hints_tag = "hints" if include_dataset_evidence else "nohints"
    limit_tag = f"n{int(test_limit)}" if test_limit else "full"
    cand_tag = f"c{max(1, int(candidate_count))}"
    filename = f"evaluation_results_{mode_tag}_{dataset_tag}_{hints_tag}_{limit_tag}_{cand_tag}_{tag}.json"
    target = runs_root / filename

    if save_strategy == "overwrite":
        return str(target), save_strategy
    if save_strategy == "increment":
        return _next_incremental_path(target), save_strategy
    if target.exists():
        return _next_incremental_path(target), save_strategy
    return str(target), save_strategy

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


def compact_semantic_for_prompt(semantic):
    if not isinstance(semantic, dict):
        return semantic

    compact = {
        "table_summary": semantic.get("table_summary", ""),
        "join_hints": semantic.get("join_hints", ""),
        "columns_semantic": {},
    }

    columns_semantic = semantic.get("columns_semantic", {})
    if not isinstance(columns_semantic, dict):
        columns_semantic = {}

    for col_name, col_info in columns_semantic.items():
        if not isinstance(col_info, dict):
            continue

        description = (
            col_info.get("description")
            or col_info.get("short_description")
            or col_info.get("long_description")
            or ""
        )
        entry = {
            "description": description,
            "sql_format_constraints": col_info.get("sql_format_constraints", ""),
        }
        literal_hints = col_info.get("literal_hints", "")
        if literal_hints:
            entry["literal_hints"] = literal_hints
        if EVAL_INCLUDE_LONG_PROFILE and col_info.get("long_description"):
            entry["long_description"] = col_info.get("long_description")

        compact["columns_semantic"][col_name] = entry

    return compact


def keyword_tokens(text):
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", (text or "").lower())
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "what", "which",
        "who", "whom", "whose", "when", "where", "into", "than", "then", "have",
        "has", "had", "been", "were", "was", "are", "is", "does", "did", "how",
        "much", "many", "more", "most", "least", "all", "not", "out", "list",
        "show", "give", "tell", "find", "year", "years", "month", "months",
        "between", "against", "among", "each", "per", "total", "average",
    }
    return {token for token in raw_tokens if len(token) >= 3 and token not in stopwords}


def _field_text_score(query_tokens, *texts):
    haystack = " ".join(str(text or "").lower() for text in texts)
    if not haystack:
        return 0
    return sum(1 for token in query_tokens if token in haystack)


def select_semantic_context(full_meta, question, evidence, max_tables=4, max_columns_per_table=4):
    if not isinstance(full_meta, dict):
        return []

    query_tokens = keyword_tokens(question) | keyword_tokens(evidence)
    if not query_tokens:
        query_tokens = keyword_tokens(question)

    table_candidates = []
    for table_name, info in full_meta.items():
        if str(table_name).startswith("__") or not isinstance(info, dict):
            continue

        sem = compact_semantic_for_prompt(info.get("semantic_description", {}))
        if not isinstance(sem, dict):
            continue

        table_summary = (sem.get("table_summary") or "").strip()
        join_hints = (sem.get("join_hints") or "").strip()
        columns_semantic = sem.get("columns_semantic", {}) or {}

        scored_columns = []
        for col_name, col_info in columns_semantic.items():
            if not isinstance(col_info, dict):
                continue

            description = (col_info.get("description") or "").strip()
            constraints = (col_info.get("sql_format_constraints") or "").strip()
            literal_hints = (col_info.get("literal_hints") or "").strip()
            score = _field_text_score(
                query_tokens,
                table_name,
                col_name,
                description,
                constraints,
                literal_hints,
            )
            if score > 0:
                scored_columns.append(
                    {
                        "name": col_name,
                        "description": description,
                        "constraints": constraints,
                        "literal_hints": literal_hints,
                        "score": score,
                    }
                )

        table_score = _field_text_score(query_tokens, table_name, table_summary, join_hints)
        if scored_columns:
            table_score += max(item["score"] for item in scored_columns)

        if table_score > 0:
            scored_columns.sort(key=lambda item: (-item["score"], item["name"]))
            table_candidates.append(
                {
                    "table_name": table_name,
                    "table_summary": table_summary,
                    "join_hints": join_hints,
                    "score": table_score,
                    "columns": scored_columns[:max_columns_per_table],
                }
            )

    table_candidates.sort(key=lambda item: (-item["score"], item["table_name"]))
    return table_candidates[:max_tables]


def render_semantic_context(selected_tables):
    if not selected_tables:
        return ""

    lines = []
    for table in selected_tables:
        header = table["table_name"]
        if table.get("table_summary"):
            header += f": {table['table_summary']}"
        lines.append(header)

        if table.get("join_hints"):
            lines.append(f"JOIN 提示: {table['join_hints']}")

        for col in table.get("columns", []):
            parts = [f"字段 {col['name']}"]
            if col.get("description"):
                parts.append(col["description"])
            if col.get("constraints"):
                parts.append(f"SQL 约束: {col['constraints']}")
            if col.get("literal_hints"):
                parts.append(f"字面量提示: {col['literal_hints']}")
            lines.append("；".join(parts))

    return "\n".join(lines)


def extract_referenced_tables(sql_text):
    if not sql_text or sql_text == "ERROR":
        return []

    matches = re.findall(
        r"\b(?:from|join|update|into)\s+[`\"]?([A-Za-z_][A-Za-z0-9_]*)[`\"]?",
        sql_text,
        flags=re.IGNORECASE,
    )
    seen = []
    for name in matches:
        if name not in seen:
            seen.append(name)
    return seen


def build_repair_schema_context(ddl_by_table, predicted_sql):
    if not ddl_by_table:
        return ""

    referenced_tables = extract_referenced_tables(predicted_sql)
    snippets = [ddl_by_table[name] for name in referenced_tables if name in ddl_by_table]
    return "\n\n".join(snippets[:4])


def build_repair_messages(system_content, question, predicted_sql, error_msg, repair_schema_context):
    repair_prompt_parts = [
        "你在修复一条执行失败的 SQLite SQL。",
        f"【原问题】:\n{question}",
        f"【上一版 SQL】:\n{predicted_sql}",
        f"【SQLite 报错】:\n{error_msg}",
    ]
    if repair_schema_context:
        repair_prompt_parts.append(f"【相关表结构】:\n{repair_schema_context}")
    repair_prompt_parts.append("请只根据这些信息修复 SQL，直接返回修正后的 ```sql 代码块，不要解释。")
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(repair_prompt_parts)},
    ]

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
            schema_rows = (
                conn.cursor()
                .execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
                .fetchall()
            )
            ddl_by_table = {name: sql for name, sql in schema_rows if name and sql}
            raw_ddl = "\n".join(sql for _, sql in schema_rows if sql)
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
        cache_path = os.path.join(get_metadata_cache_dir(), f"{db_id}_enhanced.json")
        schema_knowledge = ""
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                full_meta = json.load(f)
            selected_tables = select_semantic_context(full_meta, question, evidence)
            schema_knowledge = render_semantic_context(selected_tables)
        
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
                    repair_schema_context = build_repair_schema_context(ddl_by_table, predicted_sql)
                    messages = build_repair_messages(
                        system_content=system_content,
                        question=question,
                        predicted_sql=predicted_sql,
                        error_msg=final_error_msg,
                        repair_schema_context=repair_schema_context,
                    )
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
    run_dir=None,
    run_tag=None,
):
    print(f"\n🚀 [终极评测引擎] 雷霆出击！所有历史成绩全量作废，开启【沙盒自修正 (Reflexion)】进行最为纯净的从零评估！ 当前挂载模式: 【{mode.upper()}】")
    if mode in ("enhanced", "fused"):
        print(f"🧱 当前知识库命名空间: {describe_active_kb()} -> {get_metadata_cache_dir()}")
    started_at = datetime.now()
    started_ts = time.time()
    run_tag_in_use = run_tag or _now_tag()
    resolved_save_path, save_strategy = resolve_evaluation_save_path(
        mode=mode,
        dataset_path=dataset_path,
        test_limit=test_limit,
        include_dataset_evidence=include_dataset_evidence,
        candidate_count=candidate_count,
        save_path=save_path,
        run_dir=run_dir,
        run_tag=run_tag_in_use,
    )
    summary_path = _summary_path_for_result(resolved_save_path)
    print(f"📝 结果将写入: {resolved_save_path} (strategy={save_strategy})")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if test_limit:
        data = data[:test_limit]

    # fail-fast：在启动并发前先验证关键配置
    try:
        get_chat_client()
    except Exception as e:
        print(f"❌ 模型客户端初始化失败，评测已中止：{e}")
        ended_at = datetime.now()
        failed_summary = {
            "mode": mode,
            "run_tag": run_tag_in_use,
            "started_at": started_at.isoformat(timespec="seconds"),
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "duration_seconds": round(time.time() - started_ts, 2),
            "dataset_path": dataset_path,
            "include_dataset_evidence": bool(include_dataset_evidence),
            "candidate_count": int(max(1, candidate_count)),
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "executable_count": 0,
            "execution_success_rate": 0.0,
            "error_count": 0,
            "save_path": resolved_save_path,
            "summary_path": summary_path,
            "save_strategy": save_strategy,
            "error": f"MODEL_INIT_FAILED: {e}",
        }
        _write_json_file(summary_path, failed_summary)
        persist_evaluation_summary(failed_summary)
        return failed_summary

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
            future_args = futures[future]
            index, test_case = future_args[0], future_args[1]
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
    with open(resolved_save_path, "w", encoding="utf-8") as f:
        json.dump(output_log, f, ensure_ascii=False, indent=2)

    error_count = sum(1 for row in output_log if row.get("error_msg"))
    generation_error_count = sum(1 for row in output_log if row.get("predicted_sql") == "ERROR")
    execution_error_count = sum(
        1 for row in output_log if row.get("predicted_sql") != "ERROR" and row.get("error_msg")
    )
    executable_count = sum(
        1 for row in output_log if row.get("predicted_sql") != "ERROR" and not row.get("error_msg")
    )
    ended_at = datetime.now()

    summary = {
        "mode": mode,
        "run_tag": run_tag_in_use,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started_ts, 2),
        "dataset_path": dataset_path,
        "test_limit": int(test_limit) if test_limit else None,
        "include_dataset_evidence": bool(include_dataset_evidence),
        "candidate_count": int(max(1, candidate_count)),
        "total": int(total_processed),
        "correct": int(correct_count),
        "accuracy": float((correct_count / total_processed) * 100) if total_processed else 0.0,
        "executable_count": int(executable_count),
        "execution_success_rate": _safe_percent(executable_count, total_processed),
        "error_count": int(error_count),
        "generation_error_count": int(generation_error_count),
        "execution_error_count": int(execution_error_count),
        "save_path": resolved_save_path,
        "summary_path": summary_path,
        "save_strategy": save_strategy,
    }
    _write_json_file(summary_path, summary)
    persist_evaluation_summary(summary)

    if total_processed > 0:
        final_acc = (correct_count / total_processed) * 100
        print("\n" + "=" * 60)
        print(f"🏆 【{mode.upper()} 组】学术消融实验收官战报！")
        print(f"📈 最终 EX 准确率 (Execution Acc): {final_acc:.2f}%")
        print(f"📊 可执行 SQL 比例: {summary['execution_success_rate']:.2f}%")
        print(f"✅ 成功跑通数目: {correct_count} / {total_processed}")
        print(f"💾 结果文件: {resolved_save_path}")
        print(f"🧾 指标摘要: {summary_path}")
        print("=" * 60)
    return summary
