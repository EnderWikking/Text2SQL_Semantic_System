import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict

from knowledge_base_paths import describe_active_kb, get_metadata_cache_dir
from provider_clients import get_chat_client, get_chat_model


PROFILE_SAMPLE_VALUE_LIMIT = max(50, int(os.getenv("PROFILE_SAMPLE_VALUE_LIMIT", "300")))
PROFILE_TOP_K = max(3, int(os.getenv("PROFILE_TOP_K", "8")))
PROFILE_MINHASH_NUM_PERM = max(16, int(os.getenv("PROFILE_MINHASH_NUM_PERM", "32")))
PROFILE_MINHASH_SAMPLE_LIMIT = max(50, int(os.getenv("PROFILE_MINHASH_SAMPLE_LIMIT", "256")))
PROFILE_MINHASH_MIN_SIMILARITY = float(os.getenv("PROFILE_MINHASH_MIN_SIMILARITY", "0.45"))
PROFILE_LLM_SLEEP_SECONDS = max(0.0, float(os.getenv("PROFILE_LLM_SLEEP_SECONDS", "0.5")))
PROFILE_DISABLE_LLM = os.getenv("PROFILE_DISABLE_LLM", "0").strip() == "1"
PROFILE_VERSION = "v2.3"
PROFILE_IMPLICIT_MAX_PER_COLUMN = max(3, int(os.getenv("PROFILE_IMPLICIT_MAX_PER_COLUMN", "5")))

BOOLEAN_TEXT_VALUES = {
    "0",
    "1",
    "y",
    "n",
    "yes",
    "no",
    "true",
    "false",
    "t",
    "f",
}


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def is_internal_table(table_name):
    return str(table_name).lower().startswith("sqlite_")


def to_json_safe(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def looks_numeric_text(text):
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", text))


def detect_value_patterns(value_text):
    text = (value_text or "").strip()
    if not text:
        return ["empty"]

    patterns = []
    if looks_numeric_text(text):
        patterns.append("numeric_like")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        patterns.append("date_iso")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", text):
        patterns.append("datetime_iso")
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", text):
        patterns.append("uuid_like")
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text):
        patterns.append("email_like")
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        patterns.append("json_like")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", text):
        patterns.append("code_like")

    if not patterns:
        patterns.append("text_like")
    return patterns


def compute_value_shape_stats(samples):
    if not samples:
        return {}

    lengths = []
    pattern_counter = Counter()
    prefix_counter = Counter()
    char_class_counter = Counter()

    for raw in samples:
        text = str(raw).strip()
        if not text:
            continue

        lengths.append(len(text))
        for p in detect_value_patterns(text):
            pattern_counter[p] += 1

        if len(text) >= 3:
            prefix_counter[text[:3]] += 1
        elif text:
            prefix_counter[text] += 1

        for ch in text:
            if ch.isdigit():
                char_class_counter["digit"] += 1
            elif ch.isalpha():
                if ch.isupper():
                    char_class_counter["upper"] += 1
                elif ch.islower():
                    char_class_counter["lower"] += 1
                else:
                    char_class_counter["alpha_other"] += 1
            elif ch.isspace():
                char_class_counter["space"] += 1
            else:
                char_class_counter["punct"] += 1

    if not lengths:
        return {}

    char_total = sum(char_class_counter.values()) or 1
    sample_total = len(samples) or 1

    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": round(sum(lengths) / len(lengths), 3),
        "pattern_distribution": {
            k: round(v / sample_total, 3)
            for k, v in pattern_counter.most_common(6)
        },
        "common_prefixes": [
            {"prefix": p, "count": c}
            for p, c in prefix_counter.most_common(5)
        ],
        "char_class_ratio": {
            k: round(v / char_total, 3)
            for k, v in char_class_counter.items()
        },
    }


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def compute_numeric_stats(samples):
    nums = []
    for raw in samples:
        text = str(raw).strip()
        if looks_numeric_text(text):
            try:
                nums.append(float(text))
            except Exception:
                continue
    if not nums:
        return {}

    nums.sort()
    count = len(nums)
    mean = sum(nums) / count
    variance = sum((x - mean) ** 2 for x in nums) / count
    return {
        "count": count,
        "min": nums[0],
        "max": nums[-1],
        "mean": round(mean, 6),
        "std": round(variance**0.5, 6),
        "p10": round(percentile(nums, 0.10), 6),
        "p50": round(percentile(nums, 0.50), 6),
        "p90": round(percentile(nums, 0.90), 6),
    }


def compute_date_stats(samples):
    date_like = []
    datetime_like = []
    for raw in samples:
        text = str(raw).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            date_like.append(text)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", text):
            datetime_like.append(text)

    if not date_like and not datetime_like:
        return {}

    combined = sorted(date_like + datetime_like)
    return {
        "min": combined[0],
        "max": combined[-1],
        "has_time_component": bool(datetime_like),
        "sample_count": len(combined),
    }


def entropy_from_top_values(top_values):
    import math

    counts = [int(item.get("count", 0) or 0) for item in top_values if int(item.get("count", 0) or 0) > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        p = c / total
        entropy -= p * math.log2(p)
    return round(entropy, 6)


def classify_cardinality(distinct_ratio, distinct_count):
    if distinct_count <= 1:
        return "constant"
    if distinct_ratio <= 0.02:
        return "very_low"
    if distinct_ratio <= 0.10:
        return "low"
    if distinct_ratio <= 0.50:
        return "medium"
    return "high"


def infer_semantic_type(col_name, declared_type, shape_stats, top_values):
    name = str(col_name).lower()
    dtype = str(declared_type or "").lower()
    patterns = (shape_stats or {}).get("pattern_distribution", {})
    pattern_keys = set(patterns.keys())
    top_vals = [str(item.get("value", "")).strip().lower() for item in top_values[:5]]

    if "json" in dtype or "json_like" in pattern_keys:
        return "json"
    if "date" in dtype or "time" in dtype or "date_iso" in pattern_keys or "datetime_iso" in pattern_keys:
        return "datetime"
    if name.endswith("_id") or name == "id" or (name.endswith("id") and "code_like" in pattern_keys):
        return "identifier"
    if "int" in dtype or "real" in dtype or "numeric" in dtype or "decimal" in dtype:
        return "numeric"
    if "numeric_like" in pattern_keys and ("char" not in dtype and "text" not in dtype):
        return "numeric"
    if top_vals and all(v in BOOLEAN_TEXT_VALUES for v in top_vals if v != ""):
        return "boolean"
    if "char" in dtype or "text" in dtype or "clob" in dtype:
        return "text"
    return "unknown"


def infer_sql_format_constraints(semantic_type, declared_type, pattern_keys):
    dtype = str(declared_type or "").lower()

    if semantic_type == "numeric":
        if "char" in dtype or "text" in dtype:
            return "该列语义为数值但声明为文本，建议比较前 CAST(... AS REAL)。"
        return "该列为数值语义，比较时通常不加引号。"
    if semantic_type == "boolean":
        return "该列疑似布尔语义，常用取值 0/1 或 true/false。"
    if semantic_type == "datetime" or "date_iso" in pattern_keys or "datetime_iso" in pattern_keys:
        return "该列疑似日期/时间，建议使用 strftime 或日期范围过滤。"
    if semantic_type == "json":
        return "该列疑似 JSON 文本，通常不用于等值 JOIN。"
    if semantic_type == "identifier":
        return "该列疑似标识符，JOIN/过滤时保持与目标列类型一致。"
    return "文本比较时请使用单引号。"


def normalize_join_token(raw):
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if len(text) > 160:
        text = text[:160]
    return text


def is_identifier_like_name(name):
    n = str(name or "").strip().lower()
    if not n:
        return False
    patterns = [
        r"(^id$)",
        r"(_id$)",
        r"(id$)",
        r"(^uuid$|_uuid$|uuid$)",
        r"(^code$|_code$|code$)",
        r"(^key$|_key$|key$)",
    ]
    return any(re.search(p, n) for p in patterns)


def compute_sample_overlap_metrics(sample_set_a, sample_set_b):
    if not sample_set_a or not sample_set_b:
        return {
            "intersect_count": 0,
            "jaccard": 0.0,
            "contain_a_in_b": 0.0,
            "contain_b_in_a": 0.0,
        }

    inter = len(sample_set_a & sample_set_b)
    union = len(sample_set_a | sample_set_b)
    return {
        "intersect_count": inter,
        "jaccard": round(inter / union, 4) if union else 0.0,
        "contain_a_in_b": round(inter / len(sample_set_a), 4) if sample_set_a else 0.0,
        "contain_b_in_a": round(inter / len(sample_set_b), 4) if sample_set_b else 0.0,
    }


def semantic_types_compatible(type_a, type_b):
    a = str(type_a or "unknown")
    b = str(type_b or "unknown")
    if a == "unknown" or b == "unknown":
        return True
    if a == b:
        return True
    compatible_pairs = {
        ("identifier", "numeric"),
        ("numeric", "identifier"),
        ("identifier", "text"),
        ("text", "identifier"),
        ("boolean", "numeric"),
        ("numeric", "boolean"),
    }
    return (a, b) in compatible_pairs


def relation_confidence(score):
    if score >= 0.72:
        return "high"
    if score >= 0.58:
        return "medium"
    return "low"


def stable_hash64(text, seed):
    payload = f"{seed}::{text}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big", signed=False)


def compute_minhash(values, num_perm=PROFILE_MINHASH_NUM_PERM):
    if not values:
        return None

    max_u64 = (1 << 64) - 1
    signature = [max_u64] * num_perm
    deduped_values = list(dict.fromkeys(str(v) for v in values if str(v).strip()))

    for val in deduped_values:
        for seed in range(num_perm):
            hv = stable_hash64(val, seed)
            if hv < signature[seed]:
                signature[seed] = hv

    return signature


def minhash_similarity(sig_a, sig_b):
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    equal = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return equal / len(sig_a)


def name_relation(column_a, column_b):
    a = str(column_a).lower()
    b = str(column_b).lower()

    if a == b:
        return 0.25, "same_name"

    if a.endswith("id") and b.endswith("id"):
        return 0.12, "both_id_like"

    tokens_a = set(re.findall(r"[a-z0-9]+", a))
    tokens_b = set(re.findall(r"[a-z0-9]+", b))
    overlap = tokens_a & tokens_b
    if overlap:
        return 0.08, f"token_overlap:{','.join(sorted(list(overlap))[:3])}"

    return 0.0, "none"


def fetch_scalar(cursor, sql, params=()):
    row = cursor.execute(sql, params).fetchone()
    if not row:
        return None
    return row[0]


def fetch_column_samples(cursor, table, col, sample_limit=PROFILE_SAMPLE_VALUE_LIMIT):
    q_table = quote_identifier(table)
    q_col = quote_identifier(col)
    sql = f"SELECT {q_col} FROM {q_table} WHERE {q_col} IS NOT NULL LIMIT ?"
    rows = cursor.execute(sql, (sample_limit,)).fetchall()
    return [to_json_safe(row[0]) for row in rows if row and row[0] is not None]


def fetch_top_values(cursor, table, col, top_k=PROFILE_TOP_K):
    q_table = quote_identifier(table)
    q_col = quote_identifier(col)
    sql = (
        f"SELECT {q_col} AS v, COUNT(*) AS c "
        f"FROM {q_table} WHERE {q_col} IS NOT NULL "
        f"GROUP BY {q_col} ORDER BY c DESC LIMIT ?"
    )
    rows = cursor.execute(sql, (top_k,)).fetchall()
    return [{"value": to_json_safe(v), "count": int(c)} for v, c in rows]


def fetch_table_indexes(cursor, table):
    q_table = quote_identifier(table)
    index_rows = cursor.execute(f"PRAGMA index_list({q_table})").fetchall()
    indexes = []
    unique_single_columns = set()

    for row in index_rows:
        # SQLite index_list shape: seq, name, unique, origin, partial
        index_name = row[1] if len(row) > 1 else ""
        is_unique = bool(int(row[2])) if len(row) > 2 else False
        origin = row[3] if len(row) > 3 else ""
        partial = bool(int(row[4])) if len(row) > 4 else False

        if not index_name:
            continue

        col_rows = cursor.execute(f"PRAGMA index_info({quote_identifier(index_name)})").fetchall()
        cols = [c[2] for c in col_rows if len(c) > 2 and c[2]]
        if is_unique and len(cols) == 1:
            unique_single_columns.add(cols[0])

        indexes.append(
            {
                "name": index_name,
                "unique": is_unique,
                "origin": origin,
                "partial": partial,
                "columns": cols,
            }
        )

    return indexes, unique_single_columns


def build_column_profile(cursor, table, col_name, declared_type, row_count, is_primary_key=False, is_unique_index=False):
    q_table = quote_identifier(table)
    q_col = quote_identifier(col_name)

    sql_counts = (
        f"SELECT "
        f"SUM(CASE WHEN {q_col} IS NULL THEN 1 ELSE 0 END) AS null_count, "
        f"COUNT(DISTINCT {q_col}) AS distinct_count "
        f"FROM {q_table}"
    )
    null_count, distinct_count = cursor.execute(sql_counts).fetchone()
    null_count = int(null_count or 0)
    distinct_count = int(distinct_count or 0)
    non_null_count = max(0, int(row_count) - null_count)

    min_val = fetch_scalar(cursor, f"SELECT MIN({q_col}) FROM {q_table}")
    max_val = fetch_scalar(cursor, f"SELECT MAX({q_col}) FROM {q_table}")
    samples = fetch_column_samples(cursor, table, col_name, PROFILE_SAMPLE_VALUE_LIMIT)
    top_values = fetch_top_values(cursor, table, col_name, PROFILE_TOP_K)

    text_samples = [str(x) for x in samples if x is not None]
    shape_stats = compute_value_shape_stats(text_samples)
    numeric_stats = compute_numeric_stats(text_samples)
    date_stats = compute_date_stats(text_samples)

    distinct_ratio = round(distinct_count / row_count, 4) if row_count else 0.0
    non_null_distinct_ratio = round(distinct_count / non_null_count, 4) if non_null_count else 0.0
    null_ratio = round(null_count / row_count, 4) if row_count else 0.0
    cardinality_class = classify_cardinality(distinct_ratio, distinct_count)
    semantic_type = infer_semantic_type(col_name, declared_type, shape_stats, top_values)
    pattern_keys = set((shape_stats or {}).get("pattern_distribution", {}).keys())

    top_value_total = sum(int(item.get("count", 0) or 0) for item in top_values)
    top_value_coverage = round(top_value_total / non_null_count, 4) if non_null_count else 0.0
    dominant_value_ratio = (
        round(int(top_values[0].get("count", 0) or 0) / non_null_count, 4)
        if top_values and non_null_count
        else 0.0
    )
    value_entropy = entropy_from_top_values(top_values)

    quality_flags = []
    if row_count > 0 and null_ratio >= 0.8:
        quality_flags.append("mostly_null")
    if non_null_count > 0 and distinct_count <= 1:
        quality_flags.append("constant_like")
    if cardinality_class in ("very_low", "low") and dominant_value_ratio >= 0.6:
        quality_flags.append("highly_skewed")

    id_like = is_identifier_like_name(col_name)
    key_score = 0.0
    if is_primary_key:
        key_score += 0.7
    if is_unique_index:
        key_score += 0.45
    if non_null_count > 0 and distinct_count == non_null_count:
        key_score += 0.3
    if id_like:
        key_score += 0.12
    if semantic_type == "identifier":
        key_score += 0.12
    if null_count > 0:
        key_score -= 0.15
    key_score = max(0.0, min(1.0, round(key_score, 4)))
    likely_key = key_score >= 0.52

    profile = {
        "declared_type": declared_type or "",
        "is_primary_key": bool(is_primary_key),
        "is_unique_index": bool(is_unique_index),
        "nulls": null_count,
        "non_nulls": non_null_count,
        "null_ratio": null_ratio,
        "distinct": distinct_count,
        "distinct_ratio": distinct_ratio,
        "non_null_distinct_ratio": non_null_distinct_ratio,
        "cardinality_class": cardinality_class,
        "inferred_semantic_type": semantic_type,
        "likely_key": likely_key,
        "key_score": key_score,
        "min": to_json_safe(min_val),
        "max": to_json_safe(max_val),
        "top_values": top_values,
        "top_value_coverage": top_value_coverage,
        "dominant_value_ratio": dominant_value_ratio,
        "top_value_entropy": value_entropy,
        "samples": samples[:8],
        "value_shape": shape_stats,
        "numeric_stats": numeric_stats,
        "date_stats": date_stats,
        "sql_format_constraints": infer_sql_format_constraints(semantic_type, declared_type, pattern_keys),
        "quality_flags": quality_flags,
        "_sample_for_minhash": text_samples[:PROFILE_MINHASH_SAMPLE_LIMIT],
    }
    return profile


def build_table_profile(cursor, table):
    q_table = quote_identifier(table)

    row_count = int(fetch_scalar(cursor, f"SELECT COUNT(*) FROM {q_table}") or 0)

    col_rows = cursor.execute(f"PRAGMA table_info({q_table})").fetchall()
    primary_keys = [row[1] for row in col_rows if int(row[5] or 0) > 0]

    fk_rows = cursor.execute(f"PRAGMA foreign_key_list({q_table})").fetchall()
    foreign_keys = []
    for fk in fk_rows:
        foreign_keys.append(
            {
                "from_column": fk[3],
                "to_table": fk[2],
                "to_column": fk[4],
            }
        )
    indexes, unique_single_columns = fetch_table_indexes(cursor, table)

    columns = {}
    for row in col_rows:
        col_name = row[1]
        declared_type = row[2] or ""
        try:
            columns[col_name] = build_column_profile(
                cursor,
                table,
                col_name,
                declared_type,
                row_count,
                is_primary_key=(col_name in primary_keys),
                is_unique_index=(col_name in unique_single_columns),
            )
        except Exception as e:
            columns[col_name] = {
                "declared_type": declared_type,
                "error": str(e),
                "samples": [],
                "top_values": [],
                "value_shape": {},
                "_sample_for_minhash": [],
            }

    return {
        "row_count": row_count,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "indexes": indexes,
        "columns": columns,
    }


def detect_implicit_join_candidates(db_profile):
    explicit_links = set()
    for src_table, table_info in db_profile.items():
        for fk in table_info.get("foreign_keys", []):
            explicit_links.add(
                (
                    str(src_table).lower(),
                    str(fk.get("from_column", "")).lower(),
                    str(fk.get("to_table", "")).lower(),
                    str(fk.get("to_column", "")).lower(),
                )
            )

    entries = []
    for table_name, table_info in db_profile.items():
        cols = table_info.get("columns", {})
        for col_name, col_info in cols.items():
            samples = col_info.get("_sample_for_minhash", [])
            non_nulls = int(col_info.get("non_nulls", 0) or 0)
            distinct = int(col_info.get("distinct", 0) or 0)

            if non_nulls < 20 or distinct < 5 or len(samples) < 20:
                continue

            signature = compute_minhash(samples, PROFILE_MINHASH_NUM_PERM)
            if not signature:
                continue

            normalized_tokens = {
                normalize_join_token(x)
                for x in samples[:PROFILE_MINHASH_SAMPLE_LIMIT]
                if normalize_join_token(x)
            }
            if len(normalized_tokens) < 15:
                continue

            entries.append(
                {
                    "table": table_name,
                    "column": col_name,
                    "qualified": f"{table_name}.{col_name}",
                    "signature": signature,
                    "semantic_type": col_info.get("inferred_semantic_type", "unknown"),
                    "distinct_ratio": float(col_info.get("distinct_ratio", 0.0) or 0.0),
                    "likely_key": bool(col_info.get("likely_key", False)),
                    "key_score": float(col_info.get("key_score", 0.0) or 0.0),
                    "cardinality_class": col_info.get("cardinality_class", "unknown"),
                    "sample_set": normalized_tokens,
                }
            )

    adjacency = defaultdict(list)

    for i in range(len(entries)):
        left = entries[i]
        for j in range(i + 1, len(entries)):
            right = entries[j]
            if left["table"] == right["table"]:
                continue

            if not semantic_types_compatible(left.get("semantic_type"), right.get("semantic_type")):
                continue

            sim = minhash_similarity(left["signature"], right["signature"])
            overlap = compute_sample_overlap_metrics(left["sample_set"], right["sample_set"])
            bonus_lr, relation_lr = name_relation(left["column"], right["column"])
            bonus_rl, relation_rl = name_relation(right["column"], left["column"])

            signal = max(sim, overlap["contain_a_in_b"], overlap["contain_b_in_a"], overlap["jaccard"])
            if signal < 0.28 and max(bonus_lr, bonus_rl) < 0.12:
                continue

            def directional_prior(src, dst):
                prior = 0.0
                if is_identifier_like_name(src.get("column")):
                    prior += 0.04
                if str(src.get("semantic_type")) == "identifier":
                    prior += 0.04
                if dst.get("likely_key"):
                    prior += 0.08
                if dst.get("key_score", 0.0) >= 0.8:
                    prior += 0.05
                if src.get("likely_key"):
                    prior -= 0.04
                if src.get("cardinality_class") in {"very_low", "low"}:
                    prior -= 0.05
                return prior

            score_lr = (
                (0.40 * sim)
                + (0.30 * overlap["contain_a_in_b"])
                + (0.12 * overlap["jaccard"])
                + bonus_lr
                + directional_prior(left, right)
            )
            score_rl = (
                (0.40 * sim)
                + (0.30 * overlap["contain_b_in_a"])
                + (0.12 * overlap["jaccard"])
                + bonus_rl
                + directional_prior(right, left)
            )

            if score_lr >= PROFILE_MINHASH_MIN_SIMILARITY:
                key = (
                    str(left["table"]).lower(),
                    str(left["column"]).lower(),
                    str(right["table"]).lower(),
                    str(right["column"]).lower(),
                )
                if key not in explicit_links:
                    score = round(score_lr, 4)
                    adjacency[left["qualified"]].append(
                        {
                            "to_table": right["table"],
                            "to_column": right["column"],
                            "score": score,
                            "confidence": relation_confidence(score),
                            "minhash_similarity": round(sim, 4),
                            "sample_overlap_jaccard": overlap["jaccard"],
                            "sample_containment": overlap["contain_a_in_b"],
                            "reverse_containment": overlap["contain_b_in_a"],
                            "intersect_count": overlap["intersect_count"],
                            "name_relation": relation_lr,
                            "target_likely_key": bool(right.get("likely_key")),
                            "target_semantic_type": right.get("semantic_type", "unknown"),
                        }
                    )

            if score_rl >= PROFILE_MINHASH_MIN_SIMILARITY:
                key = (
                    str(right["table"]).lower(),
                    str(right["column"]).lower(),
                    str(left["table"]).lower(),
                    str(left["column"]).lower(),
                )
                if key not in explicit_links:
                    score = round(score_rl, 4)
                    adjacency[right["qualified"]].append(
                        {
                            "to_table": left["table"],
                            "to_column": left["column"],
                            "score": score,
                            "confidence": relation_confidence(score),
                            "minhash_similarity": round(sim, 4),
                            "sample_overlap_jaccard": overlap["jaccard"],
                            "sample_containment": overlap["contain_b_in_a"],
                            "reverse_containment": overlap["contain_a_in_b"],
                            "intersect_count": overlap["intersect_count"],
                            "name_relation": relation_rl,
                            "target_likely_key": bool(left.get("likely_key")),
                            "target_semantic_type": left.get("semantic_type", "unknown"),
                        }
                    )

    for qualified, candidates in adjacency.items():
        from_table, from_column = qualified.split(".", 1)
        dedup = {}
        for item in candidates:
            key = (item.get("to_table"), item.get("to_column"))
            old = dedup.get(key)
            if not old or float(item.get("score", 0.0) or 0.0) > float(old.get("score", 0.0) or 0.0):
                dedup[key] = item
        candidates = list(dedup.values())
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[:PROFILE_IMPLICIT_MAX_PER_COLUMN]
        table_info = db_profile.get(from_table)
        if not table_info:
            continue
        table_info.setdefault("implicit_join_candidates", []).append(
            {
                "from_column": from_column,
                "candidates": best,
            }
        )
        table_info["implicit_join_candidates"].sort(key=lambda x: x.get("from_column", ""))


def cleanup_private_profile_fields(db_profile):
    for table_info in db_profile.values():
        for col_info in table_info.get("columns", {}).values():
            col_info.pop("_sample_for_minhash", None)


def build_llm_profile_payload(table_profile):
    compact = {
        "row_count": table_profile.get("row_count", 0),
        "primary_keys": table_profile.get("primary_keys", []),
        "foreign_keys": table_profile.get("foreign_keys", []),
        "indexes": table_profile.get("indexes", [])[:10],
        "implicit_join_candidates": table_profile.get("implicit_join_candidates", [])[:8],
        "columns": {},
    }

    for col_name, col_info in table_profile.get("columns", {}).items():
        compact["columns"][col_name] = {
            "declared_type": col_info.get("declared_type", ""),
            "nulls": col_info.get("nulls", 0),
            "non_nulls": col_info.get("non_nulls", 0),
            "distinct": col_info.get("distinct", 0),
            "distinct_ratio": col_info.get("distinct_ratio", 0.0),
            "non_null_distinct_ratio": col_info.get("non_null_distinct_ratio", 0.0),
            "cardinality_class": col_info.get("cardinality_class", "unknown"),
            "inferred_semantic_type": col_info.get("inferred_semantic_type", "unknown"),
            "likely_key": col_info.get("likely_key", False),
            "key_score": col_info.get("key_score", 0.0),
            "min": col_info.get("min"),
            "max": col_info.get("max"),
            "top_values": col_info.get("top_values", [])[:3],
            "top_value_coverage": col_info.get("top_value_coverage", 0.0),
            "dominant_value_ratio": col_info.get("dominant_value_ratio", 0.0),
            "top_value_entropy": col_info.get("top_value_entropy", 0.0),
            "value_shape": col_info.get("value_shape", {}),
            "numeric_stats": col_info.get("numeric_stats", {}),
            "date_stats": col_info.get("date_stats", {}),
            "quality_flags": col_info.get("quality_flags", []),
        }

    return compact


def build_fallback_semantic_description(table_name, table_profile):
    row_count = int(table_profile.get("row_count", 0) or 0)
    pk = table_profile.get("primary_keys", [])
    fk = table_profile.get("foreign_keys", [])
    implicit = table_profile.get("implicit_join_candidates", [])

    join_parts = []
    if fk:
        join_parts.append(
            "显式外键: " + "; ".join(
                f"{item.get('from_column')}->{item.get('to_table')}.{item.get('to_column')}"
                for item in fk[:8]
            )
        )
    if implicit:
        join_parts.append(
            "隐式关联候选: " + "; ".join(
                f"{item.get('from_column')}=>" + ", ".join(
                    f"{cand.get('to_table')}.{cand.get('to_column')}({cand.get('score')})"
                    for cand in item.get("candidates", [])[:2]
                )
                for item in implicit[:5]
            )
        )

    result = {
        "table_summary": (
            f"表 {table_name} 含 {row_count} 行，主键 {pk if pk else '无显式主键'}，"
            f"显式外键 {len(fk)} 条，隐式关联候选 {len(implicit)} 组。"
        ),
        "join_hints": " | ".join(join_parts) if join_parts else "暂无可靠 JOIN 线索，优先根据键名和样本值匹配关系。",
        "columns_semantic": {},
    }

    for col_name, col_info in table_profile.get("columns", {}).items():
        declared_type = col_info.get("declared_type", "") or "UNKNOWN"
        distinct = int(col_info.get("distinct", 0) or 0)
        nulls = int(col_info.get("nulls", 0) or 0)
        non_nulls = int(col_info.get("non_nulls", 0) or 0)
        semantic_type = col_info.get("inferred_semantic_type", "unknown")
        key_score = float(col_info.get("key_score", 0.0) or 0.0)
        key_hint = "疑似主键/连接键" if col_info.get("likely_key") else "普通属性列"
        cardinality = col_info.get("cardinality_class", "unknown")
        shape = col_info.get("value_shape", {})
        patterns = shape.get("pattern_distribution", {}) if isinstance(shape, dict) else {}
        numeric_stats = col_info.get("numeric_stats", {})
        date_stats = col_info.get("date_stats", {})
        quality_flags = col_info.get("quality_flags", [])

        format_hint = infer_sql_format_constraints(semantic_type, declared_type, set(patterns.keys()))
        if col_info.get("likely_key"):
            format_hint = f"{format_hint} 该列键值性较强，适合作为 JOIN 键。"

        top_values = col_info.get("top_values", [])
        literal_hint = ", ".join(str(item.get("value")) for item in top_values[:3]) if top_values else ""
        quality_text = f"质量标记: {', '.join(quality_flags)}。" if quality_flags else "质量标记: 正常。"
        stats_segments = []
        if numeric_stats:
            stats_segments.append(
                f"数值统计(min={numeric_stats.get('min')}, p50={numeric_stats.get('p50')}, max={numeric_stats.get('max')})"
            )
        if date_stats:
            stats_segments.append(
                f"日期范围({date_stats.get('min')} ~ {date_stats.get('max')})"
            )
        stats_text = "；".join(stats_segments) if stats_segments else "暂无稳定数值/日期统计。"

        result["columns_semantic"][col_name] = {
            "short_description": (
                f"{col_name} 列，类型 {declared_type}，语义推断={semantic_type}，"
                f"{key_hint}，cardinality={cardinality}。"
            ),
            "long_description": (
                f"{col_name} 列在 {table_name} 中用于记录业务属性；"
                f"distinct={distinct}, non_nulls={non_nulls}, nulls={nulls}, key_score={key_score}；"
                f"样式特征 {patterns}；{stats_text} {quality_text}"
            ),
            "description": (
                f"{col_name} 列，类型 {declared_type}，语义推断 {semantic_type}，"
                f"非空占比约 {round((1 - (nulls / row_count)) * 100, 2) if row_count else 0}% 。"
            ),
            "sql_format_constraints": format_hint,
            "literal_hints": literal_hint,
        }

    result["profile_quality"] = "fallback"
    return result


def extract_json_object(text):
    if not text:
        return None

    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def normalize_semantic_description(llm_payload, table_name, table_profile):
    fallback = build_fallback_semantic_description(table_name, table_profile)
    if not isinstance(llm_payload, dict):
        return fallback

    result = {
        "table_summary": str(llm_payload.get("table_summary") or fallback["table_summary"]),
        "join_hints": str(llm_payload.get("join_hints") or fallback["join_hints"]),
        "columns_semantic": {},
        "profile_quality": "llm",
    }

    llm_cols = llm_payload.get("columns_semantic", {})
    if not isinstance(llm_cols, dict):
        llm_cols = {}

    for col_name, fallback_col in fallback["columns_semantic"].items():
        raw = llm_cols.get(col_name, {})
        if not isinstance(raw, dict):
            raw = {}

        short_desc = str(raw.get("short_description") or fallback_col.get("short_description") or "")
        long_desc = str(raw.get("long_description") or fallback_col.get("long_description") or "")
        description = str(
            raw.get("description")
            or short_desc
            or long_desc
            or fallback_col.get("description")
            or ""
        )
        sql_constraints = str(raw.get("sql_format_constraints") or fallback_col.get("sql_format_constraints") or "")
        literal_hints = str(raw.get("literal_hints") or fallback_col.get("literal_hints") or "")

        result["columns_semantic"][col_name] = {
            "short_description": short_desc,
            "long_description": long_desc,
            "description": description,
            "sql_format_constraints": sql_constraints,
            "literal_hints": literal_hints,
        }

    return result


def enhance_semantics_with_retry(db_name, table_name, table_profile, max_retries=3):
    compact_profile = build_llm_profile_payload(table_profile)
    prompt = (
        f"你是数据库语义建模专家，请分析数据库 {db_name} 的表 {table_name}。\n"
        f"下面是结构化 profiling 结果（JSON）:\n{json.dumps(compact_profile, ensure_ascii=False)}\n\n"
        "请输出一个严格 JSON 对象，且仅输出 JSON，不要输出解释。\n"
        "JSON 结构必须为:\n"
        "{\n"
        '  "table_summary": "...",\n'
        '  "join_hints": "...",\n'
        '  "columns_semantic": {\n'
        '    "列名": {\n'
        '      "short_description": "...",\n'
        '      "long_description": "...",\n'
        '      "description": "...",\n'
        '      "sql_format_constraints": "...",\n'
        '      "literal_hints": "..."\n'
        "    }\n"
        "  }\n"
        "}\n"
        "要求：\n"
        "1) short_description 简洁解释字段业务含义；\n"
        "2) long_description 包含值域、单位、编码或取值模式；\n"
        "3) sql_format_constraints 说明生成 SQL 时是否需要 CAST、是否加引号、是否日期函数；\n"
        "4) literal_hints 尽量给出可直接用于过滤条件的典型值。"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": get_chat_model(),
                "messages": [
                    {"role": "system", "content": "你只输出合法 JSON 对象，不要输出额外文本。"},
                    {"role": "user", "content": prompt},
                ],
                "timeout": 60,
            }

            use_json_mode = os.getenv("PROFILE_USE_JSON_MODE", "1").strip() != "0"
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = get_chat_client().chat.completions.create(**kwargs)
            except Exception as e:
                if "response_format" in str(e).lower() and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    response = get_chat_client().chat.completions.create(**kwargs)
                else:
                    raise

            content = response.choices[0].message.content
            parsed = extract_json_object(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM 返回非 JSON 对象")

            normalized = normalize_semantic_description(parsed, table_name, table_profile)
            return normalized
        except Exception as e:
            last_error = e
            print(f"    ⏳ 语义增强失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2**attempt)

    fallback = build_fallback_semantic_description(table_name, table_profile)
    if last_error:
        fallback["generation_error"] = str(last_error)
    return fallback


def get_db_profile(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_rows = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    include_internal = os.getenv("PROFILE_INCLUDE_INTERNAL_TABLES", "0").strip() == "1"

    db_profile = {}
    for (table_name,) in table_rows:
        if not include_internal and is_internal_table(table_name):
            continue

        try:
            db_profile[table_name] = build_table_profile(cursor, table_name)
        except Exception as e:
            print(f"  ⚠️ 警告：跳过表 {table_name}，原因: {e}")

    detect_implicit_join_candidates(db_profile)
    cleanup_private_profile_fields(db_profile)

    conn.close()
    return db_profile


def read_cached_profile_meta(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            meta = payload.get("__profile_meta__", {})
            if isinstance(meta, dict):
                return meta
    except Exception:
        return {}
    return {}


def read_cached_profile_payload(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def split_cached_tables(payload):
    if not isinstance(payload, dict):
        return {}
    tables = {}
    for table_name, table_payload in payload.items():
        if str(table_name).startswith("__"):
            continue
        if isinstance(table_payload, dict):
            tables[table_name] = table_payload
    return tables


def write_profile_cache(cache_file, table_payload, meta):
    payload = dict(table_payload)
    payload["__profile_meta__"] = dict(meta or {})
    tmp_path = f"{cache_file}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_file)


def should_skip_cache(cache_file, force_rebuild=False):
    if force_rebuild:
        return False, "force_rebuild=1"
    if not os.path.exists(cache_file):
        return False, "cache_missing"

    meta = read_cached_profile_meta(cache_file)
    cached_ver = str(meta.get("profile_version", "")).strip()
    ver_tag = cached_ver or "unknown"
    pipeline_status = str(meta.get("pipeline_status", "")).strip().lower()
    if cached_ver == PROFILE_VERSION and pipeline_status == "completed":
        return True, f"cache_hit_{ver_tag}_completed"
    if cached_ver == PROFILE_VERSION:
        return False, f"cache_resume_{ver_tag}_{pipeline_status or 'unknown'}"
    return False, f"cache_stale_{ver_tag}"


def run_industrial_pipeline(base_dir):
    output_dir = get_metadata_cache_dir()
    os.makedirs(output_dir, exist_ok=True)

    force_rebuild = os.getenv("PROFILE_FORCE_REBUILD", "0").strip() == "1"

    db_folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]
    print(f"🧱 当前知识库命名空间: {describe_active_kb()} -> {output_dir}")
    print(f"🚀 侦测到 {len(db_folders)} 个数据库，开始执行离线元数据提取流水线...")

    for db_name in db_folders:
        db_path = os.path.join(base_dir, db_name, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            continue

        print(f"\n📁 正在处理数据库: [{db_name}]")
        cache_file = os.path.join(output_dir, f"{db_name}_enhanced.json")
        skip_cache, cache_reason = should_skip_cache(cache_file, force_rebuild=force_rebuild)
        if skip_cache:
            print(f"  ⏭️ 检测到缓存，跳过（{cache_reason}）")
            continue
        if os.path.exists(cache_file):
            print(f"  ♻️ 发现旧缓存，准备续跑/重建（{cache_reason}）")

        print("  - 阶段 1: 深度 Profiling（含隐式关联候选）...")
        raw_profile = get_db_profile(db_path)

        cached_payload = read_cached_profile_payload(cache_file) if (os.path.exists(cache_file) and not force_rebuild) else {}
        cached_meta = cached_payload.get("__profile_meta__", {}) if isinstance(cached_payload, dict) else {}
        cached_version = str(cached_meta.get("profile_version", "")).strip()
        can_resume = cached_version == PROFILE_VERSION
        resumed_tables = split_cached_tables(cached_payload) if can_resume else {}

        if resumed_tables:
            print(f"  ↪️ 断点续跑：已复用 {len(resumed_tables)} 张表的结果。")

        enhanced_profile = dict(resumed_tables)
        table_items = list(raw_profile.items())
        table_limit = int(os.getenv("PROFILE_TABLE_LIMIT", "0") or 0)
        if table_limit > 0:
            table_items = table_items[:table_limit]
        total_table_count = len(table_items)
        pending_table_items = [(name, info) for name, info in table_items if name not in enhanced_profile]

        if not pending_table_items:
            print("  ✅ 当前库所有目标表都已完成，无需继续。")
            final_meta = {
                "profile_version": PROFILE_VERSION,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_enhanced": bool(cached_meta.get("llm_enhanced", False)),
                "sample_value_limit": PROFILE_SAMPLE_VALUE_LIMIT,
                "top_k": PROFILE_TOP_K,
                "minhash_num_perm": PROFILE_MINHASH_NUM_PERM,
                "minhash_similarity_threshold": PROFILE_MINHASH_MIN_SIMILARITY,
                "implicit_max_per_column": PROFILE_IMPLICIT_MAX_PER_COLUMN,
                "table_count": total_table_count,
                "processed_table_count": total_table_count,
                "pipeline_status": "completed",
            }
            write_profile_cache(cache_file, enhanced_profile, final_meta)
            continue

        runtime_disable_llm = os.getenv("PROFILE_DISABLE_LLM", "0").strip() == "1"
        llm_enabled = not (PROFILE_DISABLE_LLM or runtime_disable_llm)
        if llm_enabled:
            try:
                get_chat_client()
            except Exception as e:
                llm_enabled = False
                print(f"  ⚠️ 语义 LLM 初始化失败，自动切换 fallback 语义: {e}")
        else:
            print("  ℹ️ PROFILE_DISABLE_LLM=1，使用规则语义描述（fallback）。")

        try:
            for idx, (table_name, table_info) in enumerate(pending_table_items, start=1):
                done_count = len(enhanced_profile)
                print(
                    f"  - 阶段 2: 语义增强 [{done_count + 1}/{total_table_count}] "
                    f"(pending {idx}/{len(pending_table_items)}) 表 {table_name} ..."
                )
                if llm_enabled:
                    semantic_desc = enhance_semantics_with_retry(db_name, table_name, table_info)
                else:
                    semantic_desc = build_fallback_semantic_description(table_name, table_info)
                enhanced_profile[table_name] = {
                    "physical_stats": table_info,
                    "semantic_description": semantic_desc,
                }

                checkpoint_meta = {
                    "profile_version": PROFILE_VERSION,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "llm_enhanced": bool(llm_enabled),
                    "sample_value_limit": PROFILE_SAMPLE_VALUE_LIMIT,
                    "top_k": PROFILE_TOP_K,
                    "minhash_num_perm": PROFILE_MINHASH_NUM_PERM,
                    "minhash_similarity_threshold": PROFILE_MINHASH_MIN_SIMILARITY,
                    "implicit_max_per_column": PROFILE_IMPLICIT_MAX_PER_COLUMN,
                    "table_count": total_table_count,
                    "processed_table_count": len(enhanced_profile),
                    "pipeline_status": "in_progress",
                }
                write_profile_cache(cache_file, enhanced_profile, checkpoint_meta)

                if llm_enabled and PROFILE_LLM_SLEEP_SECONDS > 0:
                    time.sleep(PROFILE_LLM_SLEEP_SECONDS)
        except KeyboardInterrupt:
            interrupt_meta = {
                "profile_version": PROFILE_VERSION,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_enhanced": bool(llm_enabled),
                "sample_value_limit": PROFILE_SAMPLE_VALUE_LIMIT,
                "top_k": PROFILE_TOP_K,
                "minhash_num_perm": PROFILE_MINHASH_NUM_PERM,
                "minhash_similarity_threshold": PROFILE_MINHASH_MIN_SIMILARITY,
                "implicit_max_per_column": PROFILE_IMPLICIT_MAX_PER_COLUMN,
                "table_count": total_table_count,
                "processed_table_count": len(enhanced_profile),
                "pipeline_status": "in_progress",
            }
            write_profile_cache(cache_file, enhanced_profile, interrupt_meta)
            print("  ⏸️ 检测到中断，已保存断点进度。下次直接重跑同命令即可续跑。")
            raise

        final_meta = {
            "profile_version": PROFILE_VERSION,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_enhanced": bool(llm_enabled),
            "sample_value_limit": PROFILE_SAMPLE_VALUE_LIMIT,
            "top_k": PROFILE_TOP_K,
            "minhash_num_perm": PROFILE_MINHASH_NUM_PERM,
            "minhash_similarity_threshold": PROFILE_MINHASH_MIN_SIMILARITY,
            "implicit_max_per_column": PROFILE_IMPLICIT_MAX_PER_COLUMN,
            "table_count": total_table_count,
            "processed_table_count": total_table_count,
            "pipeline_status": "completed",
        }
        write_profile_cache(cache_file, enhanced_profile, final_meta)
        print(f"  ✅ [{db_name}] 元数据提取完成，已落盘: {cache_file}")


if __name__ == "__main__":
    dev_path = os.path.join("data", "dev_databases")
    run_industrial_pipeline(dev_path)
