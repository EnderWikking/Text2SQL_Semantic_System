import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
STATIC_ROOT = REPO_ROOT / "web_console" / "static"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation_core import run_unified_evaluation  # noqa: E402
from evaluate_sixway_comparison import run_sixway_comparison  # noqa: E402
from knowledge_base_paths import describe_active_kb, get_metadata_cache_dir  # noqa: E402
import provider_clients  # noqa: E402
from provider_clients import get_api_mode  # noqa: E402


RUNTIME_CONFIG_PATH = REPO_ROOT / "web_console" / "runtime_config.json"
PRIVATE_EXPORTS_PATH = REPO_ROOT / "scripts" / "api_mode_exports.private.sh"
SUPPORTED_API_MODES = ("deepseek", "aliyun", "lab")
MODE_DEFAULT_ENV = {
    "deepseek": {
        "API_MODE": "deepseek",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "DEEPSEEK_CHAT_MODEL": "deepseek-chat",
        "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "ALIYUN_EMBED_MODEL": "text-embedding-v3",
    },
    "aliyun": {
        "API_MODE": "aliyun",
        "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "ALIYUN_CHAT_MODEL": "qwen3.6-plus",
        "ALIYUN_EMBED_MODEL": "text-embedding-v3",
    },
    "lab": {
        "API_MODE": "lab",
        "LAB_CHAT_BASE_URL": "https://qwen.nju-slab.cn/v1",
        "LAB_CHAT_MODEL": "qwen-local",
        "LAB_EMBED_BASE_URL": "https://embedding.nju-slab.cn/v1",
        "LAB_EMBED_MODEL": "intfloat/multilingual-e5-large",
    },
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def parse_iso(value):
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def repo_relative(path_value):
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ensure_within_repo(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PermissionError("不允许读取仓库外部文件。") from exc
    return resolved


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_runtime_config():
    if not RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        payload = read_json(RUNTIME_CONFIG_PATH)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def save_runtime_config(payload):
    write_json(RUNTIME_CONFIG_PATH, payload)


def parse_private_exports():
    values = {}
    if not PRIVATE_EXPORTS_PATH.exists():
        return values

    pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for raw_line in PRIVATE_EXPORTS_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def reset_provider_client_cache():
    provider_clients._CHAT_CLIENT = None
    provider_clients._EMBED_CLIENT = None
    provider_clients._CHAT_CONFIG_CACHE = None
    provider_clients._EMBED_CONFIG_CACHE = None


def configure_mode_env(mode):
    mode = str(mode or "").strip().lower()
    if mode not in SUPPORTED_API_MODES:
        raise ValueError("不支持的 API 模式。")

    defaults = MODE_DEFAULT_ENV[mode]
    for key, value in defaults.items():
        os.environ[key] = value

    if mode == "aliyun":
        os.environ["ALIYUN_CHAT_BASE_URL"] = os.getenv("ALIYUN_CHAT_BASE_URL", "").strip() or os.environ["DASHSCOPE_BASE_URL"]
        os.environ["ALIYUN_EMBED_BASE_URL"] = os.getenv("ALIYUN_EMBED_BASE_URL", "").strip() or os.environ["DASHSCOPE_BASE_URL"]
    elif mode == "deepseek":
        os.environ["DEEPSEEK_BASE_URL"] = os.getenv("DEEPSEEK_BASE_URL", "").strip() or MODE_DEFAULT_ENV["deepseek"]["DEEPSEEK_BASE_URL"]
    elif mode == "lab":
        os.environ["LAB_CHAT_BASE_URL"] = os.getenv("LAB_CHAT_BASE_URL", "").strip() or MODE_DEFAULT_ENV["lab"]["LAB_CHAT_BASE_URL"]
        os.environ["LAB_EMBED_BASE_URL"] = os.getenv("LAB_EMBED_BASE_URL", "").strip() or MODE_DEFAULT_ENV["lab"]["LAB_EMBED_BASE_URL"]

    private_exports = parse_private_exports()
    for key, value in private_exports.items():
        if value:
            os.environ[key] = value

    reset_provider_client_cache()


def current_mode_env_summary():
    mode = safe_call(get_api_mode, default=os.getenv("API_MODE", "deepseek"))
    issues = []
    if mode == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY", "").strip():
            issues.append("缺少 DEEPSEEK_API_KEY")
        if not (
            os.getenv("DEEPSEEK_EMBED_API_KEY", "").strip()
            or os.getenv("ALIYUN_EMBED_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        ):
            issues.append("缺少 embedding key")
    elif mode == "aliyun":
        if not (os.getenv("ALIYUN_CHAT_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()):
            issues.append("缺少阿里云 chat key")
        if not (
            os.getenv("ALIYUN_EMBED_API_KEY", "").strip()
            or os.getenv("ALIYUN_CHAT_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        ):
            issues.append("缺少阿里云 embedding key")
    elif mode == "lab":
        if not os.getenv("LAB_API_KEY", "").strip():
            issues.append("缺少 LAB_API_KEY")

    return {
        "mode": mode,
        "issues": issues,
        "ready": not issues,
        "available_modes": list(SUPPORTED_API_MODES),
    }


def bootstrap_runtime_env():
    runtime = load_runtime_config()
    selected_mode = str(runtime.get("api_mode") or os.getenv("API_MODE") or "deepseek").strip().lower()
    if selected_mode not in SUPPORTED_API_MODES:
        selected_mode = "deepseek"
    configure_mode_env(selected_mode)
    if not runtime:
        save_runtime_config({"api_mode": selected_mode, "updated_at": now_iso()})


def load_history_rows():
    history_path = REPO_ROOT / "data" / "evaluation_runs" / "history.jsonl"
    rows = []
    if not history_path.exists():
        return rows
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(
        key=lambda item: parse_iso(item.get("ended_at") or item.get("started_at")),
        reverse=True,
    )
    return rows


def load_dataset_options():
    dataset_dir = REPO_ROOT / "data"
    options = []
    if not dataset_dir.exists():
        return options

    skip_markers = ("tables", "append", "literal", "evaluation_results")
    for path in sorted(dataset_dir.glob("*.json")):
        name = path.name.lower()
        if any(marker in name for marker in skip_markers):
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, list) or not payload:
            continue
        sample = payload[0]
        if not isinstance(sample, dict):
            continue
        if {"db_id", "question", "SQL"}.issubset(sample.keys()):
            options.append(
                {
                    "label": path.name,
                    "value": repo_relative(path),
                    "size": len(payload),
                }
            )
    return options


def latest_rows_by_mode_and_hints(rows):
    grouped = {}
    for row in rows:
        key = (str(row.get("mode", "unknown")), bool(row.get("include_dataset_evidence", False)))
        previous = grouped.get(key)
        if previous is None or parse_iso(row.get("ended_at") or row.get("started_at")) > parse_iso(
            previous.get("ended_at") or previous.get("started_at")
        ):
            grouped[key] = row
    order = [
        ("baseline", False),
        ("baseline", True),
        ("official", False),
        ("official", True),
        ("enhanced", False),
        ("enhanced", True),
        ("fused", False),
        ("fused", True),
    ]
    ordered = []
    for key in order:
        if key in grouped:
            ordered.append(grouped[key])
    for key, value in sorted(grouped.items()):
        if value not in ordered:
            ordered.append(value)
    return ordered


def serialize_run_row(row):
    hints = bool(row.get("include_dataset_evidence", row.get("hints", False)))
    result_path = row.get("save_path") or row.get("result_file") or ""
    summary_path = row.get("summary_path") or row.get("summary_file") or ""
    accuracy = float(row.get("accuracy", 0.0) or 0.0)
    exec_rate = float(row.get("execution_success_rate", 0.0) or 0.0)
    return {
        "name": row.get("name") or f"{row.get('mode', 'unknown')}_{'hints' if hints else 'no_hints'}",
        "label": row.get("name") or f"{row.get('mode', 'unknown')}-{'hints' if hints else 'no-hints'}",
        "mode": str(row.get("mode", "unknown")),
        "hints": hints,
        "accuracy": round(accuracy, 2),
        "execution_success_rate": round(exec_rate, 2),
        "correct": int(row.get("correct", 0) or 0),
        "total": int(row.get("total", 0) or 0),
        "candidate_count": int(row.get("candidate_count", 0) or 0) if row.get("candidate_count") is not None else None,
        "duration_seconds": round(float(row.get("duration_seconds", 0.0) or 0.0), 2),
        "run_tag": row.get("run_tag", ""),
        "dataset_path": row.get("dataset_path", ""),
        "summary_path": repo_relative(summary_path) if summary_path else "",
        "result_path": repo_relative(result_path) if result_path else "",
        "started_at": row.get("started_at", ""),
        "ended_at": row.get("ended_at", ""),
        "error_count": int(row.get("error_count", 0) or 0),
        "error": row.get("error", ""),
    }


def load_comparison_payload():
    latest_sixway_path = REPO_ROOT / "data" / "evaluation_results_sixway_latest.json"
    if latest_sixway_path.exists():
        try:
            payload = read_json(latest_sixway_path)
            rows = payload.get("rows", [])
            if isinstance(rows, list) and rows:
                return {
                    "source": "latest_sixway",
                    "title": "最近一次六组对比",
                    "run_tag": payload.get("run_tag", ""),
                    "rows": [serialize_run_row(row) for row in rows],
                }
        except Exception:
            pass

    history_rows = load_history_rows()
    latest_rows = latest_rows_by_mode_and_hints(history_rows)
    return {
        "source": "history_latest",
        "title": "按模式汇总的最近结果",
        "run_tag": "",
        "rows": [serialize_run_row(row) for row in latest_rows],
    }


bootstrap_runtime_env()


class JobLogBuffer(io.TextIOBase):
    def __init__(self, state, job_id):
        self.state = state
        self.job_id = job_id

    def write(self, text):
        if text:
            self.state.append_log(self.job_id, text)
        return len(text)

    def flush(self):
        return None


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.active_job_id = None

    def start_job(self, job_type, params):
        with self.lock:
            active_job = self.jobs.get(self.active_job_id) if self.active_job_id else None
            if active_job and active_job.get("status") in {"queued", "running"}:
                raise RuntimeError("当前已有任务在运行，请等待它完成后再启动新的测试。")

            job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job = {
                "id": job_id,
                "type": job_type,
                "status": "queued",
                "params": params,
                "started_at": now_iso(),
                "ended_at": "",
                "updated_at": now_iso(),
                "logs": [],
                "summary": None,
                "error": "",
            }
            self.jobs[job_id] = job
            self.active_job_id = job_id
            return self._clone_job(job)

    def set_status(self, job_id, status):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job["status"] = status
            job["updated_at"] = now_iso()

    def append_log(self, job_id, text):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            chunks = str(text).replace("\r", "\n").split("\n")
            for chunk in chunks:
                line = chunk.strip()
                if not line:
                    continue
                job["logs"].append(line)
            job["logs"] = job["logs"][-240:]
            job["updated_at"] = now_iso()

    def finish_job(self, job_id, summary=None, error=""):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job["status"] = "failed" if error else "completed"
            job["ended_at"] = now_iso()
            job["updated_at"] = now_iso()
            job["summary"] = summary
            job["error"] = error
            if self.active_job_id == job_id:
                self.active_job_id = None

    def snapshot(self):
        with self.lock:
            active_job = None
            if self.active_job_id:
                active_job = self._clone_job(self.jobs.get(self.active_job_id))
            recent_jobs = sorted(
                self.jobs.values(),
                key=lambda item: parse_iso(item.get("started_at")),
                reverse=True,
            )[:10]
            return {
                "busy": bool(active_job),
                "active_job": active_job,
                "recent_jobs": [self._clone_job(job, include_logs=False) for job in recent_jobs],
            }

    def _clone_job(self, job, include_logs=True):
        if not job:
            return None
        data = {
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "params": dict(job.get("params", {})),
            "started_at": job.get("started_at", ""),
            "ended_at": job.get("ended_at", ""),
            "updated_at": job.get("updated_at", ""),
            "summary": job.get("summary"),
            "error": job.get("error", ""),
        }
        if include_logs:
            data["logs"] = list(job.get("logs", []))
        return data


STATE = AppState()


def normalize_limit(value):
    if value in ("", None):
        return None
    return max(1, int(value))


def normalize_candidate_count(value):
    if value in ("", None):
        return 1
    return max(1, int(value))


def run_single_job(job_id, params):
    STATE.set_status(job_id, "running")
    buffer = JobLogBuffer(STATE, job_id)
    summary = None
    error = ""
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            summary = run_unified_evaluation(
                mode=params["mode"],
                dataset_path=params["dataset_path"],
                test_limit=params["test_limit"],
                include_dataset_evidence=params["include_dataset_evidence"],
                candidate_count=params["candidate_count"],
                run_dir=params["run_dir"],
            )
        if isinstance(summary, dict):
            summary = serialize_run_row(summary)
    except Exception as exc:
        error = str(exc)
        STATE.append_log(job_id, traceback.format_exc())
    STATE.finish_job(job_id, summary=summary, error=error)


def run_sixway_job(job_id, params):
    STATE.set_status(job_id, "running")
    buffer = JobLogBuffer(STATE, job_id)
    summary = None
    error = ""
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            run_sixway_comparison(
                dataset_path=params["dataset_path"],
                test_limit=params["test_limit"],
            )
        latest_payload = read_json(REPO_ROOT / "data" / "evaluation_results_sixway_latest.json")
        summary = {
            "kind": "sixway",
            "run_tag": latest_payload.get("run_tag", ""),
            "run_dir": repo_relative(latest_payload.get("run_dir", "")),
            "summary_path": repo_relative(latest_payload.get("summary_path", "")),
            "rows": [serialize_run_row(row) for row in latest_payload.get("rows", [])],
        }
    except Exception as exc:
        error = str(exc)
        STATE.append_log(job_id, traceback.format_exc())
    STATE.finish_job(job_id, summary=summary, error=error)


def start_background_job(job_type, params):
    job = STATE.start_job(job_type, params)
    if job_type == "single":
        target = run_single_job
    else:
        target = run_sixway_job
    thread = threading.Thread(target=target, args=(job["id"], params), daemon=True)
    thread.start()
    return job


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "Text2SQLWebConsole/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.serve_static("index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/static/"):
            relative_path = parsed.path.replace("/static/", "", 1)
            return self.serve_static(relative_path, self.guess_content_type(relative_path))
        if parsed.path == "/api/config":
            mode_summary = current_mode_env_summary()
            return self.send_json(
                {
                    "api_mode": mode_summary["mode"],
                    "api_mode_ready": mode_summary["ready"],
                    "api_mode_issues": mode_summary["issues"],
                    "available_api_modes": mode_summary["available_modes"],
                    "active_kb": safe_call(describe_active_kb, default="default"),
                    "metadata_cache_dir": repo_relative(
                        safe_call(get_metadata_cache_dir, default="data/metadata_cache")
                    ),
                    "datasets": load_dataset_options(),
                    "default_run_dir": "data/evaluation_runs/ui_runs",
                }
            )
        if parsed.path == "/api/status":
            return self.send_json(STATE.snapshot())
        if parsed.path == "/api/history":
            query = parse_qs(parsed.query)
            limit = max(1, int(query.get("limit", ["12"])[0]))
            rows = [serialize_run_row(row) for row in load_history_rows()[:limit]]
            return self.send_json({"rows": rows})
        if parsed.path == "/api/comparison":
            return self.send_json(load_comparison_payload())
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            raw_path = unquote(query.get("path", [""])[0]).strip()
            if not raw_path:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "缺少 path 参数。")
            try:
                target = ensure_within_repo(raw_path)
            except Exception as exc:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            if not target.exists() or not target.is_file():
                return self.send_error_json(HTTPStatus.NOT_FOUND, "目标文件不存在。")
            if target.suffix not in {".json", ".jsonl", ".txt", ".log"}:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "当前仅支持查看 json/jsonl/txt/log 文件。")
            if target.suffix == ".json":
                try:
                    payload = read_json(target)
                    return self.send_json(
                        {
                            "path": repo_relative(target),
                            "content_type": "json",
                            "content": payload,
                        }
                    )
                except Exception as exc:
                    return self.send_error_json(HTTPStatus.BAD_REQUEST, f"JSON 读取失败: {exc}")
            text = target.read_text(encoding="utf-8", errors="replace")
            if target.suffix == ".jsonl":
                text = "\n".join(text.splitlines()[:200])
            return self.send_json(
                {
                    "path": repo_relative(target),
                    "content_type": "text",
                    "content": text,
                }
            )
        return self.send_error_json(HTTPStatus.NOT_FOUND, "接口不存在。")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run/single", "/api/run/sixway", "/api/config/api-mode"}:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "接口不存在。")
        try:
            payload = self.read_json_body()
        except ValueError as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        if parsed.path == "/api/config/api-mode":
            try:
                mode = str(payload.get("mode", "")).strip().lower()
                if mode not in SUPPORTED_API_MODES:
                    raise ValueError("mode 仅支持 deepseek / aliyun / lab。")
                configure_mode_env(mode)
                save_runtime_config({"api_mode": mode, "updated_at": now_iso()})
                return self.send_json(
                    {
                        "message": f"API 模式已切换为 {mode}",
                        **current_mode_env_summary(),
                    }
                )
            except Exception as exc:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        try:
            dataset_path = payload.get("dataset_path", "data/mini_dev.json")
            ensure_within_repo(dataset_path)
        except Exception as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, f"数据集路径不可用: {exc}")

        try:
            if parsed.path == "/api/run/single":
                params = {
                    "mode": str(payload.get("mode", "enhanced")).strip().lower(),
                    "dataset_path": dataset_path,
                    "test_limit": normalize_limit(payload.get("test_limit")),
                    "include_dataset_evidence": bool(payload.get("include_dataset_evidence", True)),
                    "candidate_count": normalize_candidate_count(payload.get("candidate_count")),
                    "run_dir": str(payload.get("run_dir", "data/evaluation_runs/ui_runs")).strip()
                    or "data/evaluation_runs/ui_runs",
                }
                if params["mode"] not in {"baseline", "official", "enhanced", "fused"}:
                    raise ValueError("mode 仅支持 baseline / official / enhanced / fused。")
                job = start_background_job("single", params)
            else:
                params = {
                    "dataset_path": dataset_path,
                    "test_limit": normalize_limit(payload.get("test_limit")),
                }
                job = start_background_job("sixway", params)
        except RuntimeError as exc:
            return self.send_error_json(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        return self.send_json({"job": job}, status=HTTPStatus.ACCEPTED)

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("请求体不是合法 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象。")
        return payload

    def serve_static(self, relative_path, content_type):
        target = (STATIC_ROOT / relative_path).resolve()
        try:
            target.relative_to(STATIC_ROOT)
        except ValueError:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "静态文件不存在。")
        if not target.exists() or not target.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "静态文件不存在。")
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_content_type(self, filename):
        suffix = Path(filename).suffix.lower()
        if suffix == ".css":
            return "text/css; charset=utf-8"
        if suffix == ".js":
            return "application/javascript; charset=utf-8"
        return "text/plain; charset=utf-8"

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)

    def log_message(self, format, *args):
        return


def safe_call(func, default=""):
    try:
        return func()
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(description="Lightweight local web console for Text2SQL evaluation")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    httpd = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Text2SQL Web Console running at {url}")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
