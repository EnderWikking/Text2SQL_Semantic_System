import os
import threading
from openai import OpenAI

_LOCK = threading.Lock()
_CHAT_CLIENT = None
_EMBED_CLIENT = None
_CHAT_CONFIG_CACHE = None
_EMBED_CONFIG_CACHE = None


def _require_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def get_api_mode():
    mode = os.getenv("API_MODE", "deepseek").strip().lower()
    if mode not in ("lab", "deepseek"):
        raise RuntimeError("API_MODE 仅支持 'deepseek' 或 'lab'")
    return mode


def _chat_config():
    mode = get_api_mode()
    if mode == "lab":
        return {
            "api_key": _require_env("LAB_API_KEY"),
            "base_url": os.getenv("LAB_CHAT_BASE_URL", "https://qwen.nju-slab.cn/v1").strip(),
            "model": os.getenv("LAB_CHAT_MODEL", "qwen-local").strip(),
        }
    return {
        "api_key": _require_env("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        "model": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat").strip(),
    }


def _embed_config():
    mode = get_api_mode()
    if mode == "lab":
        return {
            "api_key": os.getenv("LAB_EMBED_API_KEY", os.getenv("LAB_API_KEY", "")).strip() or _require_env("LAB_API_KEY"),
            "base_url": os.getenv("LAB_EMBED_BASE_URL", "https://embedding.nju-slab.cn/v1").strip(),
            "model": os.getenv("LAB_EMBED_MODEL", "intfloat/multilingual-e5-large").strip(),
        }
    embed_api_key = os.getenv("DEEPSEEK_EMBED_API_KEY", "").strip()
    embed_model = os.getenv("DEEPSEEK_EMBED_MODEL", "").strip()
    if embed_api_key and embed_model:
        return {
            "api_key": embed_api_key,
            "base_url": os.getenv("DEEPSEEK_EMBED_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip(),
            "model": embed_model,
        }
    # DeepSeek 模式下若未单独配置 embedding，则回落到阿里 embedding 配置
    aliyun_embed_key = os.getenv("ALIYUN_EMBED_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if aliyun_embed_key:
        return {
            "api_key": aliyun_embed_key,
            "base_url": os.getenv("ALIYUN_EMBED_BASE_URL", os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).strip(),
            "model": os.getenv("ALIYUN_EMBED_MODEL", "text-embedding-v3").strip(),
        }
    raise RuntimeError("API_MODE=deepseek 时，embedding 未配置。请设置 DEEPSEEK_EMBED_API_KEY+DEEPSEEK_EMBED_MODEL，或配置阿里 embedding。")


def get_chat_client():
    global _CHAT_CLIENT, _CHAT_CONFIG_CACHE
    cfg = _chat_config()
    with _LOCK:
        if _CHAT_CLIENT is None or _CHAT_CONFIG_CACHE != cfg:
            _CHAT_CLIENT = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            _CHAT_CONFIG_CACHE = cfg
    return _CHAT_CLIENT


def get_embedding_client():
    global _EMBED_CLIENT, _EMBED_CONFIG_CACHE
    cfg = _embed_config()
    with _LOCK:
        if _EMBED_CLIENT is None or _EMBED_CONFIG_CACHE != cfg:
            _EMBED_CLIENT = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            _EMBED_CONFIG_CACHE = cfg
    return _EMBED_CLIENT


def get_chat_model():
    return _chat_config()["model"]


def get_embedding_model():
    return _embed_config()["model"]
