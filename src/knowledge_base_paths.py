import os
import re


KB_NAME_ENV = "KB_NAME"
KB_ROOT_ENV = "KB_ROOT_DIR"

DEFAULT_METADATA_CACHE_DIR = os.path.join("data", "metadata_cache")
DEFAULT_VECTOR_INDEX_PATH = os.path.join("data", "vector_index.npz")
DEFAULT_LITERAL_INDEX_PATH = os.path.join("data", "literal_index.json")
DEFAULT_KB_ROOT_DIR = os.path.join("data", "knowledge_bases")


def _slugify(text):
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    slug = slug.strip("._-")
    return slug


def get_kb_name():
    return _slugify(os.getenv(KB_NAME_ENV, ""))


def get_kb_root_dir():
    kb_name = get_kb_name()
    if not kb_name:
        return ""
    root_dir = os.getenv(KB_ROOT_ENV, "").strip() or DEFAULT_KB_ROOT_DIR
    return os.path.join(root_dir, kb_name)


def get_metadata_cache_dir():
    kb_root = get_kb_root_dir()
    if kb_root:
        return os.path.join(kb_root, "metadata_cache")
    return DEFAULT_METADATA_CACHE_DIR


def get_vector_index_path():
    kb_root = get_kb_root_dir()
    if kb_root:
        return os.path.join(kb_root, "vector_index.npz")
    return DEFAULT_VECTOR_INDEX_PATH


def get_literal_index_path():
    kb_root = get_kb_root_dir()
    if kb_root:
        return os.path.join(kb_root, "literal_index.json")
    return DEFAULT_LITERAL_INDEX_PATH


def describe_active_kb():
    kb_name = get_kb_name()
    if kb_name:
        return kb_name
    return "default"
