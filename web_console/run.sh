#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 $PYTHON_BIN"
  echo "请先创建虚拟环境并安装 requirements.txt 中的依赖。"
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m web_console.app "$@"
