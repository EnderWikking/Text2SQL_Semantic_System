#!/usr/bin/env bash
# 用法：
#   source scripts/api_mode_exports.sh deepseek
#   source scripts/api_mode_exports.sh lab
#
# 说明：
# - 请用 `source` 执行，这样 export 的变量会保留在当前终端会话中。
# - 真实 key 请放在同目录下 `api_mode_exports.private.sh`（已加入 .gitignore）。

# 兼容 bash/zsh：仅允许被 source 执行
if ! (return 0 2>/dev/null); then
  echo "请使用 source 执行：source scripts/api_mode_exports.sh [deepseek|lab]"
  exit 1
fi

MODE="${1:-deepseek}"
if [ -n "${ZSH_VERSION:-}" ]; then
  SCRIPT_PATH="${(%):-%N}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PRIVATE_FILE="${SCRIPT_DIR}/api_mode_exports.private.sh"

if [ "$MODE" = "deepseek" ]; then
  # ===== DeepSeek 模式 =====
  export API_MODE=deepseek
  export DEEPSEEK_BASE_URL='https://api.deepseek.com'
  export DEEPSEEK_CHAT_MODEL='deepseek-chat'
  # deepseek 模式 embedding 默认回落到阿里 embedding
  export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
  export ALIYUN_EMBED_MODEL='text-embedding-v3'

elif [ "$MODE" = "lab" ]; then
  # ===== 实验室模式 =====
  export API_MODE=lab
  export LAB_CHAT_BASE_URL='https://qwen.nju-slab.cn/v1'
  export LAB_CHAT_MODEL='qwen-local'
  export LAB_EMBED_BASE_URL='https://embedding.nju-slab.cn/v1'
  export LAB_EMBED_MODEL='intfloat/multilingual-e5-large'

else
  echo "不支持的模式: $MODE"
  echo "可选值: deepseek | lab"
  return 1
fi

# 读取本地私有覆盖（不入库）
if [ -f "$PRIVATE_FILE" ]; then
  # shellcheck source=/dev/null
  source "$PRIVATE_FILE"
fi

# 模式基础校验
if [ "$MODE" = "deepseek" ]; then
  if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "缺少 DEEPSEEK_API_KEY。请在 ${PRIVATE_FILE} 中配置。"
    return 1
  fi
  if [ -z "${DEEPSEEK_EMBED_API_KEY:-}" ] && [ -z "${ALIYUN_EMBED_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "deepseek 模式缺少 embedding key。请配置 DEEPSEEK_EMBED_API_KEY 或 DASHSCOPE_API_KEY。"
    return 1
  fi
fi

if [ "$MODE" = "lab" ]; then
  if [ -z "${LAB_API_KEY:-}" ]; then
    echo "缺少 LAB_API_KEY。请在 ${PRIVATE_FILE} 中配置。"
    return 1
  fi
fi

MODE_UPPER="$(printf '%s' "$MODE" | tr '[:lower:]' '[:upper:]')"
echo "[api_mode_exports] 已切换到 ${MODE_UPPER} 模式"
echo "API_MODE=$API_MODE"
