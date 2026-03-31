#!/usr/bin/env bash
# 复制为 scripts/api_mode_exports.private.sh 使用
# 该文件会被 source scripts/api_mode_exports.sh 自动加载

# ===== DeepSeek Chat =====
export DEEPSEEK_API_KEY='YOUR_DEEPSEEK_API_KEY'

# ===== Embedding（二选一或都配）=====
# 方案 A：DeepSeek embedding
# export DEEPSEEK_EMBED_API_KEY='YOUR_DEEPSEEK_EMBED_API_KEY'
# export DEEPSEEK_EMBED_BASE_URL='https://api.deepseek.com'
# export DEEPSEEK_EMBED_MODEL='YOUR_DEEPSEEK_EMBED_MODEL'

# 方案 B：阿里 embedding（推荐你当前沿用）
export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'
export ALIYUN_EMBED_API_KEY='YOUR_DASHSCOPE_API_KEY'

# ===== Lab Mode =====
export LAB_API_KEY='YOUR_LAB_API_KEY'
export LAB_EMBED_API_KEY='YOUR_LAB_EMBED_API_KEY_OR_SAME_AS_LAB_API_KEY'
