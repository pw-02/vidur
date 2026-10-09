#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Run inside a GPU allocation with Vidur's Sarathi profiling dependencies installed.
qwen_profile_model="${QWEN_PROFILE_MODEL:-Qwen/Qwen3-8B}"
qwen_profile_tokens="${QWEN_PROFILE_MAX_TOKENS:-4096}"
qwen_profile_batch="${QWEN_PROFILE_MAX_BATCH:-8}"
python vidur/profiling/mlp/main.py \
  --models "$qwen_profile_model" --num_gpus 1 \
  --num_tensor_parallel_workers 1 --max_tokens "$qwen_profile_tokens"
python vidur/profiling/attention/main.py \
  --models "$qwen_profile_model" --num_gpus 1 \
  --num_tensor_parallel_workers 1 \
  --max_model_len "$qwen_profile_tokens" --max_seq_len "$qwen_profile_tokens" \
  --max_batch_size "$qwen_profile_batch"
