#!/usr/bin/env bash
set -eu

rm -rf tl_out trace-qwen && time TORCH_COMPILE_DEBUG=1 TORCH_LOGS="all,dynamo" TORCH_TRACE=trace-qwen TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 ENABLE_TORCH_COMPILE=1 uv run qwen3_600m.py 2>&1 | tee qwen.log && tlparse trace-qwen/*.log
