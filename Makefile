.PHONY := all clean

TRACE_DIR := trace-qwen
TL_OUT_DIR := tl_out
QWEN_LOG := qwen.log
RUN_ENV := TORCH_COMPILE_DEBUG=1 TORCH_LOGS="all,dynamo" TORCH_TRACE=$(TRACE_DIR) TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 ENABLE_TORCH_COMPILE=1

all:
	@rm -rf $(TL_OUT_DIR) $(TRACE_DIR)
	time $(RUN_ENV) uv run qwen3_600m.py 2>&1 | tee $(QWEN_LOG)
	uv run tlparse --overwrite --no-browser $(TRACE_DIR)/*.log

clean:
	rm -rf $(TL_OUT_DIR) $(TRACE_DIR) $(QWEN_LOG)
