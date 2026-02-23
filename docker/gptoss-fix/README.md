# vLLM gpt-oss Stop Token Fix — Docker Images

Pre-patched Docker images for running **gpt-oss** models (e.g., `openai/gpt-oss-20b`) with working tool calls.

**Issue:** [vllm-project/vllm#22519](https://github.com/vllm-project/vllm/issues/22519)

## The Problem

When `tools` are included in a request to a gpt-oss model, vLLM crashes with:

```
Unexpected token 12606 while expecting start token 200006
```

The root cause: gpt-oss stop tokens (`</call>` = 200012, `</return>` = 200002) are loaded into `default_sampling_params` at server startup but silently discarded on every request because `to_sampling_params()` never falls back to the defaults.

## Quick Start

### GPU (recommended — `openai/gpt-oss-20b` uses MXFP4 and fits on a single GPU)

```bash
# Build
docker build -f Dockerfile.gpu -t vllm-gpu-gptoss:latest .

# Run
docker run -d --name vllm-gptoss \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models \
  --shm-size=8g \
  -e HF_TOKEN="hf_..." \
  -e HF_HOME="/models/hf" \
  vllm-gpu-gptoss:latest \
  openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --max-model-len 4096 \
  --trust-remote-code \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

### CPU

```bash
# Build
docker build -f Dockerfile.cpu -t vllm-cpu-gptoss:latest .

# Run (needs ~40 GB RAM for BF16 weights)
docker run -d --name vllm-gptoss \
  -p 8000:8000 \
  -v /path/to/models:/models \
  --shm-size=8g \
  -e HF_TOKEN="hf_..." \
  -e HF_HOME="/models/hf" \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  vllm-cpu-gptoss:latest \
  unsloth/gpt-oss-20b-BF16 \
  --served-model-name gpt-oss-20b \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

## Test a Tool Call

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 200
  }' | python3 -m json.tool
```

## Custom Base Image

Both Dockerfiles accept a `BASE_IMAGE` build arg:

```bash
docker build -f Dockerfile.gpu \
  --build-arg BASE_IMAGE=your-registry/vllm:tag \
  -t vllm-gptoss:custom .
```

## What the Patch Does

1. **protocol.py** — merges server-default `stop_token_ids` with any client-specified ones in `to_sampling_params()`, so generation stops at `</call>` / `</return>`.
2. **harmony_utils.py** — adds a safety-net break in `parse_output_into_messages()` so the parser stops at stop tokens instead of crashing on trailing garbage.

The `apply_fix.py` script auto-detects whether vLLM uses the flat layout (`protocol.py`) or the subdirectory layout (`chat_completion/protocol.py` + `completion/protocol.py`).
