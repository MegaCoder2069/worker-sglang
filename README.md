![SGLang worker banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sglang_banner-A9R2vQzvSUmLvqMZ8MzehfZtRDxHJR.jpeg)

Run LLMs and VLMs using [SGLang](https://docs.sglang.ai). By default this worker starts an H200-only FP8 preset for [sgl-project/DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8).

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sglang)](https://www.runpod.io/console/hub/runpod-workers/worker-sglang)

---

## Endpoint Configuration

All behaviour is controlled through environment variables:

| Environment Variable              | Description                                       | Default                               | Options                                                                                   |
| --------------------------------- | ------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `SGLANG_PRESET`                   | Launch preset                                     | "deepseek-v4-flash-fp8"               | "deepseek-v4-flash-fp8", "none"                                                           |
| `DEEPSEEK_V4_HARDWARE`            | Hardware target for the DeepSeek V4 FP8 preset   | "h200"                                | "h200" in the RunPod Hub template                                                         |
| `DEEPSEEK_V4_RECIPE`              | DeepSeek V4 serving recipe                        | "balanced"                            | "low-latency", "balanced", "max-throughput"                                               |
| `MODEL_NAME`                      | Hugging Face model name or local path             | "sgl-project/DeepSeek-V4-Flash-FP8"   | Hugging Face repo ID or local folder path                                                 |
| `HF_TOKEN`                        | HuggingFace access token for gated/private models |                                       | Your HuggingFace access token                                                             |
| `TOKENIZER_PATH`                  | Path of the tokenizer                             |                                       |                                                                                           |
| `TOKENIZER_MODE`                  | Tokenizer mode                                    | "auto"                                | "auto", "slow"                                                                            |
| `LOAD_FORMAT`                     | Format of model weights to load                   | "auto"                                | "auto", "pt", "safetensors", "npcache", "dummy"                                           |
| `DTYPE`                           | Data type for weights and activations             | "auto"                                | "auto", "half", "float16", "bfloat16", "float", "float32"                                 |
| `CONTEXT_LENGTH`                  | Model's maximum context length                    | 400000 in preset                      |                                                                                           |
| `QUANTIZATION`                    | Quantization method                               |                                       | "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "squeezellm", "bitsandbytes" |
| `SERVED_MODEL_NAME`               | Override model name in API                        | "deepseek-ai/DeepSeek-V4-Flash"       |                                                                                           |
| `CHAT_TEMPLATE`                   | Chat template name or path                        |                                       |                                                                                           |
| `MEM_FRACTION_STATIC`             | Fraction of memory for static allocation          |                                       |                                                                                           |
| `MAX_RUNNING_REQUESTS`            | Maximum number of running requests                |                                       |                                                                                           |
| `MAX_TOTAL_TOKENS`                | Maximum tokens in memory pool                     |                                       |                                                                                           |
| `CHUNKED_PREFILL_SIZE`            | Max tokens in chunk for chunked prefill           |                                       |                                                                                           |
| `MAX_PREFILL_TOKENS`              | Max tokens in prefill batch                       | 16384                                 |                                                                                           |
| `SCHEDULE_POLICY`                 | Request scheduling policy                         | "fcfs"                                | "lpm", "random", "fcfs", "dfs-weight"                                                     |
| `SCHEDULE_CONSERVATIVENESS`       | Conservativeness of schedule policy               | 1.0                                   |                                                                                           |
| `TENSOR_PARALLEL_SIZE` / `TP`     | Tensor parallelism size                           | H200: 4 in preset                     |                                                                                           |
| `STREAM_INTERVAL`                 | Streaming interval in token length                | 1                                     |                                                                                           |
| `RANDOM_SEED`                     | Random seed                                       |                                       |                                                                                           |
| `LOG_LEVEL`                       | Logging level for all loggers                     | "info"                                |                                                                                           |
| `LOG_LEVEL_HTTP`                  | Logging level for HTTP server                     |                                       |                                                                                           |
| `API_KEY`                         | API key for the server                            |                                       |                                                                                           |
| `FILE_STORAGE_PATH`               | Directory for storing uploaded/generated files    | "sglang_storage"                      |                                                                                           |
| `DATA_PARALLEL_SIZE` / `DP`       | Data parallelism size                             | H200: 4 in balanced preset            |                                                                                           |
| `MOE_A2A_BACKEND`                 | MoE all-to-all backend                            | "deepep" in balanced/max-throughput   | "deepep", "none"                                                                          |
| `DEEPEP_CONFIG`                   | JSON passed to `--deepep-config`                  | DeepEP 96-SM config in preset         |                                                                                           |
| `CUDA_GRAPH_MAX_BS`               | CUDA graph max batch size                         | H200 balanced: 128                    |                                                                                           |
| `SPECULATIVE_ALGO`                | Speculative decoding algorithm                    | "EAGLE" in low-latency/balanced       | "EAGLE", "none"                                                                           |
| `SPECULATIVE_NUM_STEPS`           | Speculative decoding steps                        | balanced: 1, low-latency: 3           |                                                                                           |
| `SPECULATIVE_EAGLE_TOPK`          | EAGLE top-k                                       | 1                                     |                                                                                           |
| `SPECULATIVE_NUM_DRAFT_TOKENS`    | Speculative draft tokens                          | balanced: 2, low-latency: 4           |                                                                                           |
| `LOAD_BALANCE_METHOD`             | Load balancing strategy                           | "round_robin"                         | "round_robin", "shortest_queue"                                                           |
| `SKIP_TOKENIZER_INIT`             | Skip tokenizer init                               | false                                 | boolean (true or false)                                                                   |
| `TRUST_REMOTE_CODE`               | Allow custom models from Hub                      | true in preset                        | boolean (true or false)                                                                   |
| `LOG_REQUESTS`                    | Log inputs and outputs of requests                | false                                 | boolean (true or false)                                                                   |
| `SHOW_TIME_COST`                  | Show time cost of custom marks                    | false                                 | boolean (true or false)                                                                   |
| `DISABLE_RADIX_CACHE`             | Disable RadixAttention for prefix caching         | false                                 | boolean (true or false)                                                                   |
| `DISABLE_CUDA_GRAPH`              | Disable CUDA Graph                                | false                                 | boolean (true or false)                                                                   |
| `DISABLE_OUTLINES_DISK_CACHE`     | Disable disk cache for Outlines grammar           | false                                 | boolean (true or false)                                                                   |
| `ENABLE_TORCH_COMPILE`            | Optimize model with torch.compile                 | false                                 | boolean (true or false)                                                                   |
| `ENABLE_P2P_CHECK`                | Enable P2P check for GPU access                   | false                                 | boolean (true or false)                                                                   |
| `ENABLE_FLASHINFER_MLA`           | Enable FlashInfer MLA optimization                | false                                 | boolean (true or false)                                                                   |
| `ENABLE_DP_ATTENTION`             | Enable data-parallel attention                    | true in balanced/max-throughput       | boolean (true or false)                                                                   |
| `TRITON_ATTENTION_REDUCE_IN_FP32` | Cast Triton attention reduce op to FP32           | false                                 | boolean (true or false)                                                                   |
| `TOOL_CALL_PARSER`                | Defines the parser used to interpret responses    | "deepseekv4" in preset                | "llama3", "llama4", "mistral", "qwen25", "deepseekv3", "deepseekv4", "none"               |
| `REASONING_PARSER`                | Defines the parser used for reasoning traces      | "deepseek-v4" in preset               | "llama3", "llama4", "mistral", "qwen25", "deepseekv3", "deepseek-v4", "none"              |

## DeepSeek V4 Flash FP8 Preset

The default `SGLANG_PRESET=deepseek-v4-flash-fp8` targets 4x H200 GPUs with the SGLang Docker image `lmsysorg/sglang:deepseek-v4-hopper`.

- Model path: `sgl-project/DeepSeek-V4-Flash-FP8`
- Served model name: `deepseek-ai/DeepSeek-V4-Flash`
- Default recipe: `balanced`
- Context length: `400000`
- RunPod Hub GPU pool: `HOPPER_141`
- RunPod Hub GPU count: `4`
- H200 default: `--tp 4 --dp 4 --enable-dp-attention`
- Tool parser: `deepseekv4`
- Reasoning parser: `deepseek-v4`

Every preset value can be overridden with the matching environment variable. Set parser/backend options to `none` to suppress those flags, or set `SGLANG_PRESET=none` to run the generic SGLang worker configuration.

The included `docker-compose.yml` uses the same H200-only DeepSeek V4 Flash FP8 defaults as the RunPod Hub template and expects 4 H200 GPUs.

The Dockerfile also sets the same runtime defaults, so a RunPod build that pulls this repository starts with `MODEL_NAME=sgl-project/DeepSeek-V4-Flash-FP8`, `SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash`, `CONTEXT_LENGTH=400000`, `TOOL_CALL_PARSER=deepseekv4`, and `REASONING_PARSER=deepseek-v4`. RunPod's GitHub build integration has an 80 GB image limit, so this Dockerfile does not bake model weights into the image; weights are downloaded into the configured cache at runtime.

## Serverless Model Cache

RunPod's documented Serverless pattern is to initialize the model once when the worker starts, outside the request handler, and to use a Network Volume mounted at `/runpod-volume` for persistent model cache. This worker follows that pattern: `handler.py` starts SGLang at module import, then each job only forwards requests to the already-running SGLang server.

For production, attach a Network Volume in the same datacenter as the H200 endpoint and size it for the DeepSeek V4 Flash FP8 weights. The template defaults cache paths to `/runpod-volume/huggingface-cache/hub` through `HF_HOME` and `HUGGINGFACE_HUB_CACHE`.

- Warm worker: model is already loaded; requests do not download or reload weights.
- Cold start with Network Volume: SGLang reuses the cached weights from `/runpod-volume`.
- Cold start without Network Volume: weights are downloaded into the worker's ephemeral container disk and must be downloaded again for a new worker.

## Tool/Function Calling and Reasoning

- **Tool/Function calling**: Set the `TOOL_CALL_PARSER` environment variable to match your model family. Supported values include `llama3`, `llama4`, `mistral`, `qwen25`, `deepseekv3`, and `deepseekv4`. In the DeepSeek V4 preset this defaults to `deepseekv4`; set it to `none` to suppress the flag.

  - Example (docker-compose): set `TOOL_CALL_PARSER=deepseekv4` under `environment:`.
  - Example (RunPod Hub): set the `TOOL_CALL_PARSER` env var in the UI.

- **Reasoning**: Set the `REASONING_PARSER` environment variable to match your model family if you want to enable reasoning traces parsing. In the DeepSeek V4 preset this defaults to `deepseek-v4`; set it to `none` to suppress the flag.
  - Example (docker-compose): set `REASONING_PARSER=deepseek-v4` under `environment:`.
  - Example (RunPod Hub): set the `REASONING_PARSER` env var in the UI.

## API Usage

This worker supports two API formats: **RunPod native** and **OpenAI-compatible**.

### RunPod Native API

For testing directly in the RunPod UI, use these examples in your endpoint's request tab.

#### Chat Completions

```json
{
  "input": {
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

#### Chat Completions (Streaming)

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "max_tokens": 500,
    "temperature": 0.8,
    "stream": true
  }
}
```

#### Native Text Generation

For direct SGLang text generation without OpenAI chat format:

```json
{
  "input": {
    "text": "The capital of France is",
    "sampling_params": {
      "max_new_tokens": 64,
      "temperature": 0.0
    }
  }
}
```

#### List Models

```json
{
  "input": {
    "openai_route": "/v1/models"
  }
}
```

---

### OpenAI-Compatible API

For external clients and SDKs, use the `/openai/v1` path prefix with your RunPod API key.

#### Chat Completions

**Path:** `/openai/v1/chat/completions`

```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Chat Completions (Streaming)

```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "messages": [
    { "role": "user", "content": "Write a short story about a robot." }
  ],
  "max_tokens": 500,
  "temperature": 0.8,
  "stream": true
}
```

#### List Models

**Path:** `/openai/v1/models`

```json
{}
```

#### Response Format

Both APIs return the same response format:

```json
{
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Paris." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10 }
}
```

---

## Usage

Below are minimal `python` snippets so you can copy-paste to get started quickly.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI compatible API

Minimal Python example using the official `openai` SDK:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
```

`Chat Completions (Non-Streaming)`

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash",
    messages=[{"role": "user", "content": "Give a two lines on Planet Earth ?"}],
    temperature=0,
    max_tokens=100,

)
print(f"Response: {response}")
```

`Chat Completions (Streaming)`

```python
response_stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash",
    messages=[{"role": "user", "content": "Give a two lines on Planet Earth ?"}],
    temperature=0,
    max_tokens=100,
    stream=True

)
for response in response_stream:
    print(response.choices[0].delta.content or "", end="", flush=True)
```

## Compatibility

Anything not recognized by worker-sglang is forwarded verbatim to `/generate`, so advanced options in the SGLang docs (logprobs, sessions, images, etc.) also work.
