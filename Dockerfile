# Pin the linux/amd64 manifest directly. The tag also publishes an OCI index with
# an attestation manifest, and RunPod's cache exporter has failed on that path.
FROM --platform=linux/amd64 lmsysorg/sglang@sha256:d1b68460bf28412797005e79acdc3011f06663e5f820e0adf5dcc30214e12293

# Install uv package manager
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# install dependencies
COPY requirements.txt ./
RUN uv pip install --system --break-system-packages -r requirements.txt

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/
RUN mkdir -p /runpod-volume/huggingface-cache/hub

# Runtime defaults for the H200 DeepSeek V4 Flash FP8 deployment.
ARG MODEL_NAME="sgl-project/DeepSeek-V4-Flash-FP8"
ARG SERVED_MODEL_NAME="deepseek-ai/DeepSeek-V4-Flash"
ARG CONTEXT_LENGTH="400000"
ARG TOOL_CALL_PARSER="deepseekv4"
ARG REASONING_PARSER="deepseek-v4"
ARG TRUST_REMOTE_CODE="true"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG SGLANG_PRESET="deepseek-v4-flash-fp8"

ENV MODEL_NAME=$MODEL_NAME \
    SERVED_MODEL_NAME=$SERVED_MODEL_NAME \
    CONTEXT_LENGTH=$CONTEXT_LENGTH \
    TOOL_CALL_PARSER=$TOOL_CALL_PARSER \
    REASONING_PARSER=$REASONING_PARSER \
    TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    SGLANG_PRESET=$SGLANG_PRESET \
    DEEPSEEK_V4_RECIPE=balanced \
    DEEPSEEK_V4_HARDWARE=h200 \
    SGLANG_DSV4_FP4_EXPERTS=0 \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    HF_HUB_VERBOSITY=info \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    PYTHONUNBUFFERED=1

CMD ["python3", "handler.py"]
