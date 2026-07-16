# --- Base python stage for gateway ---
FROM python:3.10-slim AS gateway

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Share the host pip cache directory across rebuilds via Mount Cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    httpx==0.28.1 \
    pydantic==2.11.3 \
    safetensors==0.5.3 \
    transformers==4.48.3 \
    torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

COPY gateway.py models.py index.html ./

EXPOSE 8080
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8080"]


# --- CUDA stage for workers ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Share the host pip cache directory across rebuilds via Mount Cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    httpx==0.28.1 \
    pydantic==2.11.3 \
    safetensors==0.5.3 \
    transformers==4.48.3 \
    accelerate>=1.2.0 \
    torch==2.6.0 \
    zstandard==0.23.0 \
    lz4==4.3.3

COPY worker.py models.py ./

EXPOSE 8000
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
