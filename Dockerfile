# --- Base python stage for gateway ---
FROM python:3.10-slim AS gateway

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 1. Install heavy dependencies (cached long-term)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install light dependencies (frequently updated)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    httpx==0.28.1 \
    pydantic==2.11.3 \
    safetensors==0.5.3 \
    transformers==4.48.3 \
    hivemind>=1.1.10

COPY gateway.py models.py transport.py index.html ./

EXPOSE 8080 4001
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8080"]


# --- CUDA stage for workers ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

# CPU thread cap — MERGE_PERF_PLAN.md §4.1A.
# Several worker containers share one host, so a per-process default of
# os.cpu_count() intra-op threads oversubscribes the machine by that factor
# during an averaging round, and every ATen op above the ~32k parallel grain
# size pays a contended OpenMP fork/join. OMP/MKL must be set in the
# environment because those libraries size their pools at init; the
# KIWI_TORCH_THREADS block at the top of worker.py reads it and calls
# torch.set_num_threads() to match. Raise it for one-worker-per-host.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    KIWI_TORCH_THREADS=2

# Install Python 3.10, pip, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 1. Install heavy dependencies (cached long-term, over 2GB of disk size)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.6.0 \
    transformers==4.48.3 \
    accelerate>=1.2.0

# 2. Install light dependencies (frequently updated or configuration-based)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    httpx==0.28.1 \
    pydantic==2.11.3 \
    safetensors==0.5.3 \
    zstandard==0.23.0 \
    lz4==4.3.3 \
    peft>=0.11.0 \
    hivemind>=1.1.10

COPY worker.py models.py transport.py mesh.py ./

EXPOSE 8000 4001
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]

