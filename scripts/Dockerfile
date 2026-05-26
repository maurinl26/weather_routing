# Image GPU pour cloud (Lambda, RunPod, GCP). Basée CUDA 12.4 + Python 3.11.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libgeos-dev libproj-dev libnetcdf-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Outils Python
RUN pip install --no-cache-dir uv

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

RUN uv sync --frozen || uv sync

ENTRYPOINT ["uv", "run"]
CMD ["wxr-train", "experiment=finetune_bog", "cluster=cloud"]
