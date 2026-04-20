# Base image: pet-infra GPU image tag TBD — see pet-infra/docker/ for actual tag.
# For M1 use python:3.11-slim as portable placeholder; upgrade to pet-infra CUDA tag in M2.
FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY core/pyproject.toml core/params.yaml /app/core/
COPY core/src /app/core/src
COPY core/tests /app/core/tests

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e "/app/core[dev,detector,tracker]"

WORKDIR /app/core
CMD ["pytest", "tests/", "-v", "-m", "not gpu"]
