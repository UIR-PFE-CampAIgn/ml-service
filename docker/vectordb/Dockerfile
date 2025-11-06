FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build tooling needed by some python deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install chromadb server with numpy pinned below 2.0
# Some chromadb dependencies try to bump numpy back to 2.x, so we
# force the re-install at the end to keep the runtime compatible.
RUN pip install "chromadb[server]==0.4.22" \
    && pip install --upgrade --force-reinstall "numpy<2"

RUN mkdir -p /chroma/data

EXPOSE 8000

ENV ANONYMIZED_TELEMETRY=false

CMD ["chroma", "run", "--path", "/chroma/data", "--host", "0.0.0.0", "--port", "8000"]
