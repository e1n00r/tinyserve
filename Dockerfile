FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY tinyserve/ ./tinyserve/

RUN pip install --no-cache-dir --break-system-packages ".[server]"

ENV HF_HOME=/cache/huggingface

EXPOSE 8000

ENTRYPOINT ["tinyserve", "serve"]
CMD ["--model", "openai/gpt-oss-20b", "--port", "8000"]
