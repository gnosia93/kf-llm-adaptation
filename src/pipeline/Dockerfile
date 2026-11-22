# 파이프라인의 모든 단계가 쓰는 이미지 (학습 · 평가 · 관문 · 배포).
#   docker build -f pipelines/Dockerfile -t <레지스트리>/llm-ft-pipeline:<태그> .
#   docker push <레지스트리>/llm-ft-pipeline:<태그>
#   PIPELINE_IMAGE=<레지스트리>/llm-ft-pipeline:<태그> python -m pipelines.pipeline
# Linux 용 PyPI torch 휠에는 CUDA 런타임이 들어 있어서 CUDA 베이스 이미지가 필요 없다 (노드에 NVIDIA 드라이버만 있으면 됨).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-cuda.txt requirements-pipeline.txt ./
RUN pip install --no-cache-dir -r requirements-cuda.txt -r requirements-pipeline.txt

# 코드만 복사 (.dockerignore 가 data/ raw/ models/ 등을 뺀다). 데이터는 파이프라인의 prepare-data 가 받는다
COPY . /app
ENV PYTHONPATH=/app \
    LLM_FT_APP=/app \
    HF_HOME=/tmp/hf \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    TOKENIZERS_PARALLELISM=false
