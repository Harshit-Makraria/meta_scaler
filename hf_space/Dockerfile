FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --upgrade pip && \
    pip install \
      "openenv-core==0.2.3" \
      "fastapi>=0.110.0" \
      "uvicorn>=0.29.0" \
      "pydantic>=2.0.0" \
      "numpy>=1.26.0" \
      "python-dotenv>=1.0.0"

ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
