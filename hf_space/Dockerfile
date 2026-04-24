FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal — we don't need torch for the env itself)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/

# Python deps
RUN pip install --upgrade pip && \
    pip install openenv-core fastapi uvicorn pydantic numpy python-dotenv

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
