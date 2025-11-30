FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy server code (includes pyproject with all deps) and app code.
COPY server ./server
COPY app ./app

# Install Python deps via uv:
# 1) install server package (MCP + DB deps)
# 2) install chat-gateway package (FastAPI/OpenAI deps)
RUN uv pip install --system ./server && uv pip install --system ./app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
