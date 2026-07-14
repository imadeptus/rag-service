FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt qdrant-client

COPY src/ src/
COPY eval/ eval/
COPY sample_docs/ sample_docs/

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
