FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB collection at image build time so the container starts fast
# and works even if the image is later run without network access.
RUN python retrieval.py

ENV MOCK_LLM=1
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
