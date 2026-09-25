"""FastAPI wrapper around the LangGraph support-assistant pipeline (Task 5).

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \\
         -d '{"query": "What is the delivery fee for a small order?"}'
"""
from fastapi import FastAPI

from graph_app import answer_query
from retrieval import ingest_documents
from schemas import AskRequest, AskResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="A small RAG service answering questions about Zepto's own policies.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    # Make sure the ChromaDB collection exists before the first request arrives.
    ingest_documents(force=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return answer_query(request.query)
