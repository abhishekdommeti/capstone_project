"""Embedding + ChromaDB layer (Task 1).

Ingestion:  load the 8 docs/*.txt files -> one chunk per document (they are short
enough that per-document chunking is appropriate) -> embed each chunk locally with
sentence-transformers' all-MiniLM-L6-v2 -> store in a persistent ChromaDB collection.

Retrieval always runs for real, in both MOCK_LLM modes: embedding a query with a
local sentence-transformers model and querying ChromaDB needs no API key and no
network call, so only the final answer-generation step (in graph_app.py) branches
on MOCK_LLM, never this module.

Run `python retrieval.py` once to (re)build the ChromaDB collection on disk at
./chroma_db before starting the API (main.py imports get_collection(), which builds
the collection automatically on first use if it does not exist yet).
"""
import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

_embedder = None       # lazy singletons so import alone doesn't load the model
_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_client() -> "chromadb.ClientAPI":
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def load_documents() -> list[dict]:
    """One chunk per document file, sorted by filename (doc_01 .. doc_08)."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]        # e.g. "doc_01"
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        chunks.append({"id": doc_id, "text": text})
    if not chunks:
        raise FileNotFoundError(f"No docs/doc_*.txt files found under {DOCS_DIR}")
    return chunks


def ingest_documents(force: bool = False) -> "chromadb.Collection":
    """Embed all 8 corpus chunks and (re)store them in the ChromaDB collection."""
    client = get_client()
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0 and not force:
        return collection

    chunks = load_documents()
    embedder = get_embedder()
    embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["id"]} for c in chunks],
    )
    return collection


def get_collection() -> "chromadb.Collection":
    """Return the collection, building it on first use if empty."""
    collection = ingest_documents(force=False)
    if collection.count() == 0:
        collection = ingest_documents(force=True)
    return collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed `query` and return the top-k most similar chunks via cosine similarity.

    Returns a list of {"id", "text", "distance"} ordered by similarity (best first).
    """
    collection = get_collection()
    embedder = get_embedder()
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    hits = []
    for doc_id, text, dist in zip(result["ids"][0], result["documents"][0], result["distances"][0]):
        hits.append({"id": doc_id, "text": text, "distance": dist})
    return hits


if __name__ == "__main__":
    col = ingest_documents(force=True)
    print(f"Ingested {col.count()} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_PATH}")
    for q in ["What is the delivery fee?", "How does Zepto Pass+ compare to Zepto Pass?"]:
        print(f"\nquery: {q!r}")
        for hit in retrieve(q):
            print(f"  {hit['id']}  dist={hit['distance']:.4f}  {hit['text'][:80]}...")
