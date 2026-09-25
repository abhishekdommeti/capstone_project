# /support_assistant — Zepto policy support assistant (Module 3)

A small RAG service: 8 Zepto policy documents → local embeddings in ChromaDB → a LangGraph intent router → a schema-validated JSON answer → a FastAPI `/ask` endpoint → a Dockerfile.

Every LLM call is gated behind `MOCK_LLM` (default: mock, fully offline, deterministic — this is what gets graded). `MOCK_LLM=0` is an optional, ungraded extension that calls a real LLM (Groq's free tier).

```
support_assistant/
├── docs/doc_01.txt … doc_08.txt   # the 8 required policy documents, verbatim
├── retrieval.py                   # Task 1: embed (sentence-transformers) + store/query (ChromaDB)
├── prompts.py                     # Task 2: structured prompt templates (MOCK_LLM=0 path only)
├── llm.py                         # single MOCK_LLM gate; optional Groq call
├── graph_app.py                   # Task 3+4: LangGraph StateGraph, 3 nodes, conditional edge, Pydantic schema + retry
├── schemas.py                     # AskRequest / AskResponse Pydantic models
├── main.py                        # Task 5: FastAPI app, POST /ask
├── generate_example_calls.py      # writes results/example_calls.md (2+ example transcripts)
├── Dockerfile                     # Task 6
├── requirements.txt
├── .env.example
└── results/example_calls.md       # GENERATED — see "How to run" below
```

## ⚠️ Important — read before running

This code was written and reviewed in an offline sandbox with no network access, so `fastapi`, `pydantic`, `chromadb`, `sentence-transformers` and `langgraph` could not actually be installed or run there. I unit-tested every function's *logic* (the keyword classifier, the LangGraph routing, the mock-mode answer templates, the Pydantic schema's validation rules, the retry loop, the FastAPI handler) against small hand-written stand-ins for those libraries, so the control flow is verified. What is **not** yet verified is the real libraries' exact runtime behavior on your machine (import paths, ChromaDB's current API surface, etc. can drift between versions). **You must actually run this end to end, watch for errors, and fix any that come up before submitting** — treat this as a strong first draft, not a guaranteed-working build.

## Setup

```bash
cd support_assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # defaults to MOCK_LLM=1 — no key needed for the graded baseline
```

## How to run

```bash
# 1. Build the ChromaDB collection (embeds the 8 docs with all-MiniLM-L6-v2; downloads
#    the model the first time, then works offline)
python retrieval.py

# 2. Start the API (MOCK_LLM left at its default = mock/offline)
uvicorn main:app --host 0.0.0.0 --port 7860

# 3. In another terminal, try it:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee if my order is under INR 149?"}'

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What'\''s your favorite color?"}'
```

Then generate the required example-call transcripts and paste `results/example_calls.md` into the section below:

```bash
python generate_example_calls.py
cat results/example_calls.md
```

### Example calls (MOCK_LLM left at its default)

> **Paste the contents of `results/example_calls.md` here before submitting** — the two example calls below are illustrative only and were produced against a text-overlap stand-in for the embedding model while this was written offline, not the real `all-MiniLM-L6-v2` model, so the exact retrieved `sources` order may differ on your run.

```json
// request 1 — should trigger retrieval (contains "delivery" and "fee")
{"query": "What is the delivery fee if my order is under INR 149?"}
// -> {"answer": "Based on the retrieved context: ...", "sources": ["doc_01", ...], "confidence": 1.0}

// request 2 — should not trigger retrieval (no policy keyword)
{"query": "What's your favorite color?"}
// -> {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
```

### Docker (required, graded baseline — local build/run only)

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How do I cancel an order?"}'
```

### Optional, ungraded: real LLM via Groq

```bash
export MOCK_LLM=0
export GROQ_API_KEY=your-free-tier-key   # https://console.groq.com, no card required
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Architecture — the RAG pipeline, stage by stage

**Ingestion.** `docs/doc_01.txt … doc_08.txt` hold the 8 Zepto policy passages verbatim. `retrieval.load_documents()` reads each file as a single chunk (per-document chunking — each doc is short enough that splitting it further would just fragment one policy across chunks), tagging each chunk with its filename as its id (`doc_01` … `doc_08`).

**Embedding.** `retrieval.get_embedder()` loads `sentence-transformers`' `all-MiniLM-L6-v2` locally (no API key, no per-call network cost after the first download). `retrieval.ingest_documents()` embeds all 8 chunks and stores them, along with their text and a `{"source": doc_id}` metadata tag, in a persistent ChromaDB collection named `zepto_policies` (`chroma_db/` on disk, via `chromadb.PersistentClient`).

**Retrieval.** `retrieval.retrieve(query, k=3)` embeds the incoming query with the same model and calls `collection.query(...)` to get the top-3 chunks by cosine similarity (the collection is created with `hnsw:space: cosine`). This function is called from the `retrieve_and_answer` LangGraph node, and it always runs for real in both `MOCK_LLM` states — embedding a query and querying a local ChromaDB collection needs no API key and no network call, so nothing about retrieval itself is mocked.

**Generation.** `graph_app.py` wires a `langgraph.graph.StateGraph` over a `TypedDict` state (`query, intent, retrieved, answer, sources, confidence`) with three nodes:
- `classify_intent` — in mock mode, a keyword heuristic (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`) decides `policy_question` vs `general_question` with no LLM call; with `MOCK_LLM=0` it asks the LLM instead, using `prompts.build_classify_prompt`.
- A conditional edge (`route_after_classify`) sends `policy_question` to `retrieve_and_answer` and `general_question` to `direct_answer`. This routing decision does not depend on `MOCK_LLM` — only what happens *inside* the two target nodes does.
- `retrieve_and_answer` — calls `retrieval.retrieve()` (always real, as above), then in mock mode returns the canned `f"Based on the retrieved context: {top_chunk_snippet}"` string built from the top hit; with `MOCK_LLM=0` it instead calls the real LLM with `prompts.build_answer_prompt(question, context)`, the structured role/context/task/format/length prompt with its negative constraint and few-shot example.
- `direct_answer` — a fixed canned string in mock mode; with `MOCK_LLM=0`, a direct (non-grounded) LLM call.

**Schema + API.** `graph_app.answer_query()` runs the compiled graph and validates the result against `schemas.AskResponse` (`answer: str`, `sources: list[str]`, `confidence: float` in `[0, 1]`). In mock mode the schema is populated deterministically by code (`sources` = retrieved chunk ids for `policy_question`, empty for `general_question`; `confidence = 1.0`), so there is no LLM output that could fail validation. In the optional `MOCK_LLM=0` path, if the real LLM's output leaves the state unable to validate, `answer_query()` retries the whole graph run up to `MAX_LLM_RETRIES = 2` additional times with a corrective instruction appended to the query, before giving up and returning a clearly marked `"Error: ..."` response. `main.py` wraps this in a FastAPI app with a `POST /ask` endpoint (`schemas.AskRequest` in, `schemas.AskResponse` out) and a `GET /health` check; `@app.on_event("startup")` builds the ChromaDB collection if it doesn't exist yet.

**What changes between the two `MOCK_LLM` states, concretely:**

| | `MOCK_LLM` unset / `1` (graded) | `MOCK_LLM=0` (optional) |
|---|---|---|
| `classify_intent` | keyword heuristic, no LLM call | LLM call via `prompts.build_classify_prompt` |
| `retrieve_and_answer`'s retrieval step | real (ChromaDB + local embeddings) | identical — unaffected |
| `retrieve_and_answer`'s generation step | canned `"Based on the retrieved context: ..."` string | LLM call via `prompts.build_answer_prompt`, grounded in the retrieved chunks |
| `direct_answer` | fixed canned string | direct LLM call, no retrieval |
| Schema population | deterministic, from code | from parsed LLM output, with retry-on-failure |
| Dependencies needed | none beyond the local model | `GROQ_API_KEY` (or another free-tier LLM API) |

## Design decisions

- **Per-document chunking.** Each policy document is one self-contained topic (delivery, returns, membership, …) and is only a few sentences long, so splitting further would separate a policy's sub-clauses (e.g. the ≤INR 1000 vs >INR 1000 rule in `doc_06`) across chunks for no benefit.
- **`sources` empty for `general_question`.** No retrieval happened, so there is nothing to cite; this is enforced by `direct_answer` explicitly setting `sources: []`.
- **Fixed `confidence = 1.0` in mock mode.** There is no model uncertainty to report when the answer is a deterministic template, so a constant is more honest than inventing a number; `MOCK_LLM=0` uses a lower fixed placeholder (0.8 grounded, 0.6 ungrounded) since a real generation step is not deterministic. This is a placeholder — parsing a real confidence estimate from the LLM's own output would be the natural next step, but is out of scope for the graded baseline.
- **Retry lives around the whole graph invocation, not inside a node,** so a failed generation on `MOCK_LLM=0` re-runs classification too; this only matters for the optional path since mock mode never fails validation.

## Known limitations / what you must verify on your machine

- All logic above was unit-tested against hand-written stand-ins for `chromadb`, `sentence-transformers`, `langgraph`, `pydantic` and `fastapi` (no network was available to install or run the real packages while building this). The control flow — classification, routing, mock-mode templating, schema rules, the retry loop, the FastAPI wiring — is confirmed correct against those stand-ins, but you should still run the real thing and check for any version-specific API differences (this module targets `chromadb>=0.5`, `langgraph>=0.2`, `pydantic>=2.6` — pin exact versions in `requirements.txt` if your installed versions behave differently).
- The retrieval *ranking quality* (which chunk comes back first for a given query) could not be verified with the real `all-MiniLM-L6-v2` model here; verify with `python retrieval.py`, which prints the top-3 chunks for two sample queries.
- `sns`-style caching doesn't apply here, but the embedding model itself is cached locally by `sentence-transformers` after its first download, so subsequent runs work offline.
