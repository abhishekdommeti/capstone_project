"""LangGraph orchestration for the Zepto support assistant (Task 3, 4).

Graph shape:

    classify_intent --(policy_question)--> retrieve_and_answer --> END
                    \\-(general_question)--> direct_answer      --> END

- classify_intent: keyword heuristic in mock mode (graded baseline); LLM call when
  MOCK_LLM=0. No LLM call is ever made in mock mode.
- retrieve_and_answer: retrieval (embed query + ChromaDB top-3) always runs for real
  in both modes, since it needs no API key/network call. Only the final
  answer-generation step branches on MOCK_LLM.
- direct_answer: canned string in mock mode; LLM call when MOCK_LLM=0.

The conditional edge out of classify_intent does not depend on MOCK_LLM -- only the
generation step inside each node does.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from llm import MOCK_LLM, call_llm
from prompts import build_answer_prompt, build_classify_prompt
from retrieval import retrieve
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]

FALLBACK_ANSWER = "I can only answer questions about Zepto policies right now."
TOP_CHUNK_SNIPPET_CHARS = 200
MAX_LLM_RETRIES = 2                # "up to 2 additional times" beyond the first attempt


class AssistantState(TypedDict, total=False):
    query: str
    intent: str                    # "policy_question" | "general_question"
    retrieved: list[dict]          # [{"id","text","distance"}, ...]
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1 — classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]

    if MOCK_LLM:
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify instead.
        raw = call_llm(build_classify_prompt(query), max_tokens=5).strip().lower()
        intent = "policy_question" if "policy_question" in raw else "general_question"

    return {**state, "intent": intent}


def route_after_classify(state: AssistantState) -> str:
    """Conditional edge target. Pure routing logic -- does not depend on MOCK_LLM."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2 — retrieve_and_answer (policy_question path)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: AssistantState) -> AssistantState:
    query = state["query"]

    # Retrieval always runs for real, in both modes: local embeddings + ChromaDB,
    # no API key, no network call.
    hits = retrieve(query, k=3)
    sources = [h["id"] for h in hits]

    if MOCK_LLM:
        # Mock mode (graded baseline): canned template built from the top chunk.
        top_chunk_snippet = hits[0]["text"][:TOP_CHUNK_SNIPPET_CHARS] if hits else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM, grounded only in the
        # retrieved chunks, using the structured template from prompts.py.
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        answer = call_llm(build_answer_prompt(query, context))
        confidence = 0.8

    return {**state, "retrieved": hits, "answer": answer, "sources": sources, "confidence": confidence}


# ---------------------------------------------------------------------------
# Node 3 — direct_answer (general_question path)
# ---------------------------------------------------------------------------
def direct_answer(state: AssistantState) -> AssistantState:
    if MOCK_LLM:
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer = FALLBACK_ANSWER
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        answer = call_llm(
            f"Answer this general question briefly and helpfully: {state['query']}"
        )
        confidence = 0.6

    return {**state, "retrieved": [], "answer": answer, "sources": [], "confidence": confidence}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Schema validation (Task 4)
# ---------------------------------------------------------------------------
def _to_validated_response(state: AssistantState) -> AskResponse:
    """Validate the graph's output against the Pydantic schema.

    Mock mode populates the schema deterministically from code, so there is no LLM
    output to fail validation -- validation here is a structural safety net, not a
    retry target. The retry-on-failure loop lives in answer_query() below, around the
    whole graph invocation, so it can re-run generation when MOCK_LLM=0 and the real
    LLM's output does not fit the schema.
    """
    return AskResponse(
        answer=state.get("answer", FALLBACK_ANSWER),
        sources=state.get("sources", []),
        confidence=state.get("confidence", 0.0),
    )


def answer_query(query: str) -> AskResponse:
    """Run the graph end-to-end and return a schema-validated AskResponse.

    Mock mode: a single graph run always produces a valid schema (nothing to retry).
    MOCK_LLM=0: if the real LLM's raw output leaves the state unable to validate
    against AskResponse, retry up to MAX_LLM_RETRIES additional times with a
    corrective instruction appended to the query before giving up.
    """
    attempt_query = query
    last_error = None

    for attempt in range(1 + (MAX_LLM_RETRIES if not MOCK_LLM else 0)):
        state = get_graph().invoke({"query": attempt_query})
        try:
            return _to_validated_response(state)
        except ValidationError as exc:
            last_error = exc
            attempt_query = (
                f"{query}\n\n(Note: your previous answer could not be parsed into the "
                f"required answer/sources/confidence format. Please answer again, "
                f"following the format instructions exactly.)"
            )

    return AskResponse(
        answer=f"Error: could not produce a schema-valid answer after {MAX_LLM_RETRIES} retries "
               f"({last_error}).",
        sources=[],
        confidence=0.0,
    )
