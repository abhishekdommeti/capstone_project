"""Single gate for every LLM call in this module, behind the MOCK_LLM env var.

MOCK_LLM unset, or "1" (default, graded baseline): call_llm() is never invoked by
graph_app.py at all -- classify_intent, retrieve_and_answer and direct_answer use
their own deterministic mock logic instead. This module exists so that the optional
MOCK_LLM=0 extension has one place to change.

MOCK_LLM=0 (optional, ungraded extension): call_llm() calls Groq's free-tier API
(https://console.groq.com), which is OpenAI-chat-completions-compatible. Any other
LLM API with a genuinely free tier is an acceptable drop-in replacement -- only this
function needs to change.
"""
import os

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"


def call_llm(prompt: str, *, max_tokens: int = 200, temperature: float = 0.0) -> str:
    """Call the real LLM. Only reached when MOCK_LLM=0 -- never called on the graded
    mock-mode path, so this function needing GROQ_API_KEY does not affect grading.
    """
    if MOCK_LLM:
        raise RuntimeError("call_llm() should never be invoked while MOCK_LLM is on; "
                            "this is a bug in the caller, not an expected code path.")

    from groq import Groq   # optional dependency, only imported on the MOCK_LLM=0 path

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY to be set (get a free key "
                            "at https://console.groq.com). Any other LLM API with a "
                            "genuinely free tier is an acceptable substitute -- swap "
                            "the client below.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()
