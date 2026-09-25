"""Generates the >=2 example-call transcripts required in the README (Task 5).

Runs the graph directly (no running server needed) with MOCK_LLM left at its
default, and writes results/example_calls.md with the raw JSON for:
  1. a query that should trigger retrieval (contains a policy keyword)
  2. a query that should not (no policy keyword)

Usage (from /support_assistant):
    python generate_example_calls.py
"""
import json
import os

from graph_app import answer_query

os.makedirs("results", exist_ok=True)

EXAMPLES = [
    "What is the delivery fee if my order is under INR 149?",
    "What's your favorite color?",
]

lines = ["# Example /ask calls (MOCK_LLM left at its default)\n"]
for query in EXAMPLES:
    response = answer_query(query)
    payload = {"query": query}
    result = response.model_dump()
    lines.append(f"### Request\n```json\n{json.dumps(payload, indent=2)}\n```")
    lines.append(f"### Response\n```json\n{json.dumps(result, indent=2)}\n```\n")
    print(query, "->", result)

with open("results/example_calls.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nsaved results/example_calls.md")
