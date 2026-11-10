"""
Stage 1 — Blueprint: draft the shape of a great answer.

Given a persona-driven query, the LLM produces an "answer blueprint": 4-6 facets
(the aspects a great answer must cover — what/who/where/how/tradeoffs/metrics/
pitfalls, adapted to the query), each with 1-2 BM25-friendly keyword search
queries, plus one anchor query (a <=40-word keyword compression of the whole
narrative).

Anti-drift rules (the naive-SQ lesson, NDCG@10 0.279 vs 0.448): every search
query must contain the topic's core entities, and queries are short keyword
style, not natural-language questions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from trec_shared.llm_json import parse_json_object

_BLUEPRINT_SYSTEM = """You are a research planning expert. Given a complex research request,
you design the BLUEPRINT of an excellent answer: which distinct facets it must cover,
and which corpus search queries will find evidence for each facet.

Rules for search queries:
- Keyword style, 5-15 words, NOT natural-language questions
- EVERY query MUST include the core topic entities (subjects, names, domains) so
  retrieval never drifts off-topic
- Facets must be genuinely distinct aspects (causes, mechanisms, effects, actors,
  tradeoffs, metrics, best practices, pitfalls, comparisons — whatever fits THIS request)"""

_BLUEPRINT_TMPL = """Research request:
{narrative}

Design the answer blueprint. Output a single JSON object:
{{
  "core_entities": ["the 2-4 entities/terms every search query must contain"],
  "facets": [
    {{
      "id": "f1",
      "name": "short facet name",
      "why": "one line: why a great answer needs this facet",
      "queries": ["keyword query 1", "keyword query 2"]
    }}
  ],
  "anchor_query": "one <=40-word keyword compression of the entire request"
}}

Constraints: 4-6 facets, 1-2 queries per facet, every query contains the core entities."""


def draft_blueprint(client: OllamaClient, topic_id: str, narrative: str) -> dict:
    """Draft an answer blueprint for one topic. Returns validated blueprint dict."""
    prompt = _BLUEPRINT_TMPL.format(narrative=narrative[:2500])
    # think=False is mandatory with format="json": thinking mode breaks the JSON
    # grammar on qwen3.5 (verified 2026-07-08: think=True -> 81s + parse fail;
    # think=False -> 11s + clean 5-facet blueprint)
    raw = client.generate(prompt, system=_BLUEPRINT_SYSTEM, think=False,
                          format="json", num_predict=4096)
    bp = parse_json_object(raw, client=client,
                           repair_prompt="This was an answer-blueprint task.")
    return _validate_blueprint(bp, topic_id, narrative)


def _validate_blueprint(bp: dict, topic_id: str, narrative: str) -> dict:
    """Enforce structure; degrade gracefully to a single-facet blueprint."""
    facets = []
    for i, f in enumerate(bp.get("facets", [])[:6]):
        if not isinstance(f, dict):
            continue
        queries = [q.strip() for q in f.get("queries", [])
                   if isinstance(q, str) and q.strip()][:2]
        if not queries:
            continue
        facets.append({
            "id": f.get("id") or f"f{i+1}",
            "name": str(f.get("name", f"facet {i+1}"))[:80],
            "why": str(f.get("why", ""))[:200],
            "queries": queries,
        })

    anchor = bp.get("anchor_query")
    if not isinstance(anchor, str) or len(anchor.split()) < 3:
        # Fallback anchor: first 40 words of the narrative
        anchor = " ".join(narrative.split()[:40])
    anchor = " ".join(anchor.split()[:40])

    if not facets:
        # Total fallback: one facet from the narrative itself
        facets = [{
            "id": "f1", "name": "main request", "why": "fallback — blueprint parse failed",
            "queries": [" ".join(narrative.split()[:15])],
        }]

    return {
        "topic_id": topic_id,
        "core_entities": [str(e)[:60] for e in bp.get("core_entities", [])[:4]],
        "facets": facets,
        "anchor_query": anchor,
    }
