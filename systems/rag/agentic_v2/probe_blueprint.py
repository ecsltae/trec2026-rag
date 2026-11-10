"""
Probe-guided adaptive blueprint (targets track goal #2: memorization vs
generalization).

Per topic:
  1. Zero-RAG draft — the model answers from parametric knowledge only.
     This IS the memorization probe: what the model already "knows".
  2. Probe blueprint — facets are derived FROM the draft:
       mode="verify"  : the draft's key checkable claims -> retrieval finds
                        citations that confirm (or fail to confirm) them
       mode="explore" : aspects of the request the draft did NOT cover ->
                        classic exploratory retrieval
Downstream stages (harvest/tighten/fuse/synthesize) are unchanged — the probe
signal acts purely through query design, so any eval difference vs
bp_agentic_v2 is attributable to it.

Usage:
    python systems/rag/agentic_v2/probe_blueprint.py --topic-set dev22_probe
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from trec_shared.ollama_client import OllamaClient
from trec_shared.llm_json import parse_json_object
from systems.rag.agentic_v2.blueprint import _validate_blueprint
from systems.rag.agentic_v2.pipeline import TOPIC_SETS, load_topic_set, topic_dir

_DRAFT_SYSTEM = """You are a knowledgeable expert. Answer from your own knowledge only.
Write 6-10 factual, specific sentences (numbers, names, mechanisms). No citations, no hedging."""

_DRAFT_TMPL = "Request: {narrative}\n\nYour best answer from memory:"

_PROBE_BP_SYSTEM = """You are a research planning expert running a verify-and-explore strategy.
You are given a research request and a DRAFT answer written from memory. Design the retrieval
blueprint that (a) VERIFIES the draft's key checkable claims against a corpus, and
(b) EXPLORES the parts of the request the draft failed to cover.

Rules for search queries:
- Keyword style, 5-15 words, NOT questions
- Every query MUST include the request's core entities
- verify queries target one specific draft claim each (its entities, numbers, names)
- explore queries target request aspects absent or thin in the draft"""

_PROBE_BP_TMPL = """REQUEST:
{narrative}

DRAFT (from memory — may contain errors):
{draft}

Output a single JSON object:
{{
  "core_entities": ["2-4 terms every query must contain"],
  "facets": [
    {{"id":"f1","name":"...","mode":"verify","why":"which draft claim this checks","queries":["..."]}},
    {{"id":"f2","name":"...","mode":"explore","why":"which request aspect the draft missed","queries":["..."]}}
  ],
  "anchor_query": "<=40-word keyword compression of the request"
}}
Constraints: 4-6 facets total, mix of verify and explore, 1-2 queries each."""


def draft_and_blueprint(client: OllamaClient, topic_id: str, narrative: str) -> tuple:
    draft = client.generate(_DRAFT_TMPL.format(narrative=narrative[:2000]),
                            system=_DRAFT_SYSTEM, think=False, num_predict=1024)
    raw = client.generate(
        _PROBE_BP_TMPL.format(narrative=narrative[:2000], draft=draft[:2500]),
        system=_PROBE_BP_SYSTEM, think=False, format="json", num_predict=4096)
    bp = parse_json_object(raw, client=client,
                           repair_prompt="This was a verify/explore blueprint task.")
    modes = {f.get("id"): f.get("mode", "explore") for f in bp.get("facets", [])
             if isinstance(f, dict)}
    validated = _validate_blueprint(bp, topic_id, narrative)
    for f in validated["facets"]:
        f["mode"] = modes.get(f["id"], "explore")
    return draft, validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-set", default="dev22_probe")
    parser.add_argument("--model", default="qwen3.5:122b")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    topics = load_topic_set(args.topic_set)
    if args.limit:
        topics = dict(sorted(topics.items())[:args.limit])
    client = OllamaClient(model=args.model, temperature=0.15)

    todo = [(tid, t) for tid, t in sorted(topics.items())
            if not (topic_dir(args.topic_set, tid) / "blueprint.json").exists()]
    print(f"[probe-bp] {len(todo)}/{len(topics)} topics to do", flush=True)
    for i, (tid, topic) in enumerate(todo, 1):
        t0 = time.time()
        narrative = topic.get("narrative", "")
        draft, bp = draft_and_blueprint(client, tid, narrative)
        d = topic_dir(args.topic_set, tid)
        (d / "draft.json").write_text(json.dumps(
            {"topic_id": tid, "draft": draft}, indent=2))
        (d / "blueprint.json").write_text(json.dumps(bp, indent=2))
        n_ver = sum(1 for f in bp["facets"] if f.get("mode") == "verify")
        n_exp = len(bp["facets"]) - n_ver
        print(f"[probe-bp] [{i}/{len(todo)}] {tid}: {n_ver} verify + {n_exp} explore "
              f"facets ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
