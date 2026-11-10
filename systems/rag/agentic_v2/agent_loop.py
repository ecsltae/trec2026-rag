"""
bp_agentic_v3 — true tool-calling agent loop.

Instead of the fixed fuse/synthesize stages, the LLM itself researches each
topic over its harvested pool: it searches, reads documents, takes cited notes
facet-by-facet, and declares finish. The Python harness enforces budgets and
citation validity (ToolBox rejects citations to unseen docs).

Reuses v2 stages 1-3 (blueprint, pool, embeddings) — run those phases first.
Embedder runs on CPU: query encoding is one short sentence, and this keeps the
whole GPU for the 122b.

Usage:
    python systems/rag/agentic_v2/agent_loop.py --topic-set dev22 [--limit 3]
        [--model qwen3.5:122b] [--max-tool-calls 15] [--max-turns 12]

Outputs per topic (cache dir): answer_v3.json, agent_trace.jsonl
Run files: runs/rag/bp_agentic_v3_{topic_set}.jsonl / .tsv
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from trec_shared.ollama_client import OllamaClient
from systems.rag.agentic_v2.tools import ToolBox, TOOL_SCHEMAS
from systems.rag.agentic_v2.pipeline import (CACHE_ROOT, RUNS_DIR, TOPIC_SETS,
                                             load_topic_set, topic_dir)

RUN_ID = "bp_agentic_v3"

_AGENT_SYSTEM = """You are a meticulous research agent. Your mission: research a complex
request using the provided tools, then produce findings as cited notes.

Method:
1. Work facet by facet. For each facet: search_pool with a targeted query, then
   take_note with 1-2 factual sentences citing the docids you used.
   Only use read_doc when the search snippets are too thin to write a specific note.
2. Notes must be specific (numbers, mechanisms, named actors) — never generic filler.
3. Every facet needs at least one note before you finish.
4. Cite ONLY docids you have seen in search results or read. Invalid citations are rejected.
5. When every facet is covered, call finish. Cover ALL facets before deepening any one.

Be efficient: you have a limited tool budget. Two tool calls per facet
(search + note) is the ideal rhythm."""

_AGENT_KICKOFF = """REQUEST:
{narrative}

FACETS to research (cover every one):
{facet_lines}

Tool budget: {budget} calls. Begin researching now."""


def run_agent_topic(client: OllamaClient, d: Path, topic_id: str, narrative: str,
                    embedder, max_tool_calls: int = 15, max_turns: int = 12) -> dict:
    bp = json.loads((d / "blueprint.json").read_text())
    box = ToolBox(d, embedder)
    trace_path = d / "agent_trace.jsonl"
    trace = open(trace_path, "w")

    def log(event: dict):
        trace.write(json.dumps(event) + "\n")
        trace.flush()

    facet_lines = "\n".join(f"- {f['id']} ({f['name']}): {f['why']}"
                            for f in bp["facets"])
    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": _AGENT_KICKOFF.format(
            narrative=narrative[:2000], facet_lines=facet_lines,
            budget=max_tool_calls)},
    ]

    tool_calls_used = 0
    nudged = False
    for turn in range(max_turns):
        content, tool_calls = client.chat(messages, tools=TOOL_SCHEMAS,
                                          think=False, num_predict=1024)
        log({"turn": turn, "content": content[:500],
             "tool_calls": [tc["name"] for tc in tool_calls]})

        if not tool_calls:
            if box.finished or box.notes and nudged:
                break
            # One nudge, then give up on the loop
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "Use take_note to record findings for uncovered facets, "
                             "or call finish if every facet has a note."})
            nudged = True
            continue

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [
                             {"function": {"name": tc["name"],
                                           "arguments": tc["arguments"]}}
                             for tc in tool_calls]})
        for tc in tool_calls:
            if tool_calls_used >= max_tool_calls:
                result = json.dumps({"error": "tool budget exhausted — "
                                              "call finish now"})
            else:
                result = box.execute(tc["name"], tc["arguments"])
                tool_calls_used += 1
            log({"turn": turn, "tool": tc["name"], "args": tc["arguments"],
                 "result": result[:300]})
            messages.append({"role": "tool", "content": result,
                             "tool_name": tc["name"]})

        if box.finished:
            break

    trace.close()

    # Notes -> answer (ordered by blueprint facet order)
    facet_order = {f["id"]: i for i, f in enumerate(bp["facets"])}
    notes = sorted(box.notes, key=lambda n: facet_order.get(n["facet_id"], 99))
    answer = [{"text": n["text"], "citations": n["citations"]} for n in notes]

    # FALLBACK when the agent harvested nothing. Without this, a topic where the loop burned its
    # whole tool budget without landing a note silently produced {"answer": [], "references": []}.
    # Eight such empty records shipped in bp_agentic_v3_official119 (rag2026-5, -31, -55, -69, -70,
    # -78, -82, -95), each with tool_calls_used=20, notes=0, finished_cleanly=false. Prefer the v2
    # synthesis already cached for the same topic; if that is missing too, flag the record rather
    # than emitting a silent blank.
    fallback = None
    if not answer:
        v2_path = d / "answer.json"
        if v2_path.exists():
            try:
                v2 = json.loads(v2_path.read_text())
                if v2.get("answer"):
                    answer = v2["answer"]
                    fallback = "v2_synthesis"
            except (json.JSONDecodeError, OSError):
                pass
        if not answer:
            fallback = "NONE_AVAILABLE"
        print(f"[v3] {topic_id}: 0 notes harvested -> fallback={fallback}", flush=True)

    covered = {n["facet_id"] for n in box.notes}
    result = {
        "topic_id": topic_id,
        "answer": answer,
        "diagnostics": {
            "tool_calls_used": tool_calls_used,
            "turns": turn + 1,
            "notes": len(box.notes),
            "facets_covered": len(covered),
            "n_facets": len(bp["facets"]),
            "finished_cleanly": box.finished,
            "docs_seen": len(box.seen_docids),
            "fallback": fallback,
        },
    }
    (d / "answer_v3.json").write_text(json.dumps(result, indent=2))
    return result


def assemble(topics: dict, topic_set: str):
    """Build v3 run files: JSONL from notes; TSV = cited docs, seen docs, v2 tail."""
    rows, answers = [], []
    for tid in sorted(topics):
        d = CACHE_ROOT / topic_set / tid
        ans_path = d / "answer_v3.json"
        if not ans_path.exists():
            continue
        a = json.loads(ans_path.read_text())
        answers.append(a)

        cited, seen_cited = [], set()
        for s in a["answer"]:
            for c in s["citations"]:
                if c not in seen_cited:
                    seen_cited.add(c)
                    cited.append(c)
        ranking = [[docid, 1000.0 - i] for i, docid in enumerate(cited)]
        # Fill with the v2 final ranking (drops docs already listed)
        fr = d / "final_ranking.json"
        if fr.exists():
            for docid, _ in json.loads(fr.read_text()):
                if docid not in seen_cited and len(ranking) < 1000:
                    seen_cited.add(docid)
                    ranking.append([docid, 1000.0 - len(ranking)])
        for rank, (docid, score) in enumerate(ranking, 1):
            rows.append((tid, docid, rank, score))

    from trec_shared.trec_utils import write_trec_run
    out_tsv = RUNS_DIR / f"{RUN_ID}_{topic_set}.tsv"
    write_trec_run(rows, str(out_tsv), RUN_ID)

    # BOUNDARY CHECK: never let an empty answer reach a run file unannounced. The per-topic
    # fallback above should prevent this, but eight empty records once shipped all the way into a
    # submission candidate and were only caught by a later audit, so fail loudly here too.
    empty_topics = [a["topic_id"] for a in answers if not a.get("answer")]
    if empty_topics:
        print(f"[v3 assemble] *** WARNING: {len(empty_topics)} EMPTY answers will be written: "
              f"{empty_topics} — these will fail submission validation "
              f"(eval/validate_submission.py) and must be fixed before use.", flush=True)

    out_jsonl = RUNS_DIR / f"{RUN_ID}_{topic_set}.jsonl"
    with open(out_jsonl, "w") as f:
        for a in answers:
            refs, seen = [], set()
            for s in a["answer"]:
                for c in s["citations"]:
                    if c not in seen:
                        seen.add(c)
                        refs.append(c)
            sentences = [{"text": s["text"],
                          "citations": [refs.index(c) for c in s["citations"]]}
                         for s in a["answer"]]
            f.write(json.dumps({
                "metadata": {"topic_id": a["topic_id"], "run_id": RUN_ID,
                             "type": "automatic"},
                "references": refs,
                "answer": sentences,
            }) + "\n")
    print(f"[v3 assemble] {len(rows)} TSV rows -> {out_tsv}")
    print(f"[v3 assemble] {len(answers)} answers -> {out_jsonl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-set", required=True, choices=list(TOPIC_SETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen3.5:122b")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=26,
                        help="Must exceed max-tool-calls: the model makes ~1 call/turn")
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()

    topics = load_topic_set(args.topic_set)
    if args.limit:
        topics = dict(sorted(topics.items())[:args.limit])

    if not args.assemble_only:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embed_model, device="cpu")
        client = OllamaClient(model=args.model, temperature=0.2)

        todo = [tid for tid in sorted(topics)
                if (topic_dir(args.topic_set, tid) / "emb_docids.json").exists()
                and not (topic_dir(args.topic_set, tid) / "answer_v3.json").exists()]
        print(f"[v3] {len(todo)}/{len(topics)} topics to do", flush=True)
        for i, tid in enumerate(todo, 1):
            t0 = time.time()
            d = topic_dir(args.topic_set, tid)
            result = run_agent_topic(client, d, tid, topics[tid].get("narrative", ""),
                                     embedder, args.max_tool_calls, args.max_turns)
            diag = result["diagnostics"]
            print(f"[v3] [{i}/{len(todo)}] {tid}: {diag['notes']} notes, "
                  f"{diag['facets_covered']}/{diag['n_facets']} facets, "
                  f"{diag['tool_calls_used']} tools, "
                  f"finished={diag['finished_cleanly']} ({time.time()-t0:.0f}s)",
                  flush=True)

    assemble(topics, args.topic_set)


if __name__ == "__main__":
    main()
