"""
bp_agentic_v2 pipeline orchestrator.

Fixed 5-stage batch pipeline with per-topic, per-stage disk checkpoints:
stage output file existence == checkpoint (resume is the default behavior).

Usage:
    python systems/rag/agentic_v2/pipeline.py --phase blueprint --topic-set dev22
    python systems/rag/agentic_v2/pipeline.py --phase harvest   --topic-set dev22
    python systems/rag/agentic_v2/pipeline.py --phase tighten   --topic-set dev22
    python systems/rag/agentic_v2/pipeline.py --phase fuse      --topic-set dev22
    python systems/rag/agentic_v2/pipeline.py --phase synthesize --topic-set dev22
    python systems/rag/agentic_v2/pipeline.py --phase assemble  --topic-set dev22

Topic sets:
    dev22       — the 22 dev topics with qrels (validation)
    dev105      — all 105 dev topics
    official119 — the official TREC RAG 2026 test topics (submission)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from trec_shared.trec_utils import load_topics, load_qrels, write_trec_run
from trec_shared.ollama_client import OllamaClient

CACHE_ROOT = ROOT / "data/rag/cache/agentic_v2"
RUNS_DIR = ROOT / "runs/rag"

TOPIC_SETS = {
    "dev22": {"topics": "data/rag/topics/trec25_narratives_final.json", "qrels_only": True},
    "dev105": {"topics": "data/rag/topics/trec25_narratives_final.json", "qrels_only": False},
    "official119": {"topics": "data/rag/topics/trec_rag_2026_queries.tsv", "qrels_only": False},
    # Probe-guided variant caches (same topics, separate cache dirs; blueprints
    # come from probe_blueprint.py — verify/explore facets)
    "dev22_probe": {"topics": "data/rag/topics/trec25_narratives_final.json", "qrels_only": True},
    "official119_probe": {"topics": "data/rag/topics/trec_rag_2026_queries.tsv", "qrels_only": False},
}

RUN_ID = "bp_agentic_v2"


def load_topic_set(name: str) -> dict:
    cfg = TOPIC_SETS[name]
    topics = load_topics(str(ROOT / cfg["topics"]))
    if cfg["qrels_only"]:
        qrels_path = next((ROOT / "data/rag/qrels").glob("*.qrels"))
        eval_ids = set(load_qrels(str(qrels_path)).keys())
        topics = {tid: t for tid, t in topics.items() if tid in eval_ids}
    return topics


def topic_dir(topic_set: str, topic_id: str) -> Path:
    d = CACHE_ROOT / topic_set / topic_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_blueprint(topics: dict, topic_set: str, args):
    from systems.rag.agentic_v2.blueprint import draft_blueprint
    client = OllamaClient(model=args.blueprint_model, temperature=0.1)
    todo = [(tid, t) for tid, t in sorted(topics.items())
            if not (topic_dir(topic_set, tid) / "blueprint.json").exists()]
    print(f"[blueprint] {len(todo)}/{len(topics)} topics to do", flush=True)
    for i, (tid, topic) in enumerate(todo, 1):
        t0 = time.time()
        narrative = topic.get("narrative", "")
        bp = draft_blueprint(client, tid, narrative)
        out = topic_dir(topic_set, tid) / "blueprint.json"
        out.write_text(json.dumps(bp, indent=2))
        n_q = sum(len(f["queries"]) for f in bp["facets"])
        print(f"[blueprint] [{i}/{len(todo)}] {tid}: {len(bp['facets'])} facets, "
              f"{n_q} queries ({time.time()-t0:.0f}s)", flush=True)
    if args.unload_after and todo:
        client.unload()
        print("[blueprint] model unloaded", flush=True)


def phase_harvest(topics: dict, topic_set: str, args):
    from systems.rag.retrieve import ApiRetriever
    from systems.rag.agentic_v2.harvest import harvest_topic, write_pool
    retriever = ApiRetriever()
    todo = [(tid, t) for tid, t in sorted(topics.items())
            if (topic_dir(topic_set, tid) / "blueprint.json").exists()
            and not (topic_dir(topic_set, tid) / "pool.jsonl").exists()]
    print(f"[harvest] {len(todo)}/{len(topics)} topics to do", flush=True)
    for i, (tid, topic) in enumerate(todo, 1):
        t0 = time.time()
        d = topic_dir(topic_set, tid)
        bp = json.loads((d / "blueprint.json").read_text())
        entries = harvest_topic(retriever, bp, topic.get("narrative", ""),
                                hits_narrative=args.hits_narrative,
                                hits_anchor=args.hits_anchor,
                                hits_facet=args.hits_facet)
        write_pool(d / "pool.jsonl", entries)
        print(f"[harvest] [{i}/{len(todo)}] {tid}: pool={len(entries)} docs "
              f"({time.time()-t0:.0f}s)", flush=True)


def phase_tighten(topics: dict, topic_set: str, args):
    from systems.rag.agentic_v2.tighten import TightenStage
    stage = TightenStage(embed_model=args.embed_model,
                         rerank_model=args.rerank_model)
    todo = [tid for tid in sorted(topics)
            if (topic_dir(topic_set, tid) / "pool.jsonl").exists()
            and not (topic_dir(topic_set, tid) / "facet_rankings.json").exists()]
    print(f"[tighten] {len(todo)}/{len(topics)} topics to do", flush=True)
    for i, tid in enumerate(todo, 1):
        t0 = time.time()
        d = topic_dir(topic_set, tid)
        narrative = topics[tid].get("narrative", "")
        stage.run_topic(d, narrative)
        print(f"[tighten] [{i}/{len(todo)}] {tid} done ({time.time()-t0:.0f}s)",
              flush=True)


def phase_fuse(topics: dict, topic_set: str, args):
    from systems.rag.agentic_v2.fuse import fuse_topic
    todo = [tid for tid in sorted(topics)
            if (topic_dir(topic_set, tid) / "facet_rankings.json").exists()]
    # fuse is cheap — always recompute (it's the tuning knob surface)
    print(f"[fuse] {len(todo)}/{len(topics)} topics", flush=True)
    for tid in todo:
        d = topic_dir(topic_set, tid)
        fuse_topic(d, anchor_weight=args.anchor_weight,
                   slot_guarantee=args.slot_guarantee, top_k=args.final_top_k)
    print(f"[fuse] done ({len(todo)} topics)", flush=True)


def phase_synthesize(topics: dict, topic_set: str, args):
    from systems.rag.agentic_v2.synthesize import synthesize_topic
    client = OllamaClient(model=args.synth_model, temperature=0.15)
    todo = [tid for tid in sorted(topics)
            if (topic_dir(topic_set, tid) / "final_ranking.json").exists()
            and not (topic_dir(topic_set, tid) / "answer.json").exists()]
    print(f"[synthesize] {len(todo)}/{len(topics)} topics to do", flush=True)
    for i, tid in enumerate(todo, 1):
        t0 = time.time()
        d = topic_dir(topic_set, tid)
        narrative = topics[tid].get("narrative", "")
        result = synthesize_topic(client, d, tid, narrative)
        n_sent = len(result.get("answer", []))
        print(f"[synthesize] [{i}/{len(todo)}] {tid}: {n_sent} sentences "
              f"({time.time()-t0:.0f}s)", flush=True)


def phase_assemble(topics: dict, topic_set: str, args):
    """Build run files idempotently from caches."""
    rows = []
    answers = []
    missing_rank, missing_ans = [], []
    for tid in sorted(topics):
        d = CACHE_ROOT / topic_set / tid
        fr = d / "final_ranking.json"
        if fr.exists():
            ranking = json.loads(fr.read_text())
            for rank, (docid, score) in enumerate(ranking, 1):
                rows.append((tid, docid, rank, score))
        else:
            missing_rank.append(tid)
        ans = d / "answer.json"
        if ans.exists():
            answers.append(json.loads(ans.read_text()))
        else:
            missing_ans.append(tid)

    out_tsv = RUNS_DIR / f"{RUN_ID}_{topic_set}.tsv"
    write_trec_run(rows, str(out_tsv), RUN_ID)
    print(f"[assemble] R TSV: {len(rows)} rows -> {out_tsv}", flush=True)

    out_jsonl = RUNS_DIR / f"{RUN_ID}_{topic_set}.jsonl"
    with open(out_jsonl, "w") as f:
        for a in answers:
            refs = []
            seen = set()
            for s in a.get("answer", []):
                for c in s.get("citations", []):
                    if c not in seen:
                        seen.add(c)
                        refs.append(c)
            sentences = [{"text": s["text"],
                          "citations": [refs.index(c) for c in s.get("citations", [])]}
                         for s in a.get("answer", [])]
            f.write(json.dumps({
                "metadata": {"topic_id": a["topic_id"], "run_id": RUN_ID,
                             "type": "automatic"},
                "references": refs,
                "answer": sentences,
            }) + "\n")
    print(f"[assemble] RAG JSONL: {len(answers)} topics -> {out_jsonl}", flush=True)
    if missing_rank:
        print(f"[assemble] WARNING missing rankings: {missing_rank}", flush=True)
    if missing_ans:
        print(f"[assemble] WARNING missing answers: {missing_ans}", flush=True)


PHASES = {
    "blueprint": phase_blueprint,
    "harvest": phase_harvest,
    "tighten": phase_tighten,
    "fuse": phase_fuse,
    "synthesize": phase_synthesize,
    "assemble": phase_assemble,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PHASES) + ["all"])
    parser.add_argument("--topic-set", required=True, choices=list(TOPIC_SETS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N topics (smoke tests)")
    # Models
    parser.add_argument("--blueprint-model", default="qwen3.5:122b")
    parser.add_argument("--synth-model", default="qwen3.5:122b")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Harvest knobs
    parser.add_argument("--hits-narrative", type=int, default=500)
    parser.add_argument("--hits-anchor", type=int, default=300)
    parser.add_argument("--hits-facet", type=int, default=200)
    # Fuse knobs
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--slot-guarantee", type=int, default=2,
                        help="Force each facet's top-N docs into the final top-k (0=off)")
    parser.add_argument("--final-top-k", type=int, default=100)
    # GPU hygiene
    parser.add_argument("--unload-after", action="store_true", default=True,
                        help="Unload Ollama model after LLM phases (VRAM for embedder)")
    args = parser.parse_args()

    topics = load_topic_set(args.topic_set)
    if args.limit:
        topics = dict(sorted(topics.items())[:args.limit])
    print(f"=== {RUN_ID} · phase={args.phase} · {args.topic_set} "
          f"({len(topics)} topics) ===", flush=True)

    if args.phase == "all":
        for name in ["blueprint", "harvest", "tighten", "fuse", "synthesize", "assemble"]:
            PHASES[name](topics, args.topic_set, args)
    else:
        PHASES[args.phase](topics, args.topic_set, args)


if __name__ == "__main__":
    main()
