"""
Local RAGDoll-compatible nugget-coverage judge.

The official generation measures are nugget/rubric coverage (vital_strict, all_strict), not
citation support. This scores answers the same way: RAGDoll's OFFICIAL assigner prompt, label
parser and metric code are imported directly from eval/RAGDoll; only the LLM executor is swapped
(local Ollama instead of the organisers' GPT-6.1 Sol). The answer text is the "passage", nuggets
are labelled in windows of 10, a window that never parses counts as not_support (as in RAGDoll).

Nuggets: the organisers' official dev nuggets for the 22 dev topics
(trec-rag-data/trec-rag-2026/development-data/rag25-dev-nuggets), human-written, vital/okay.
There are no distributed nuggets for the official 119 topics, so this scores dev22 only.

Usage:
    python eval/nugget_local_judge.py --run runs/rag/bp_agentic_v2_dev22.jsonl [more runs ...] \
        [--judge-model qwen3:32b] [--out eval/analysis/nugget_dev22_<name>.json]

Outputs:
    <run>.nugget_assignments.jsonl  — RAGDoll assignment-format rows (one per topic)
    stdout                          — per-run means of strict_vital / strict_all, words per answer
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from ragdoll.nuggetizer.prompts import NUGGET_ASSIGNER_SYSTEM, render_assign_prompt, parse_label_list
from ragdoll.nuggetizer.metrics import calculate_scores

from trec_shared.ollama_client import OllamaClient
from trec_shared.trec_utils import load_topics

NUGGETS = ROOT / "data/rag/nuggets/rag25-dev-nuggets.jsonl"
TOPICS = ROOT / "data/rag/topics/trec25_narratives_final.json"
JUDGE_CACHE = ROOT / "data/rag/cache/nugget_assignments.jsonl"
WINDOW = 10      # RAGDoll DEFAULT_WINDOW_SIZE
MAX_TRIALS = 4   # RAGDoll DEFAULT_MAX_TRIALS


def load_judge_cache() -> dict:
    cache = {}
    if JUDGE_CACHE.exists():
        for line in open(JUDGE_CACHE):
            try:
                obj = json.loads(line)
                cache[obj["key"]] = obj["labels"]
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def cache_key(model: str, query: str, context: str, nugget_texts: list) -> str:
    blob = json.dumps([model, query, context, nugget_texts], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def assign_window(client: OllamaClient, query: str, context: str, nugget_texts: list) -> list:
    """Labels for one window, or ["failed"] * n after MAX_TRIALS parse misses."""
    prompt = render_assign_prompt(query=query, context=context, nuggets=nugget_texts,
                                  assign_mode="support-grade-3")
    for trial in range(MAX_TRIALS):
        raw = client.generate(prompt, system=NUGGET_ASSIGNER_SYSTEM, think=False,
                              num_predict=512, temperature=0.0 if trial == 0 else 0.3)
        labels = parse_label_list(raw)
        if labels is not None and len(labels) == len(nugget_texts):
            return [label.strip().lower() for label in labels]
    return ["failed"] * len(nugget_texts)


def judge_run(run_path: Path, client: OllamaClient, topics: dict, nuggets: dict,
              cache: dict, cache_fh) -> list:
    rows = []
    for line in open(run_path):
        rec = json.loads(line)
        tid = rec["metadata"]["topic_id"]
        if tid not in nuggets:
            continue
        query = topics[tid]["narrative"]
        context = " ".join(s["text"] for s in rec["answer"])
        items = nuggets[tid]
        labels = []
        for start in range(0, len(items), WINDOW):
            texts = [n["text"] for n in items[start:start + WINDOW]]
            key = cache_key(client.model, query, context, texts)
            if key not in cache:
                cache[key] = assign_window(client, query, context, texts)
                cache_fh.write(json.dumps({"key": key, "labels": cache[key]}) + "\n")
                cache_fh.flush()
            labels.extend(cache[key])
        assigned = [dict(n, assignment=lab) for n, lab in zip(items, labels)]
        rows.append({"run_id": rec["metadata"]["run_id"], "qid": tid, "nuggets": assigned,
                     "scores": calculate_scores(assigned), "words": len(context.split()),
                     "n_failed": labels.count("failed")})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument("--judge-model", default="qwen3:32b")
    parser.add_argument("--out", default=None, help="summary JSON (per-run, per-topic scores)")
    parser.add_argument("--unload-after", action="store_true")
    args = parser.parse_args()

    topics = load_topics(str(TOPICS))
    nuggets = {}
    for line in open(NUGGETS):
        rec = json.loads(line)
        nuggets[str(rec["qid"])] = rec["nuggets"]

    client = OllamaClient(model=args.judge_model, temperature=0.0, timeout=600)
    cache = load_judge_cache()
    JUDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    summary = {"judge_model": args.judge_model, "runs": {}}
    with open(JUDGE_CACHE, "a") as cache_fh:
        for run in args.run:
            run_path = Path(run)
            rows = judge_run(run_path, client, topics, nuggets, cache, cache_fh)
            out_rows = run_path.with_suffix(".nugget_assignments.jsonl")
            out_rows.write_text("".join(json.dumps(r) + "\n" for r in rows))
            per_topic = {r["qid"]: dict(r["scores"], words=r["words"]) for r in rows}
            means = {m: statistics.mean(r["scores"][m] for r in rows)
                     for m in ("strict_vital_score", "strict_all_score")}
            summary["runs"][run_path.stem] = {"n_topics": len(rows), **means,
                                              "median_words": statistics.median(r["words"] for r in rows),
                                              "n_failed": sum(r["n_failed"] for r in rows),
                                              "per_topic": per_topic}
            print(f"{run_path.stem:40s} topics {len(rows):2d}  vital_strict {means['strict_vital_score']:.4f}  "
                  f"all_strict {means['strict_all_score']:.4f}  median words "
                  f"{summary['runs'][run_path.stem]['median_words']:.0f}  failed labels "
                  f"{summary['runs'][run_path.stem]['n_failed']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    if args.unload_after:
        client.unload()


if __name__ == "__main__":
    main()
