"""
Local RAGDoll-compatible support judge.

Uses RAGDoll's OFFICIAL prompt, label parser, and metric code (imported directly
from eval/RAGDoll) — only the LLM executor is swapped: local Ollama instead of
Pi/gpt-5.5. Verdict mapping: Full Support=2, Partial=1, No=0 (RAGDoll's
WEIGHTED_SCORES: 2->1.0, 1->0.5, 0->0.0).

Usage:
    python eval/ragdoll_local_judge.py \
        --run runs/rag/bp_agentic_v2_dev22.jsonl \
        [--judge-model qwen3:32b] [--cache-root data/rag/cache/agentic_v2/dev22]

Outputs:
    <run>.support_judgments.jsonl   — RAGDoll metrics-compatible rows
    stdout                          — per-topic + mean weighted/hard precision
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from ragdoll.support.prompts import render_support_prompt, parse_support_label
from ragdoll.support.metrics import support_metric

from trec_shared.ollama_client import OllamaClient

LABEL_TO_SCORE = {"FS": 2, "PS": 1, "NS": 0}
JUDGE_CACHE = ROOT / "data/rag/cache/support_judgments.jsonl"


def load_judge_cache() -> dict:
    cache = {}
    if JUDGE_CACHE.exists():
        for line in open(JUDGE_CACHE):
            try:
                obj = json.loads(line)
                cache[obj["key"]] = obj["support"]
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def cache_key(statement: str, citation_text: str, model: str,
              max_citation_chars: int = 4000) -> str:
    """Cache key for one (statement, citation, model) support judgement.

    max_citation_chars MUST be part of the key. The judge is shown citation_text[:max_citation_chars],
    so the same statement and document judged at two widths are two different questions. Without it,
    re-judging at a wider width silently returns the narrow verdicts — observed as "0 new, 66 cached"
    on a pass that should have been entirely new, which made a truncation-artifact test read as a
    clean negative result. The default reproduces the historical key byte-for-byte so the existing
    cache stays valid; only non-default widths get a new namespace.

    KNOWN LIMITATION, deliberately left in place: only the first 2000 characters of the citation are
    hashed, so two DIFFERENT documents sharing a 2000-character prefix collide and inherit each
    other's verdict when cited for the same statement. Measured across the citations we ship: 142
    distinct (statement, prefix) keys, 0 collisions. Widening the hash would invalidate every cached
    judgement, so it is documented rather than changed. Re-measure before trusting the cache on a run
    with near-duplicate or syndicated sources.
    """
    suffix = "" if max_citation_chars == 4000 else f"||w{max_citation_chars}"
    h = hashlib.sha256(f"{model}||{statement}||{citation_text[:2000]}{suffix}".encode())
    return h.hexdigest()[:32]


def resolve_text(docid: str, cache_root: Path, api_retriever=None) -> str:
    """Harvest cache first, API fallback."""
    if cache_root:
        for pool_path in cache_root.glob(f"*/pool.jsonl"):
            pass  # too slow to scan every pool per docid — use the index built below
    if api_retriever is not None:
        return api_retriever.get_text(docid) or ""
    return ""


def build_text_index(cache_root: Path) -> dict:
    """docid -> text from all pool.jsonl files under cache_root."""
    index = {}
    if cache_root and cache_root.exists():
        for pool_path in cache_root.glob("*/pool.jsonl"):
            for line in open(pool_path):
                try:
                    e = json.loads(line)
                    index.setdefault(e["docid"], e["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return index


def judge_run(run_path: Path, judge_model: str, cache_root: Path = None,
              max_citation_chars: int = 4000) -> list:
    client = OllamaClient(model=judge_model, temperature=0.0)
    judge_cache = load_judge_cache()
    text_index = build_text_index(cache_root) if cache_root else {}
    api_retriever = None

    rows = []
    n_calls, n_cached = 0, 0
    for line in open(run_path):
        obj = json.loads(line)
        meta = obj.get("metadata", {})
        topic_id = str(meta.get("topic_id", ""))
        run_id = str(meta.get("run_id", ""))
        references = obj.get("references", [])
        sentences_out = []

        for sent in obj.get("answer", []):
            statement = sent.get("text", "")
            cit_out = []
            for c in sent.get("citations", []):
                docid = references[c] if isinstance(c, int) and c < len(references) else str(c)
                text = text_index.get(docid, "")
                if not text:
                    if api_retriever is None:
                        from systems.rag.retrieve import ApiRetriever
                        api_retriever = ApiRetriever()
                    text = api_retriever.get_text(docid) or ""
                if not text:
                    cit_out.append({"docid": docid, "support": -1})  # unjudgeable
                    continue

                key = cache_key(statement, text, judge_model, max_citation_chars)
                if key in judge_cache:
                    score = judge_cache[key]
                    n_cached += 1
                else:
                    prompt = render_support_prompt(
                        statement=statement, citation=text[:max_citation_chars])
                    verdict_raw = client.generate(prompt, think=False,
                                                  num_predict=16)
                    label = parse_support_label(verdict_raw)
                    score = LABEL_TO_SCORE.get(label, 0)
                    n_calls += 1
                    with open(JUDGE_CACHE, "a") as f:
                        f.write(json.dumps({"key": key, "support": score}) + "\n")
                    judge_cache[key] = score
                cit_out.append({"docid": docid, "support": score})
            sentences_out.append({"text": statement, "citations": cit_out})

        rows.append({"topic_id": topic_id, "run_id": run_id,
                     "sentences": sentences_out})
        print(f"  judged topic {topic_id} "
              f"({sum(len(s['citations']) for s in sentences_out)} citations)",
              flush=True)

    print(f"Judge calls: {n_calls} new, {n_cached} cached", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--judge-model", default="qwen3:32b")
    parser.add_argument("--cache-root", default=None,
                        help="agentic_v2 cache dir with */pool.jsonl for text lookup")
    args = parser.parse_args()

    run_path = Path(args.run)
    cache_root = Path(args.cache_root) if args.cache_root else None

    rows = judge_run(run_path, args.judge_model, cache_root)

    out_path = run_path.with_suffix(".support_judgments.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nJudgments -> {out_path}")

    # Official RAGDoll metric computation (pure Python, imported)
    print(f"\n{'topic':>12}  {'w_prec':>7}  {'hard_prec':>9}")
    metrics = []
    for row in rows:
        m = support_metric(row)
        metrics.append(m)
        print(f"{m.topic_id:>12}  {m.weighted_precision_all_judged_citations:7.3f}  "
              f"{m.hard_precision:9.3f}")
    if metrics:
        mean_w = sum(m.weighted_precision_all_judged_citations for m in metrics) / len(metrics)
        mean_h = sum(m.hard_precision for m in metrics) / len(metrics)
        print(f"{'MEAN':>12}  {mean_w:7.3f}  {mean_h:9.3f}")
        print(f"\nGate 2 pass: mean weighted precision >= 0.85 -> "
              f"{'PASS' if mean_w >= 0.85 else 'FAIL'}")


if __name__ == "__main__":
    main()
