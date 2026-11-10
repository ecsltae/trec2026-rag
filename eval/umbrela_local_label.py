#!/usr/bin/env python3
"""Build our own relevance qrels for the OFFICIAL 119 topics, using UMBRELA prompts locally.

WHY THIS MATTERS MORE THAN ANYTHING ELSE ON THE BOARD
-----------------------------------------------------
The official 119 test topics are UNJUDGED — the organisers' own baseline READMEs say so. We
therefore have NO ground truth on the submission set, and every decision so far has rested on
proxies: a support judge that awarded FULL SUPPORT to a sentence whose entire text is ".", and an
arena judge measured to be worse than random at per-topic selection. Concretely unanswerable today:
  * is our retrieval any good on the official topics?
  * did the variable-depth truncation (1000 -> median ~100-127 docs) discard relevant documents?
  * is ce_floor the right depth, or ce_floor_robust, or cited-only?
dev22 cannot substitute: its projected qrels mark ~82-88% of judged documents relevant
(~1048 of 1197 per topic), which makes set-F1 a monotone function of depth and so cannot arbitrate
depth at all, and it shares no topics with the official baseline.

WHAT THIS DOES
--------------
Runs the OFFICIAL UMBRELA prompts (eval/RAGDoll/src/ragdoll/umbrela/prompts.py — the same labeler
family that produced data/rag/qrels/rag25-climbmix-umbrela-*.qrels) against a local Ollama model,
exactly as eval/ragdoll_local_judge.py wraps the official support prompts. Emits TREC qrels:
    topic_id 0 docid grade      grade in 0..3

POOLING — the design point that makes the result meaningful
-----------------------------------------------------------
Labelling only the documents OUR runs retrieved would be circular: we cannot miss what we never
labelled, so recall would be inflated to meaninglessness. The pool therefore follows TREC practice
and is the UNION of multiple systems:
    our 3 submitted runs  +  the organisers' FIRST-qwen3-8b baseline  +  a stratified sample of the
    tail our variable-depth cut DISCARDED (from backups/r_output_preVariableDepth_*/)
Measured: 32-37% of the pool comes from the baseline alone, i.e. documents we never retrieved, which
is what makes relative recall measurable. The discarded-tail sample is what lets us estimate how
much relevance the truncation actually cost, rather than guessing.

HONEST LIMITS, state these wherever the numbers are used
--------------------------------------------------------
* These are OUR labels from a local model, not NIST's. That is also exactly what the dev22 qrels
  are (UMBRELA-projected), and the organisers themselves ship three different judge variants
  (gpt5.5, ministral-3-14b, qwen3.5-9b), so the method is theirs; the model is ours.
* qwen3.5-9b — the variant our primary dev22 qrels use — is NOT available locally, so numbers are
  not directly comparable to dev22. They are internally consistent for comparing our runs against
  each other and against the baseline, which is the actual question.
* Relative recall over a pool is not true recall. Any system outside the pool is unrepresented.

Resumable: every judgment is cached by (topic_id, docid, model, prompt_type), so an interrupted run
resumes for free. Idempotent.

Usage:
    python eval/umbrela_local_label.py --plan                 # pool size + cost, judge nothing
    python eval/umbrela_local_label.py --depth 50             # label a depth-50 pool
    python eval/umbrela_local_label.py --depth 0 --tail 20    # full shipped depth + 20 tail/topic
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from ragdoll.umbrela.prompts import render_umbrela_prompt, parse_umbrela_judgment  # noqa: E402
from trec_shared.ollama_client import OllamaClient                                       # noqa: E402
from trec_shared.trec_utils import load_topics                                           # noqa: E402

MERGED = ROOT / "data/rag/cache/agentic_v2/merged_official119"
CACHE = ROOT / "data/rag/cache/umbrela_labels.jsonl"
OUT_DIR = ROOT / "data/rag/qrels"
OURS = ["bp_agentic_v2", "bp_probe_guided", "bp_support_argmax"]
BASELINE = ROOT / "eval/analysis/baselines/first_qwen3_8b_top100.trec"


def read_run(path: Path, depth: int = 0) -> dict:
    """topic -> [docid] in rank order, optionally truncated to `depth`."""
    per = defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) >= 6:
            per[f[0]].append((int(f[3]), f[2]))
    return {t: [d for _, d in sorted(v)][:depth] if depth else [d for _, d in sorted(v)]
            for t, v in per.items()}


def build_pool(depth: int, tail: int, seed: int = 0) -> dict:
    pool, prov = defaultdict(set), defaultdict(lambda: defaultdict(set))
    for s in OURS:
        r = read_run(ROOT / "submissions" / s / "r_output_trec_rag_2026.tsv", depth)
        for t, docs in r.items():
            pool[t] |= set(docs)
            prov[t]["ours"] |= set(docs)
    if BASELINE.exists():
        for t, docs in read_run(BASELINE, depth or 100).items():
            pool[t] |= set(docs)
            prov[t]["baseline"] |= set(docs)
    # stratified sample of what the variable-depth cut threw away
    if tail:
        rng = random.Random(seed)
        bk = sorted(ROOT.glob("backups/r_output_preVariableDepth_*"))
        if bk:
            for s in OURS:
                p = bk[-1] / f"{s}__r_output.tsv"
                if not p.exists():
                    continue
                for t, docs in read_run(p).items():
                    discarded = [d for d in docs if d not in pool[t]]
                    if discarded:
                        pick = rng.sample(discarded, min(tail, len(discarded)))
                        pool[t] |= set(pick)
                        prov[t]["discarded_tail"] |= set(pick)
    return pool, prov


def load_cache() -> dict:
    c = {}
    if CACHE.exists():
        for line in open(CACHE):
            try:
                o = json.loads(line)
                c[o["key"]] = o["grade"]
            except (json.JSONDecodeError, KeyError):
                continue
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="report pool + cost, judge nothing")
    ap.add_argument("--depth", type=int, default=0,
                    help="per-run depth to pool (0 = full shipped depth)")
    ap.add_argument("--tail", type=int, default=20,
                    help="stratified docs/topic sampled from the DISCARDED tail")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--prompt-type", default="bing", choices=["bing", "basic"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N new judgments")
    a = ap.parse_args()

    topics = load_topics(str(ROOT / "data/rag/topics/trec_rag_2026_queries.tsv"))
    pool, prov = build_pool(a.depth, a.tail)
    total = sum(len(v) for v in pool.values())
    n_ours = sum(len(prov[t]["ours"]) for t in prov)
    n_base_only = sum(len(prov[t]["baseline"] - prov[t]["ours"]) for t in prov)
    n_tail = sum(len(prov[t]["discarded_tail"]) for t in prov)
    cache = load_cache()
    todo = [(t, d) for t, ds in pool.items() for d in ds
            if f"{t}|{d}|{a.model}|{a.prompt_type}" not in cache]

    print(f"POOL: {total} (topic, doc) pairs over {len(pool)} topics ({total/max(1,len(pool)):.0f}/topic)")
    print(f"  from our 3 runs         : {n_ours}")
    print(f"  baseline-only (unseen)  : {n_base_only} ({100*n_base_only/max(1,total):.0f}%) "
          f"<- makes relative recall measurable")
    print(f"  discarded-tail sample   : {n_tail} <- estimates what truncation cost")
    print(f"  already cached          : {total-len(todo)}")
    print(f"  TO JUDGE                : {len(todo)}  -> est {len(todo)*0.93/3600:.1f}h "
          f"at 0.93 s/label")
    if a.plan:
        print("\n--plan: nothing judged.")
        return

    from eval.ragdoll_local_judge import build_text_index, resolve_text
    print("  building passage index ...", flush=True)
    index = build_text_index(MERGED)
    print(f"  {len(index):,} docids resolvable locally")

    client = OllamaClient(model=a.model, temperature=0.0)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    t0, done, unresolved, unparsed = time.time(), 0, 0, 0
    if a.limit:
        todo = todo[: a.limit]
    with open(CACHE, "a") as cf:
        for i, (tid, docid) in enumerate(todo, 1):
            text = index.get(docid, "")
            if not text.strip():
                unresolved += 1
                continue
            query = topics[tid]["narrative"] if tid in topics else ""
            if not query:
                continue
            prompt = render_umbrela_prompt(query=query, passage=text[:4000],
                                           prompt_type=a.prompt_type)
            raw = client.generate(prompt, think=False, num_predict=64)
            grade = parse_umbrela_judgment(raw)
            if grade is None:
                unparsed += 1
                grade = 0
            key = f"{tid}|{docid}|{a.model}|{a.prompt_type}"
            cf.write(json.dumps({"key": key, "topic_id": tid, "docid": docid,
                                 "grade": grade}) + "\n")
            cf.flush()
            done += 1
            if i % 200 == 0:
                el = time.time() - t0
                print(f"    [{i}/{len(todo)}] {el/i:.2f}s/label  "
                      f"eta {(len(todo)-i)*el/i/3600:.1f}h  unresolved={unresolved} "
                      f"unparsed={unparsed}", flush=True)

    # ---------------------------------------------------------------- emit qrels
    cache = load_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = a.model.replace(":", "-")
    out = OUT_DIR / f"official119-umbrela-{tag}-{a.prompt_type}.qrels"
    grades = defaultdict(int)
    with open(out, "w") as f:
        for key, g in sorted(cache.items()):
            tid, docid, model, ptype = key.split("|")
            if model != a.model or ptype != a.prompt_type:
                continue
            f.write(f"{tid} 0 {docid} {g}\n")
            grades[g] += 1
    print(f"\n  judged this run: {done}  unresolved: {unresolved}  unparsed->0: {unparsed}")
    print(f"  grade distribution: {dict(sorted(grades.items()))}")
    rel = sum(v for k, v in grades.items() if k >= 1)
    tot = sum(grades.values())
    print(f"  relevant (>=1): {rel}/{tot} = {100*rel/max(1,tot):.1f}%  "
          f"(dev22's projected qrels are 82-88% relevant, which is why they cannot arbitrate depth)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
