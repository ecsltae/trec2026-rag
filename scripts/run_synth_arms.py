"""
Re-run Stage 5 (synthesize) from the cached retrieval, once per pre-registered arm.

Nothing upstream is recomputed: blueprint, pool and facet rankings are read from
data/rag/cache/agentic_v2/<topic_set>/<topic>/, so every arm writes from the same evidence and
only the writing configuration differs. See eval/analysis/PREREG_post_results_rerun.md.

Arms (systems/rag/agentic_v2/synthesize.py ARMS):
    rerun     the shipped configuration, run again: the null control for generator noise
    budget    shipped prompt, 950-word cap instead of 450, twice the sentence target
    coverage  budget + a prompt that asks for coverage instead of "silently omit"
    evidence  coverage + the best-matching 1,000 characters of each document instead of its head
    multipass evidence + a weight per sub-topic + up to three review-and-repair passes (addendum A)

Each arm writes answer_<arm>.json next to the shipped answer.json (never over it) and
runs/rag/bp_v2_<arm>_<topic_set>.jsonl, assembled exactly as pipeline.phase_assemble does.
Resumable: topics that already have answer_<arm>.json are skipped.

Usage:
    python scripts/run_synth_arms.py --topic-set dev22 --arms rerun budget coverage evidence
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import httpx
from ollama import ResponseError

from trec_shared.ollama_client import OllamaClient
from systems.rag.agentic_v2.pipeline import load_topic_set, topic_dir, RUNS_DIR
from systems.rag.agentic_v2.synthesize import synthesize_topic, ARMS

ARM_CONFIG = {"rerun": "shipped", "budget": "budget", "coverage": "coverage", "evidence": "evidence",
              "multipass": "multipass"}  # multipass: addendum A of the pre-registration (2026-11-09)


def assemble(topic_set: str, tids: list, out_name: str, run_id: str, suffix: str = "") -> Path:
    """Same JSONL as pipeline.phase_assemble: references derived from the citations."""
    out_jsonl = RUNS_DIR / f"{run_id}_{topic_set}{suffix}.jsonl"
    with open(out_jsonl, "w") as f:
        for tid in tids:
            a = json.loads((topic_dir(topic_set, tid) / out_name).read_text())
            refs = []
            for s in a.get("answer", []):
                for c in s.get("citations", []):
                    if c not in refs:
                        refs.append(c)
            sentences = [{"text": s["text"],
                          "citations": [refs.index(c) for c in s.get("citations", [])]}
                         for s in a.get("answer", [])]
            f.write(json.dumps({
                "metadata": {"topic_id": a["topic_id"], "run_id": run_id, "type": "automatic"},
                "references": refs,
                "answer": sentences,
            }) + "\n")
    return out_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-set", default="dev22")
    parser.add_argument("--arms", nargs="+", default=list(ARM_CONFIG), choices=list(ARM_CONFIG))
    parser.add_argument("--synth-model", default="qwen3.5:122b")
    parser.add_argument("--expect-topics", type=int, default=None,
                        help="abort unless the topic set has exactly this many topics")
    parser.add_argument("--unload-after", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, metavar="FILE",
                        help="restrict to the topic ids listed in these files (one per line)")
    parser.add_argument("--run-suffix", default="",
                        help="appended to the run file name, e.g. _judged42 for a subset")
    args = parser.parse_args()

    topics = load_topic_set(args.topic_set)
    if args.expect_topics is not None and len(topics) != args.expect_topics:
        sys.exit(f"{args.topic_set}: {len(topics)} topics, expected {args.expect_topics}")
    tids = sorted(topics)
    if args.only:
        wanted = {l.strip() for f in args.only for l in open(f) if l.strip()}
        missing = wanted - set(tids)
        if missing:
            sys.exit(f"--only lists topics not in {args.topic_set}: {sorted(missing)[:5]}")
        tids = [t for t in tids if t in wanted]
        print(f"--only: {len(tids)} topics", flush=True)
    client = OllamaClient(model=args.synth_model, temperature=0.15, timeout=2400)

    for arm in args.arms:
        assert ARM_CONFIG[arm] in ARMS
        out_name = f"answer_{arm}.json"
        todo = [t for t in tids if not (topic_dir(args.topic_set, t) / out_name).exists()]
        print(f"[{arm}] {len(todo)}/{len(tids)} topics to do", flush=True)
        for i, tid in enumerate(todo, 1):
            t0 = time.time()
            # The 122b sits at the edge of the A100; a CUDA OOM kills the runner mid-generation
            # (2026-11-08, topic 213), and CPU contention from other jobs can stall a request past the
            # timeout (topic 273). Ollama reloads or frees the runner on the next request, so retry.
            for attempt in range(1, 4):
                try:
                    result = synthesize_topic(client, topic_dir(args.topic_set, tid), tid,
                                              topics[tid].get("narrative", ""),
                                              out_name=out_name, arm=ARM_CONFIG[arm])
                    break
                except (ResponseError, httpx.TimeoutException) as e:
                    if attempt == 3:
                        raise
                    print(f"[{arm}] {tid}: Ollama error ({e}), retry {attempt}/2 in 60s", flush=True)
                    time.sleep(60)
            words = sum(len(s["text"].split()) for s in result["answer"])
            print(f"[{arm}] [{i}/{len(todo)}] {tid}: {len(result['answer'])} sentences, "
                  f"{words} words ({time.time()-t0:.0f}s)", flush=True)
        out = assemble(args.topic_set, tids, out_name, f"bp_v2_{arm}", args.run_suffix)
        print(f"[{arm}] -> {out}", flush=True)

    if args.unload_after:
        client.unload()


if __name__ == "__main__":
    main()
