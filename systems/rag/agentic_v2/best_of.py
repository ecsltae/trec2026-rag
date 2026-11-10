"""
Best-of arena selection: compose a per-topic-winner run from candidate runs.

For each topic, all candidate answers battle round-robin using RAGDoll's
official arena prompt (both A/B orders; win only if consistent, else tie ->
0.5 points each). The per-topic winner's answer goes into the composite run.

Methodological note for the paper: the selection judge and any later arena
evaluation share the same model, so "composite beats its ingredients in our
arena" is true partly by construction. Independent checks: citation-support
precision (different task), and the official human/organizer battles.

Usage:
    python systems/rag/agentic_v2/best_of.py \
        --runs runs/rag/bp_agentic_v2_dev22.jsonl \
               runs/rag/bp_agentic_v2_32b_dev22.jsonl \
               runs/rag/bp_agentic_v3_dev22.jsonl \
        --names v2_122b v2_32b v3_loop \
        --topics data/rag/topics/trec25_narratives_final.json \
        --out runs/rag/best_of_arena_dev22.jsonl
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from trec_shared.ollama_client import OllamaClient
from trec_shared.trec_utils import load_topics
from eval.pairwise_arena_local import load_answers, judge_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", default="qwen3.5:122b")
    args = parser.parse_args()
    assert len(args.runs) == len(args.names)

    topics = load_topics(args.topics)
    # name -> {topic_id -> answer text} and full objects for composition
    texts, objects = {}, {}
    for path, name in zip(args.runs, args.names):
        texts[name] = load_answers(Path(path))
        objects[name] = {}
        for line in open(path):
            obj = json.loads(line)
            objects[name][str(obj["metadata"]["topic_id"])] = obj

    client = OllamaClient(model=args.judge_model, temperature=0.0)
    common = sorted(set.intersection(*[set(t) for t in texts.values()])
                    & set(topics))
    print(f"Composing best-of over {len(common)} topics, "
          f"{len(args.names)} candidates: {args.names}", flush=True)

    composition = {n: 0 for n in args.names}
    results = []
    for tid in common:
        query = topics[tid]["narrative"]
        candidates = [n for n in args.names if texts[n].get(tid)]
        points = {n: 0.0 for n in candidates}
        for a, b in itertools.combinations(candidates, 2):
            v1 = judge_pair(client, query, texts[a][tid], texts[b][tid])
            v2 = judge_pair(client, query, texts[b][tid], texts[a][tid])
            v2 = {"A": "B", "B": "A", "Tie": "Tie"}[v2]
            verdict = v1 if v1 == v2 else "Tie"
            if verdict == "A":
                points[a] += 1
            elif verdict == "B":
                points[b] += 1
            else:
                points[a] += 0.5
                points[b] += 0.5
        # winner: max points; ties broken by candidate priority (list order)
        winner = max(candidates, key=lambda n: (points[n], -args.names.index(n)))
        composition[winner] += 1
        chosen = dict(objects[winner][tid])
        chosen.setdefault("metadata", {})["selected_from"] = winner
        chosen["metadata"]["arena_points"] = points
        results.append(chosen)
        print(f"  {tid}: {winner}  {points}", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nComposition: {composition}")
    print(f"Wrote {len(results)} topics -> {args.out}")


if __name__ == "__main__":
    main()
