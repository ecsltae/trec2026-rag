"""
TREC RAG 2026 Retrieval Evaluation.

Evaluates a TREC run file against projected UMBRELA qrels.
Computes NDCG@10, NDCG@100, R@100, R@1000, MAP over graded relevance.

Usage:
    python eval/eval_rag.py \\
        --run runs/rag/r_output_trec_rag_2026.tsv \\
        --qrels data/rag/qrels/rag25-climbmix-umbrela-qwen3.5-9b.qrels \\
        [--qrels data/rag/qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning.qrels] \\
        [--topics data/rag/topics/trec25_narratives_final.json] \\
        [--cutoffs 10 100 1000]

Or compare multiple runs:
    python eval/eval_rag.py \\
        --run runs/rag/run_a.tsv runs/rag/run_b.tsv \\
        --qrels data/rag/qrels/rag25-climbmix-umbrela-qwen3.5-9b.qrels
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trec_shared.trec_utils import read_trec_run, load_qrels


def dcg(gains, k):
    """Discounted Cumulative Gain at cutoff k."""
    return sum(g / math.log2(r + 2) for r, g in enumerate(gains[:k]))


def ndcg_at_k(retrieved, qrel, k, gain_fn=None):
    """NDCG@k for a single topic.

    Args:
        retrieved: list of doc_ids in rank order
        qrel: dict of doc_id -> relevance (0-3)
        k: cutoff
        gain_fn: maps relevance -> gain (default: 2^rel - 1)
    """
    if gain_fn is None:
        gain_fn = lambda r: (2 ** r - 1)

    gains = [gain_fn(qrel.get(doc, 0)) for doc in retrieved[:k]]
    actual_dcg = dcg(gains, k)

    ideal_rels = sorted(qrel.values(), reverse=True)
    ideal_gains = [gain_fn(r) for r in ideal_rels]
    ideal_dcg = dcg(ideal_gains, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved, qrel, k, min_rel=1):
    """Recall@k — fraction of relevant docs (rel >= min_rel) found in top-k."""
    relevant = {d for d, r in qrel.items() if r >= min_rel}
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def average_precision(retrieved, qrel, min_rel=1):
    """Mean Average Precision (MAP) for a single topic."""
    relevant = {d for d, r in qrel.items() if r >= min_rel}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def evaluate_run(run, qrels, cutoffs=(10, 100, 1000)):
    """Evaluate a run against qrels for all topics that have judgments.

    Args:
        run: dict topic_id -> [(doc_id, rank, score)]
        qrels: dict topic_id -> {doc_id -> relevance}
        cutoffs: sequence of k values for NDCG/R@k

    Returns:
        dict: metric_name -> mean_value
        dict: topic_id -> {metric_name -> value}
    """
    judged_topics = set(qrels.keys()) & set(run.keys())
    per_topic = {}

    for tid in sorted(judged_topics):
        retrieved = [doc for doc, _, _ in run[tid]]
        qrel = qrels[tid]
        metrics = {}
        for k in cutoffs:
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, qrel, k)
            metrics[f"r@{k}"] = recall_at_k(retrieved, qrel, k)
        metrics["map"] = average_precision(retrieved, qrel)
        metrics["n_retrieved"] = len(retrieved)
        metrics["n_relevant"] = sum(1 for r in qrel.values() if r >= 1)
        per_topic[tid] = metrics

    if not per_topic:
        return {}, {}

    # Macro-average over judged topics
    all_metric_names = list(next(iter(per_topic.values())).keys())
    macro = {m: sum(v[m] for v in per_topic.values()) / len(per_topic)
             for m in all_metric_names if m not in ("n_retrieved", "n_relevant")}
    macro["judged_topics"] = len(per_topic)
    return macro, per_topic


def print_results(run_name, macro, per_topic, cutoffs, verbose=False):
    print(f"\nRun: {run_name}  ({macro.get('judged_topics', 0)} judged topics)")
    print(f"  {'Metric':<15} {'Value':>8}")
    print(f"  {'-'*24}")
    for k in cutoffs:
        print(f"  {'NDCG@'+str(k):<15} {macro.get(f'ndcg@{k}', 0):>8.4f}")
    for k in cutoffs:
        print(f"  {'R@'+str(k):<15} {macro.get(f'r@{k}', 0):>8.4f}")
    print(f"  {'MAP':<15} {macro.get('map', 0):>8.4f}")

    if verbose and per_topic:
        print(f"\n  Per-topic NDCG@10:")
        for tid, m in sorted(per_topic.items()):
            print(f"    {tid}: ndcg@10={m.get('ndcg@10',0):.4f}  r@100={m.get('r@100',0):.4f}  "
                  f"map={m.get('map',0):.4f}  n_rel={m.get('n_relevant',0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", required=True,
                        help="One or more TREC run files (TSV)")
    parser.add_argument("--qrels", nargs="+", required=True,
                        help="One or more qrels files")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--min-rel", type=int, default=1,
                        help="Minimum relevance for Recall/MAP (default 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-topic breakdown")
    parser.add_argument("--out", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_results = {}

    for qrels_path in args.qrels:
        qrels = load_qrels(qrels_path)
        qrels_name = Path(qrels_path).stem
        print(f"\n{'='*60}")
        print(f"Qrels: {qrels_name}  ({len(qrels)} topics judged)")
        print(f"{'='*60}")

        for run_path in args.run:
            run = read_trec_run(run_path)
            run_name = Path(run_path).stem
            macro, per_topic = evaluate_run(run, qrels, args.cutoffs)
            if not macro:
                print(f"\nRun: {run_name}  — NO OVERLAP with qrels (0 judged topics)")
                continue
            print_results(run_name, macro, per_topic, args.cutoffs, args.verbose)
            all_results.setdefault(qrels_name, {})[run_name] = {
                "macro": macro, "per_topic": per_topic
            }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
