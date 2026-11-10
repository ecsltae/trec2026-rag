"""
Build TREC RAG 2026 spec-compliant submission files from the agentic_v2 cache.

Produces the officially-named files with full required metadata:
  - r_output_trec_rag_2026.tsv     (Retrieval task; 6-col TSV)
  - rag_output_trec_rag_2026.jsonl (RAG task; metadata + references + answer)

Metadata per rag-task.md: team_id, run_id, type, narrative_id, title, narrative.
narrative_id/title/narrative are taken from the official topics file. The released
test file is a TSV with id<TAB>query (no separate title column), so:
  narrative = full query text
  title     = first ~12 words (keyword-style short title) — the spec's title is a
              short keyword form; no gold title was released for the test set.

Usage:
    python systems/rag/agentic_v2/build_submission.py \
        --team-id <team> --run-id bp_agentic_v2 \
        --topic-set official119 --out-dir submissions/bp_agentic_v2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from trec_shared.trec_utils import validate_trec_run

CACHE_ROOT = ROOT / "data/rag/cache/agentic_v2"
TOPICS_TSV = ROOT / "data/rag/topics/trec_rag_2026_queries.tsv"


def load_official_topics() -> dict:
    """id -> {title, narrative} from the released TSV (id<TAB>query)."""
    topics = {}
    with open(TOPICS_TSV) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            tid, query = row[0], row[1]
            title = " ".join(query.split()[:12])
            topics[tid] = {"title": title, "narrative": query}
    return topics


def build_tsv(topic_set: str, run_id: str, out_path: Path) -> int:
    rows = []
    for tid_dir in sorted((CACHE_ROOT / topic_set).iterdir()):
        fr = tid_dir / "final_ranking.json"
        if not fr.exists():
            continue
        ranking = json.loads(fr.read_text())
        for rank, (docid, score) in enumerate(ranking, 1):
            rows.append(f"{tid_dir.name} Q0 {docid} {rank} {score:.6f} {run_id}")
    out_path.write_text("\n".join(rows) + "\n")
    return len(rows)


def build_jsonl(topic_set: str, team_id: str, run_id: str, run_type: str,
                topics: dict, out_path: Path) -> tuple:
    n, empty, orphan_refs = 0, 0, 0
    with open(out_path, "w") as f:
        for tid_dir in sorted((CACHE_ROOT / topic_set).iterdir()):
            ap = tid_dir / "answer.json"
            if not ap.exists():
                continue
            a = json.loads(ap.read_text())
            tid = tid_dir.name

            # references = deduped cited docids (in first-appearance order)
            refs, seen = [], set()
            for s in a["answer"]:
                for c in s["citations"]:
                    if c not in seen:
                        seen.add(c)
                        refs.append(c)
            sentences = [{"text": s["text"],
                          "citations": [refs.index(c) for c in s["citations"]]}
                         for s in a["answer"]]

            # Validation: every reference cited by >=1 sentence (true by construction)
            cited_positions = {c for s in sentences for c in s["citations"]}
            if len(cited_positions) != len(refs):
                orphan_refs += 1
            if not sentences:
                empty += 1

            topic_meta = topics.get(tid, {"title": "", "narrative": ""})
            obj = {
                "metadata": {
                    "team_id": team_id,
                    "run_id": run_id,
                    "type": run_type,
                    "narrative_id": tid,
                    "title": topic_meta["title"],
                    "narrative": topic_meta["narrative"],
                },
                "references": refs,
                "answer": sentences,
            }
            f.write(json.dumps(obj) + "\n")
            n += 1
    return n, empty, orphan_refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--type", default="automatic", choices=["automatic", "manual"])
    parser.add_argument("--topic-set", default="official119")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = load_official_topics()

    tsv_path = out_dir / "r_output_trec_rag_2026.tsv"
    n_rows = build_tsv(args.topic_set, args.run_id, tsv_path)
    errors = validate_trec_run(str(tsv_path), max_per_topic=100000)
    print(f"R  -> {tsv_path.name}: {n_rows} rows, "
          f"{'OK' if not errors else errors[:3]}")

    jsonl_path = out_dir / "rag_output_trec_rag_2026.jsonl"
    n, empty, orphan = build_jsonl(args.topic_set, args.team_id, args.run_id,
                                   args.type, topics, jsonl_path)
    print(f"RAG-> {jsonl_path.name}: {n} topics, empty={empty}, orphan_refs={orphan}")

    # Cross-check topic coverage
    tsv_topics = {l.split()[0] for l in open(tsv_path)}
    jsonl_topics = {json.loads(l)["metadata"]["narrative_id"]
                    for l in open(jsonl_path)}
    print(f"Coverage: R={len(tsv_topics)} topics, RAG={len(jsonl_topics)} topics, "
          f"official=119")
    missing_r = set(topics) - tsv_topics
    missing_rag = set(topics) - jsonl_topics
    if missing_r:
        print(f"  WARNING R missing: {sorted(missing_r)}")
    if missing_rag:
        print(f"  WARNING RAG missing: {sorted(missing_rag)}")
    if not missing_r and not missing_rag and not empty and not orphan:
        print("  ✓ Both files complete and spec-compliant")


if __name__ == "__main__":
    main()
