"""
Local blind pairwise arena: two RAG run JSONLs, official RAGDoll arena prompt,
local Ollama judge. Position bias controlled by judging each topic twice with
A/B swapped; a win only counts if consistent across both orders (else tie).

Usage:
    python eval/pairwise_arena_local.py \
        --run-a runs/rag/bp_agentic_v2_dev22.jsonl \
        --run-b runs/rag/agentic_iterative_sq_v1.jsonl \
        --topics data/rag/topics/trec25_narratives_final.json \
        [--judge-model qwen3.5:122b] [--sample 10] [--seed 42]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from ragdoll.arena.prompts import ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER

from trec_shared.ollama_client import OllamaClient
from trec_shared.trec_utils import load_topics

VERDICT_RE = re.compile(r"\[\[(A|B|Tie)\]\]", re.IGNORECASE)


def load_answers(path: Path) -> dict:
    """topic_id -> plain answer text (sentences joined, citations stripped)."""
    answers = {}
    for line in open(path):
        obj = json.loads(line)
        tid = str(obj.get("metadata", {}).get("topic_id", ""))
        sentences = [s.get("text", "") for s in obj.get("answer", [])]
        answers[tid] = " ".join(sentences).strip()
    return answers


def judge_pair(client, query: str, answer_a: str, answer_b: str) -> str:
    """Returns 'A', 'B', or 'Tie'."""
    prompt = ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER.format(
        query=query, answer_a=answer_a or "(empty answer)",
        answer_b=answer_b or "(empty answer)")
    raw = client.generate(prompt, think=False, num_predict=2048)
    matches = VERDICT_RE.findall(raw)
    if matches:
        v = matches[-1].capitalize()
        return "Tie" if v == "Tie" else v
    return "Tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--judge-model", default="qwen3.5:122b")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    name_a = Path(args.run_a).stem
    name_b = Path(args.run_b).stem
    answers_a = load_answers(Path(args.run_a))
    answers_b = load_answers(Path(args.run_b))
    topics = load_topics(args.topics)

    common = sorted(set(answers_a) & set(answers_b) & set(topics))
    common = [t for t in common if answers_a[t] and answers_b[t]]
    if not common:
        print("No common topics with non-empty answers.")
        return
    random.seed(args.seed)
    sample = random.sample(common, min(args.sample, len(common)))

    client = OllamaClient(model=args.judge_model, temperature=0.0)
    wins = {"A": 0, "B": 0, "Tie": 0}
    print(f"A = {name_a}\nB = {name_b}\n")
    for tid in sample:
        query = topics[tid]["narrative"]
        v1 = judge_pair(client, query, answers_a[tid], answers_b[tid])
        # Swap order to control position bias
        v2 = judge_pair(client, query, answers_b[tid], answers_a[tid])
        v2 = {"A": "B", "B": "A", "Tie": "Tie"}[v2]
        verdict = v1 if v1 == v2 else "Tie"
        wins[verdict] += 1
        print(f"  topic {tid:>12}: {verdict:4} (orders: {v1}/{v2})", flush=True)

    n = len(sample)
    print(f"\n{name_a}: {wins['A']}/{n} wins")
    print(f"{name_b}: {wins['B']}/{n} wins")
    print(f"ties: {wins['Tie']}/{n}")


if __name__ == "__main__":
    main()
