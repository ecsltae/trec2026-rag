#!/bin/bash
# Addendum A of eval/analysis/PREREG_post_results_rerun.md: the lab's multi-pass writer (2026-11-09).
#
# STAGES
#   1  write dev22 with the multipass arm, writer qwen3.5:122b-gpu36, from the cached retrieval
#                                                   -> runs/rag/bp_v2_multipass_dev22.jsonl
#   2  score it on the official dev nuggets, judge qwen3:32b
#                                                   -> eval/analysis/nugget_dev22_multipass.json
#   3  citation support of the shipped, evidence and multipass answers, judge qwen3:32b
#                                                   -> runs/rag/<run>.support_judgments.jsonl
#   4  paired comparison (stdout and the log)
#
# Idempotent like run_post_results_rerun.sh: the writer skips topics that have answer_multipass.json,
# and both judges cache every call, so a re-run resumes where it stopped.
#
# USAGE
#   setsid nohup bash scripts/run_multipass_dev22.sh > logs/rag/multipass_dev22_$(date +%Y%m%d_%H%M).log 2>&1 &
#   stop: kill the PIDs the log prints at the top
set -euo pipefail
cd "$(dirname "$0")/.."
source "${VENV:-.venv}/bin/activate"
echo "driver PID $$"

wait_gpu_free() {
    ollama stop qwen3:32b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b-gpu36 > /dev/null 2>&1 || true
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do
        sleep 5
    done
}

echo "== $(date '+%F %T') stage 1: write dev22 with the multipass arm"
wait_gpu_free
python scripts/run_synth_arms.py --topic-set dev22 --arms multipass --expect-topics 22 \
    --synth-model qwen3.5:122b-gpu36 --unload-after

echo "== $(date '+%F %T') stage 2: score it on the dev nuggets"
wait_gpu_free
f=runs/rag/bp_v2_multipass_dev22.jsonl
[ "$(wc -l < "$f")" -eq 22 ] || { echo "ABORT: $f does not have 22 answers"; exit 1; }
python eval/nugget_local_judge.py --run "$f" --out eval/analysis/nugget_dev22_multipass.json --unload-after

echo "== $(date '+%F %T') stage 3: citation support (shipped, evidence, multipass)"
for r in bp_agentic_v2_dev22 bp_v2_evidence_dev22 bp_v2_multipass_dev22; do
    echo "-- $r"
    python eval/ragdoll_local_judge.py --run "runs/rag/$r.jsonl" --judge-model qwen3:32b \
        --cache-root data/rag/cache/agentic_v2/dev22 | grep -v "judged topic"
done
ollama stop qwen3:32b > /dev/null 2>&1 || true

echo "== $(date '+%F %T') stage 4: paired comparison"
python eval/analysis/post_results_rerun_dev22.py eval/analysis/nugget_dev22_existing.json \
    eval/analysis/nugget_dev22_arms.json eval/analysis/nugget_dev22_multipass.json
python - <<'PY'
# the pre-registered primary cell of addendum A: multipass against evidence, not against shipped
import json, statistics as st
runs = {}
for f in ("eval/analysis/nugget_dev22_arms.json", "eval/analysis/nugget_dev22_multipass.json"):
    runs.update(json.load(open(f))["runs"])
a, b = runs["bp_v2_evidence_dev22"]["per_topic"], runs["bp_v2_multipass_dev22"]["per_topic"]
for m in ("strict_all_score", "strict_vital_score"):
    d = [b[t][m] - a[t][m] for t in sorted(set(a) & set(b))]
    se = st.stdev(d) / len(d) ** 0.5
    print(f"PRIMARY multipass - evidence, {m}: {st.mean(d):+.4f}  SE {se:.4f}  {st.mean(d) / se:+.1f} SE"
          f"  ({sum(x > 0 for x in d)}-{sum(x < 0 for x in d)}, n {len(d)})")
PY
echo "== $(date '+%F %T') done"
