#!/bin/bash
# Checks requested by the review of 2026-11-10 (eval/analysis/review_checks_2026_11_10.py), all on
# qwen3:32b once the official119 writer job has finished. Resumable: every judge caches its calls.
#   1  support judge against the passage each writer was shown       -> runs/rag/<run>.support_shown.jsonl
#   2  original support judge on the coverage-prompt variant          -> runs/rag/bp_v2_coverage_dev22.support_judgments.jsonl
#   3  nugget judge on the grounded copies (both judges)              -> eval/analysis/nugget_dev22_review.json
#   4  relevance labels for Table 1's top tens on dev22
#   5  relevance labels for the depth-matched counterfactual (82 judged narratives)
#   6  report                                                          -> eval/analysis/review_checks_2026_11_10.json
#   setsid nohup bash scripts/run_review_checks.sh > logs/rag/review_checks_$(date +%Y%m%d_%H%M).log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source "${VENV:-.venv}/bin/activate"
echo "driver PID $$"
WRITER=${1:-}
if [ -n "$WRITER" ]; then
    echo "== $(date '+%F %T') waiting for the writer job (PID $WRITER)"
    while kill -0 "$WRITER" 2>/dev/null; do sleep 30; done
fi
wait_gpu_free() {
    ollama stop qwen3.5:122b-gpu36 > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b > /dev/null 2>&1 || true
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do sleep 5; done
}
wait_gpu_free
echo "== $(date '+%F %T') 1 support judge against the shown passage"
python eval/analysis/review_checks_2026_11_10.py support-shown
echo "== $(date '+%F %T') 2 original support judge on the coverage-prompt variant"
python eval/ragdoll_local_judge.py --run runs/rag/bp_v2_coverage_dev22.jsonl --judge-model qwen3:32b \
    --cache-root data/rag/cache/agentic_v2/dev22 | grep -v "judged topic" | tail -3
python eval/analysis/grounded_coverage_dev22.py build
echo "== $(date '+%F %T') 3 nugget judge on the grounded copies"
python eval/nugget_local_judge.py --out eval/analysis/nugget_dev22_review.json --run \
    runs/rag/bp_v2_coverage_dev22_grounded.jsonl runs/rag/bp_v2_coverage_dev22_supported.jsonl \
    runs/rag/bp_agentic_v2_dev22_shown_grounded.jsonl runs/rag/bp_v2_coverage_dev22_shown_grounded.jsonl \
    runs/rag/bp_v2_evidence_dev22_shown_grounded.jsonl runs/rag/bp_v2_multipass_dev22_shown_grounded.jsonl
python eval/analysis/grounded_coverage_dev22.py report || true
echo "== $(date '+%F %T') 4 relevance labels, Table 1 top tens"
python eval/analysis/review_checks_2026_11_10.py dev22-label
echo "== $(date '+%F %T') 5 relevance labels, depth-matched counterfactual"
python eval/analysis/review_checks_2026_11_10.py depth-label
ollama stop qwen3:32b > /dev/null 2>&1 || true
echo "== $(date '+%F %T') 6 report"
python eval/analysis/review_checks_2026_11_10.py report
echo "== $(date '+%F %T') done"
