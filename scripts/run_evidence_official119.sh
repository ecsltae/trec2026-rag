#!/bin/bash
# The step the pre-registration licenses once its primary passed (PREREG_post_results_rerun.md,
# "What a result licenses"): rewrite the official 119 answers with the `evidence` arm, writer stage
# only, from the existing official119 cache, and validate them. A post-hoc run, never filed; it can be
# scored only if the organisers release the official nuggets or assignments.
#
# Order (operational log of 2026-11-09 20:42):
#   1  the 42 narratives the organisers judged for generation (batch 1 + batch 2), the only ones that
#      can ever be scored          -> runs/rag/bp_v2_evidence_official119_judged42.jsonl
#   2  addendum B on dev22 (coverage counting only grounded sentences), judge qwen3:32b
#                                  -> eval/analysis/nugget_dev22_grounded.json
#   3  the other 77                -> runs/rag/bp_v2_evidence_official119.jsonl (all 119)
#
#   setsid nohup bash scripts/run_evidence_official119.sh [START_STAGE] > logs/rag/evidence_official119_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# Resumable: topics that already have answer_evidence.json are skipped. START_STAGE (default 1) skips
# the earlier stages. 2026-11-10: the first launch stopped after stage 1, because the validator
# fails a 42-topic file on "coverage 119/119" (expected for a subset) and set -e made that fatal.
# The subset is now validated without stopping the job.
set -euo pipefail
cd "$(dirname "$0")/.."
source "${VENV:-.venv}/bin/activate"
echo "driver PID $$"
START=${1:-1}
JUDGED=data/rag/official_scores/generation_2026-11-03/judged42-topics.txt

wait_gpu_free() {
    ollama stop qwen3:32b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b-gpu36 > /dev/null 2>&1 || true
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do
        sleep 5
    done
}

if [ "$START" -le 1 ]; then
echo "== $(date '+%F %T') stage 1: the 42 judged narratives"
# no wait_gpu_free here: the writer may still be loaded from an interrupted run, and reloading it
# from disk costs minutes
python scripts/run_synth_arms.py --topic-set official119 --arms evidence --only "$JUDGED" \
    --run-suffix _judged42 --synth-model qwen3.5:122b-gpu36 --unload-after
# a subset fails the 119-topic coverage check by design; every other check must pass
python eval/validate_submission.py --run runs/rag/bp_v2_evidence_official119_judged42.jsonl || true
fi

echo "== $(date '+%F %T') stage 2: addendum B, grounded coverage on dev22"
wait_gpu_free
G=""
for b in bp_agentic_v2_dev22 bp_v2_evidence_dev22 bp_v2_multipass_dev22; do
    for v in grounded supported; do G="$G runs/rag/${b}_${v}.jsonl"; done
done
python eval/nugget_local_judge.py --run $G --out eval/analysis/nugget_dev22_grounded.json --unload-after
python eval/analysis/grounded_coverage_dev22.py report

echo "== $(date '+%F %T') stage 3: the other 77"
wait_gpu_free
python scripts/run_synth_arms.py --topic-set official119 --arms evidence --expect-topics 119 \
    --synth-model qwen3.5:122b-gpu36 --unload-after
python eval/validate_submission.py --run runs/rag/bp_v2_evidence_official119.jsonl
echo "== $(date '+%F %T') done"
