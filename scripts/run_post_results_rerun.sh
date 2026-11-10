#!/bin/bash
# Post-results rerun of the writer on dev22, as pre-registered in
# eval/analysis/PREREG_post_results_rerun.md.
#
# The sequencing lives in bash so it survives the session that launched it (see run_r2_stages.sh).
#
# STAGES
#   1  score the existing dev22 answers (shipped v2, probe, 32B, v3) on the official dev nuggets,
#      judge qwen3:32b                                 -> eval/analysis/nugget_dev22_existing.json
#   2  rewrite dev22 under the four arms, writer qwen3.5:122b, from the cached retrieval
#                                                      -> runs/rag/bp_v2_<arm>_dev22.jsonl
#   3  score the four arms, judge qwen3:32b            -> eval/analysis/nugget_dev22_arms.json
#   4  paired comparison                               -> stdout and the log
#   [vram] each stage unloads its model: 122b (~88 GB resident) and 32b cannot share the A100.
#
# Idempotent: the writer skips topics whose answer_<arm>.json exists, and the judge caches every
# window in data/rag/cache/nugget_assignments.jsonl, so a re-run resumes where it stopped.
#
# USAGE
#   bash scripts/run_post_results_rerun.sh
#   stop: pkill -f run_post_results_rerun.sh; pkill -f run_synth_arms.py; pkill -f nugget_local_judge.py
set -euo pipefail
cd "$(dirname "$0")/.."
source "${VENV:-.venv}/bin/activate"

ARMS="rerun budget coverage evidence"

# keep_alive=0 returns before the runner has released its VRAM. On 2026-11-08 the 122b load
# started one second after the 32b unload and died with CUDA out of memory, so wait for the card.
wait_gpu_free() {
    ollama stop qwen3:32b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b > /dev/null 2>&1 || true
    ollama stop qwen3.5:122b-gpu36 > /dev/null 2>&1 || true
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do
        sleep 5
    done
}
EXISTING="runs/rag/bp_agentic_v2_dev22.jsonl runs/rag/bp_agentic_v2_dev22_probe.jsonl
          runs/rag/bp_agentic_v2_32b_dev22.jsonl runs/rag/bp_agentic_v3_dev22.jsonl"

echo "== $(date '+%F %T') stage 1: score existing dev22 answers"
python eval/nugget_local_judge.py --run $EXISTING \
    --out eval/analysis/nugget_dev22_existing.json --unload-after

echo "== $(date '+%F %T') stage 2: rewrite dev22 under the arms"
wait_gpu_free
# qwen3.5:122b-gpu36 is qwen3.5:122b with PARAMETER num_gpu 36: same weights, four fewer layers
# on the GPU. With Ollama's default of 40 the runner ran out of CUDA memory mid-generation.
python scripts/run_synth_arms.py --topic-set dev22 --arms $ARMS --expect-topics 22 \
    --synth-model qwen3.5:122b-gpu36 --unload-after

echo "== $(date '+%F %T') stage 3: score the arms"
wait_gpu_free
ARM_RUNS=""
for a in $ARMS; do
    f="runs/rag/bp_v2_${a}_dev22.jsonl"
    [ "$(wc -l < "$f")" -eq 22 ] || { echo "ABORT: $f does not have 22 answers"; exit 1; }
    ARM_RUNS="$ARM_RUNS $f"
done
python eval/nugget_local_judge.py --run $ARM_RUNS \
    --out eval/analysis/nugget_dev22_arms.json --unload-after

echo "== $(date '+%F %T') stage 4: paired comparison"
python eval/analysis/post_results_rerun_dev22.py \
    eval/analysis/nugget_dev22_existing.json eval/analysis/nugget_dev22_arms.json
echo "== $(date '+%F %T') done"
