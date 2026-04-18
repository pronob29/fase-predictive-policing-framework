#!/bin/bash
# =============================================================================
# FASE — Master Pipeline  (setup → data → train → simulate)
#
# Usage:
#   bash jobs/run_all_phases.sh
#
# Monitor:
#   squeue -u $USER --clusters=chip-gpu
#   tail -f logs/fase/phase3_train_<JOB_ID>.out
#
# Final outputs:
#   checkpoints/fase/best_model.pt
#   results/fase/training_log.csv
#   results/fase/simulation_history.csv
# =============================================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs/fase checkpoints/fase results/fase

echo "============================================================"
echo "FASE — Submitting full pipeline"
echo "$(date)"
echo "============================================================"

# Strip cluster suffix from --parsable output (e.g. "12345;chip-gpu" → "12345")
jid() { echo "$1" | cut -d';' -f1; }

# 0. CUDA env setup (GPU node, installs CUDA torch into venv)
JOB0_RAW=$(sbatch --parsable jobs/setup_cuda_env.sh)
JOB0=$(jid "$JOB0_RAW")
echo "  [0] setup_cuda_env   : job ${JOB0}"

# 1. Phase 1+2 — data engineering + graph (~2 min)
JOB1_RAW=$(sbatch --parsable \
    --dependency=afterok:${JOB0} \
    jobs/run_phase1_data.sh)
JOB1=$(jid "$JOB1_RAW")
echo "  [1] phase1_data      : job ${JOB1}  (after ${JOB0})"

# 2. Phase 3 — model training (GPU, ~1–2 h)
JOB2_RAW=$(sbatch --parsable \
    --dependency=afterok:${JOB1} \
    jobs/run_phase3_train.sh)
JOB2=$(jid "$JOB2_RAW")
echo "  [2] phase3_train     : job ${JOB2}  (after ${JOB1})"

# 3. Phase 5 — feedback simulation (GPU, ~2–4 h)
JOB3_RAW=$(sbatch --parsable \
    --dependency=afterok:${JOB2} \
    jobs/run_phase5_simulate.sh)
JOB3=$(jid "$JOB3_RAW")
echo "  [3] phase5_simulate  : job ${JOB3}  (after ${JOB2})"

echo ""
echo "All jobs queued.  Chain: ${JOB0} → ${JOB1} → ${JOB2} → ${JOB3}"
echo ""
echo "Monitor   :  squeue -u ${USER} --clusters=chip-gpu"
echo "Train log :  tail -f logs/fase/phase3_train_${JOB2}.out"
echo "Sim log   :  tail -f logs/fase/phase5_simulate_${JOB3}.out"
echo ""
echo "Results will appear in:"
echo "  checkpoints/fase/best_model.pt"
echo "  results/fase/training_log.csv"
echo "  results/fase/simulation_history.csv"
echo "============================================================"
