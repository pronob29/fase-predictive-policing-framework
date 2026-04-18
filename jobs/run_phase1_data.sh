#!/bin/bash
# =============================================================================
# Phase 1 + 2: Data Engineering & Graph Construction  (~2 min)
# =============================================================================
#SBATCH --job-name=fase_data
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fase/phase1_data_%j.out
#SBATCH --error=logs/fase/phase1_data_%j.err

source ~/.bashrc 2>/dev/null || true

PROJ_ROOT="${SLURM_SUBMIT_DIR}"
VENV="${PROJ_ROOT}/.venv"
export PATH="${VENV}/bin:$PATH"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"
cd "${PROJ_ROOT}"

mkdir -p logs/fase data/tensors/baltimore

echo "============================================================"
echo "FASE — Phase 1+2: Data Engineering & Graph Construction"
echo "Start    : $(date)"
echo "Host     : $(hostname)"
echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "============================================================"

echo ""
echo "── Phase 1: build_tensors ──────────────────────────────────"
python -m fase.data.build_tensors \
    --years 2017 2018 2019 \
    --freq  1h
[ $? -ne 0 ] && { echo "ERROR: build_tensors failed" >&2; exit 1; }

echo ""
echo "── Phase 2: build_graph ────────────────────────────────────"
python -m fase.data.build_graph
[ $? -ne 0 ] && { echo "ERROR: build_graph failed" >&2; exit 1; }

echo ""
echo "============================================================"
echo "Phases 1+2 complete.  End : $(date)"
echo "Tensors → data/tensors/baltimore/"
ls -lh data/tensors/baltimore/*.pt data/tensors/baltimore/*.pkl 2>/dev/null
echo "============================================================"
