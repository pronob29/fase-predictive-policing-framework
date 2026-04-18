#!/bin/bash
# =============================================================================
# Phase 5: Deployment Feedback Simulator on GPU  (~2–4 h, 6 cycles × 50 epochs)
# =============================================================================
#SBATCH --job-name=fase_simulate
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fase/phase5_simulate_%j.out
#SBATCH --error=logs/fase/phase5_simulate_%j.err

source ~/.bashrc 2>/dev/null || true

PROJ_ROOT="${SLURM_SUBMIT_DIR}"
VENV="${PROJ_ROOT}/.venv"
export PATH="${VENV}/bin:$PATH"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"
cd "${PROJ_ROOT}"

mkdir -p logs/fase results/fase

# Load same CUDA module used during setup
for CUDA_VER in 12.6.0 12.5.0 12.3.0 12.8.0; do
    if module load CUDA/${CUDA_VER} 2>/dev/null; then
        echo "Loaded module: CUDA/${CUDA_VER}"; break
    fi
done

echo "============================================================"
echo "FASE — Phase 5: Deployment Feedback Simulation"
echo "Start   : $(date)"
echo "Host    : $(hostname)"
echo "SLURM   : job=${SLURM_JOB_ID}  node=${SLURM_NODELIST}"
echo "============================================================"

CKPT="checkpoints/fase/best_model.pt"
[ ! -f "${CKPT}" ] && { echo "ERROR: ${CKPT} not found. Run phase3 first." >&2; exit 1; }
echo "Checkpoint : ${CKPT}"

python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available", file=sys.stderr); sys.exit(1)
print(f"GPU : {torch.cuda.get_device_name(0)}")
PYEOF
[ $? -ne 0 ] && exit 1

echo ""
echo "── Running simulation ──────────────────────────────────────"
python -m fase.simulate \
    --config         configs/fase_config.yaml \
    --checkpoint     "${CKPT}" \
    --results-dir    results/fase \
    --window-size    64

EXIT_CODE=$?
echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Simulation complete.  End : $(date)"
    echo "Results → results/fase/simulation_history.csv"
    ls -lh results/fase/ 2>/dev/null
else
    echo "ERROR: simulation failed (exit ${EXIT_CODE})" >&2
fi
echo "============================================================"
exit ${EXIT_CODE}
