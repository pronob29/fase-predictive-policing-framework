#!/bin/bash
# =============================================================================
# Phase 3: FASE Model Training on GPU  (~1–2 h for 100 epochs on RTX 2080 Ti)
# =============================================================================
#SBATCH --job-name=fase_train
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fase/phase3_train_%j.out
#SBATCH --error=logs/fase/phase3_train_%j.err

source ~/.bashrc 2>/dev/null || true

PROJ_ROOT="${SLURM_SUBMIT_DIR}"
VENV="${PROJ_ROOT}/.venv"
export PATH="${VENV}/bin:$PATH"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"
cd "${PROJ_ROOT}"

mkdir -p logs/fase checkpoints/fase results/fase

# Load same CUDA module used during setup
for CUDA_VER in 12.6.0 12.5.0 12.3.0 12.8.0; do
    if module load CUDA/${CUDA_VER} 2>/dev/null; then
        echo "Loaded module: CUDA/${CUDA_VER}"; break
    fi
done

echo "============================================================"
echo "FASE — Phase 3: GPU Training"
echo "Start   : $(date)"
echo "Host    : $(hostname)"
echo "SLURM   : job=${SLURM_JOB_ID}  node=${SLURM_NODELIST}"
echo "============================================================"

# ── Verify GPU ────────────────────────────────────────────────────────────────
python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. setup_cuda_env.sh must run first.", file=sys.stderr)
    sys.exit(1)
print(f"torch  : {torch.__version__}")
print(f"CUDA   : {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF
[ $? -ne 0 ] && exit 1

echo ""
echo "── Training ────────────────────────────────────────────────"
python -m fase.train \
    --config          configs/fase_config.yaml \
    --checkpoint-dir  checkpoints/fase \
    --results-dir     results/fase \
    --batch-size      32

EXIT_CODE=$?
echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training complete.  End : $(date)"
    echo "Best model  → checkpoints/fase/best_model.pt"
    echo "Log         → results/fase/training_log.csv"
    ls -lh checkpoints/fase/ results/fase/ 2>/dev/null
else
    echo "ERROR: training failed (exit ${EXIT_CODE})" >&2
fi
echo "============================================================"
exit ${EXIT_CODE}
