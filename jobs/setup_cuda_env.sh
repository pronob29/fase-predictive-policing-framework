#!/bin/bash
# =============================================================================
# One-time CUDA environment setup.
# Installs a CUDA-enabled PyTorch build into the project venv.
# /tmp on compute nodes is tiny — TMPDIR is redirected to project filesystem.
# =============================================================================
#SBATCH --job-name=fase_setup
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fase/setup_cuda_%j.out
#SBATCH --error=logs/fase/setup_cuda_%j.err

source ~/.bashrc 2>/dev/null || true

# Use SLURM_SUBMIT_DIR (set by sbatch) for reliable project root
PROJ_ROOT="${SLURM_SUBMIT_DIR}"
VENV="${PROJ_ROOT}/.venv"
export PATH="${VENV}/bin:$PATH"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH}"

# Redirect pip's tmp/cache to the 25 TB project filesystem
# so compute-node /tmp (often <10 GB) doesn't fill up during torch install
PIP_TMP="${PROJ_ROOT}/.pip_tmp"
mkdir -p "${PIP_TMP}" logs/fase
export TMPDIR="${PIP_TMP}"
export PIP_CACHE_DIR="${PIP_TMP}/cache"
export PIP_TMPDIR="${PIP_TMP}"

cd "${PROJ_ROOT}"

echo "============================================================"
echo "FASE — CUDA Environment Setup"
echo "Start    : $(date)"
echo "Host     : $(hostname)"
echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "TMPDIR   : ${TMPDIR}"
echo "============================================================"

# ── Load best available CUDA module ──────────────────────────────────────────
for CUDA_VER in 12.6.0 12.5.0 12.3.0 12.8.0; do
    if module load CUDA/${CUDA_VER} 2>/dev/null; then
        echo "Loaded module: CUDA/${CUDA_VER}"
        CUDA_LOADED="${CUDA_VER}"
        break
    fi
done
[ -z "${CUDA_LOADED}" ] && echo "WARNING: no CUDA module loaded"

# Map version → PyTorch wheel tag; 12.8/12.9 fall back to cu126
MAJOR="${CUDA_LOADED%%.*}"
MINOR="${CUDA_LOADED#*.}"; MINOR="${MINOR%%.*}"
RAW_TAG="cu${MAJOR}${MINOR}"
case "${RAW_TAG}" in
    cu126|cu125|cu124|cu123|cu121) CUDA_TAG="${RAW_TAG}" ;;
    cu128|cu129)                    CUDA_TAG="cu126"      ;;
    cu117|cu116)                    CUDA_TAG="cu117"      ;;
    *)                              CUDA_TAG="cu126"      ;;
esac
echo "PyTorch wheel tag: ${CUDA_TAG}"
echo ""

# ── Install CUDA-enabled torch ────────────────────────────────────────────────
echo "Downloading and installing torch+${CUDA_TAG} …"
echo "(large download ~2 GB — TMPDIR=${TMPDIR})"

pip install torch \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    --upgrade \
    --no-cache-dir

EXIT_PIP=$?
if [ ${EXIT_PIP} -ne 0 ]; then
    echo "ERROR: pip install failed (exit ${EXIT_PIP})" >&2
    exit ${EXIT_PIP}
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "── Verifying GPU tensor ops ────────────────────────────────"
python - <<'PYEOF'
import torch, sys
print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() = False", file=sys.stderr)
    sys.exit(1)
print(f"CUDA   : {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
x = torch.ones(1024, 1024, device='cuda')
print(f"Matmul : {(x @ x).mean().item():.1f}  (GPU matmul OK)")
PYEOF

EXIT_VER=$?
echo ""

# ── Clean up pip tmp ─────────────────────────────────────────────────────────
rm -rf "${PIP_TMP}"

echo "============================================================"
if [ ${EXIT_VER} -eq 0 ]; then
    echo "Setup complete.  End : $(date)"
    echo "CUDA torch is ready in ${VENV}"
else
    echo "ERROR: GPU verification failed." >&2
fi
echo "============================================================"
exit ${EXIT_VER}
