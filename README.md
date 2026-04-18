# FASE Predictive Policing Framework

FASE stands for **Fairness Aware Spatial Event Graph**. This repository contains a multi-phase research and engineering pipeline for fairness-aware predictive policing experiments using:

- spatiotemporal tensor construction from incident data
- spatial graph generation over Baltimore ZCTAs
- a dual-engine forecasting model combining an STGNN and Hawkes excitation
- fairness-constrained patrol allocation
- deployment feedback simulation and result visualization

## Repository Layout

- `fase/data/` builds tensors and the spatial graph
- `fase/models/` contains the STGNN, Hawkes, losses, and top-level `FASEModel`
- `fase/allocation/` contains the patrol allocator and fairness metrics
- `fase/simulation/` contains the deployment feedback simulator
- `fase/train.py` trains the model
- `fase/simulate.py` runs the deployment simulation
- `fase/visualize.py` generates result figures
- `configs/fase_config.yaml` defines paths, model settings, and phase parameters
- `jobs/` contains SLURM scripts for cluster execution

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some dependencies, especially `torch-geometric`, may require CUDA- or platform-specific wheels.

## Expected Data

Large datasets, tensors, checkpoints, logs, and generated outputs are intentionally not committed to this repository.

The pipeline expects local inputs such as:

- raw crime incident CSV files under `data/raw/`
- ACS and shapefile assets referenced by `configs/fase_config.yaml`
- generated tensors under `data/tensors/`

## Typical Workflow

```bash
python -m fase.data.build_tensors
python -m fase.data.build_graph
python -m fase.train
python -m fase.simulate
python -m fase.visualize
```

For cluster execution, use the scripts in `jobs/`, especially:

```bash
bash jobs/run_all_phases.sh
```

## Notes

- The default configuration targets Baltimore data.
- Generated model checkpoints and result artifacts are reproducible and therefore excluded from version control.
- This repository is structured for research iteration as well as batch execution on SLURM-based GPU environments.
