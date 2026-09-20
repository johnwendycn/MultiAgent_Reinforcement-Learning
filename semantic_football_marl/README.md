# Semantic Football Multi-Agent Reinforcement Learning (`semantic_football_marl`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.4.1+cu121](https://img.shields.io/badge/PyTorch-2.4.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![Google Research Football](https://img.shields.io/badge/GRF-2.9-brightgreen.svg)](https://github.com/google-research/football)

A modular research framework for Multi-Agent Reinforcement Learning (MARL) on **Google Research Football (GRF)** with domain-driven semantic state representations, dense reward shaping, and an immutable evaluation audit trail.

---

## Directory Structure

```text
semantic_football_marl/
├── configs/            # Composable Hydra experiment configurations
│   ├── algo/           # Algorithm hyperparameters (mappo.yaml, qmix.yaml)
│   ├── env/            # Environment scenarios (grf_academy_3v1.yaml, grf_11v11.yaml)
│   ├── experiment/     # Concrete experiment configs (exp01_baseline, exp02_semantic)
│   ├── features/       # Feature extractor configs (semantic.yaml)
│   ├── model/          # Neural network configs (actor_critic.yaml, q_mixer.yaml)
│   ├── rewards/        # Reward weights configs (dense_semantic.yaml)
│   └── config.yaml     # Root config
├── src/
│   ├── algos/          # MARL trainers (base_algo, mappo, qmix)
│   ├── envs/           # PettingZoo/Gymnasium wrapper & scenario manager
│   ├── eval/           # Standardized evaluator & domain metrics
│   ├── features/       # Semantic extractor & spatial graph transformer
│   ├── models/         # Neural networks (Actor-Critic, QMixer)
│   ├── rewards/        # Modular reward shaping (base_reward, dense_semantic)
│   └── utils/          # Seed, logger, and factory registries
├── scripts/
│   ├── train.py        # Hydra-powered main training script
│   ├── evaluate.py     # Evaluation CLI generating immutable raw CSVs
│   └── aggregate_results.py # Aggregates raw CSVs into summary tables & figures
├── results/
│   ├── raw/            # Immutable raw evaluation CSVs (NEVER edited by hand)
│   ├── processed/      # Aggregated statistics & benchmark leaderboards
│   ├── figures/        # High-resolution publication plots (PNG/PDF)
│   ├── checkpoints/    # Model weight files (*.pt)
│   └── logs/           # JSON audit logs linking CSVs to checkpoints
├── notebooks/
│   └── analysis.ipynb  # Read-only analysis notebook (reads from results/ only)
├── tests/              # Full pytest test suite (envs, features, models, algos)
├── TRACEABILITY.md     # Traceability matrix mapping claims to code & data
├── README.md           # This documentation
├── requirements.txt    # Exact pinned dependencies (CUDA 12.1)
├── Dockerfile          # CUDA-enabled Docker container
└── LICENSE             # MIT License
```

---

## Strict Research Rules & Protocols

1. **Raw CSV Immutability (`results/raw/`):**
   - Every execution of `scripts/evaluate.py` or training evaluation writes a uniquely timestamped CSV to `results/raw/`.
   - Files in `results/raw/` are set to read-only and **MUST NEVER be modified, renamed, or fabricated by hand**.
2. **Derived Summaries (`results/processed/`):**
   - Tables in `results/processed/` are derived strictly by running `scripts/aggregate_results.py`.
3. **Read-Only Analysis (`notebooks/analysis.ipynb`):**
   - The analysis notebook reads exclusively from `results/processed/` and `results/logs/`. It never computes or hardcodes experimental numbers.
4. **Manuscript Traceability (`TRACEABILITY.md`):**
   - Every claim in the manuscript is indexed to an exact Hydra config, checkpoint, raw CSV, and figure.

---

## Installation & Setup

### Option 1: Google Colab (`/content/semantic_football_marl`)

Google Colab provides Ubuntu GPU instances with CUDA acceleration:

```bash
# 1. Install Linux system libraries needed for Google Research Football
!apt-get update -qq
!apt-get install -y -qq git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

# 2. Clone/move into the project directory
%cd /content/semantic_football_marl

# 3. Install pinned dependencies
!pip install --no-cache-dir -r requirements.txt
```

### Option 2: Docker Container

A multi-stage CUDA 12.1 container with pre-installed C++ build tools and Xvfb:

```bash
# Build the Docker image
docker build -t semantic_football_marl:latest .

# Run container with NVIDIA GPU acceleration
docker run --gpus all -it --rm \
    -v $(pwd)/results:/content/semantic_football_marl/results \
    semantic_football_marl:latest
```

### Option 3: Local Python Environment

```bash
# Create virtual environment with Python 3.11
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run Unit Tests
```bash
python -m pytest tests/ -v
```

### 2. Train Multi-Agent Policy (MAPPO)
```bash
# Run baseline experiment
python scripts/train.py --config-name=exp01_baseline_mappo

# Run dense semantic reward experiment
python scripts/train.py --config-name=exp02_semantic_mappo

# Quick smoke test / dry run
python scripts/train.py dry_run=true
```

### 3. Evaluate Policy & Output Immutable Raw CSVs
```bash
python scripts/evaluate.py --config-name=exp02_semantic_mappo \
    checkpoint_path=results/checkpoints/exp02_semantic_mappo_final.pt
```

### 4. Aggregate Results & Generate Publication Figures
```bash
python scripts/aggregate_results.py
```
Outputs:
- `results/processed/summary_metrics.csv`
- `results/processed/benchmark_leaderboard.csv`
- `results/figures/winrate_comparison.png`
- `results/figures/learning_curves.png`

### 5. Inspect Manuscript Figures & Audit Logs
Open `notebooks/analysis.ipynb` in Jupyter or Colab to visualize the processed benchmark data.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
