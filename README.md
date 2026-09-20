# MultiAgent_Reinforcement-Learning

## Google Research Football (GRF) - Multi-Agent RL Setup

This directory contains a complete setup for training and experimenting with Multi-Agent Reinforcement Learning (MARL) algorithms on Google Research Football (GRF) using Google Colab and local environments.

---

## Files

- [reinforcement.ipynb](file:///c:/All%20Models/Reinforcement%20Model%20for%20Football%20Sport/reinforcement.ipynb): Google Colab ready notebook for environment provisioning, hardware verification, and multi-agent simulation.
- [requirements.txt](file:///c:/All%20Models/Reinforcement%20Model%20for%20Football%20Sport/requirements.txt): Exact pinned dependencies (no floating or "latest" versions).
- [environment_manifest.json](file:///c:/All%20Models/Reinforcement%20Model%20for%20Football%20Sport/environment_manifest.json): Machine-readable JSON manifest detailing platform, GPU, CUDA, and all installed package versions.
- [generate_manifest.py](file:///c:/All%20Models/Reinforcement%20Model%20for%20Football%20Sport/generate_manifest.py): Standalone CLI utility to inspect hardware, audit packages, and produce `environment_manifest.json`.

---

## Pinned Dependencies

| Package | Exact Pinned Version | Rationale & Compatibility |
| :--- | :--- | :--- |
| `torch` | `2.4.1+cu121` | PyTorch with CUDA 12.1 runtime acceleration matching Google Colab GPUs (T4, L4, A100). |
| `torchvision` | `0.19.1+cu121` | Vision architectures and tensor transforms for pixel-based football representations. |
| `torchaudio` | `2.4.1+cu121` | PyTorch audio ecosystem matching core torch build. |
| `gfootball` | `2.9` | Official Google Research Football 3D environment engine. |
| `numpy` | `1.26.4` | Pinned to 1.26.4 to prevent C-extension ABI breaks occurring in NumPy 2.0 with Gym/GFootball. |
| `scipy` | `1.13.1` | Scientific routines and spatial distance matrices. |
| `pandas` | `2.2.3` | High-performance event log and trajectory dataframes. |
| `matplotlib` | `3.9.2` | Visualizing pitch heatmaps, pass maps, and reward curves. |
| `seaborn` | `0.13.2` | Statistical plotting and distribution visualizations. |
| `gym` | `0.26.2` | Underlying legacy Gym environment API expected by GFootball engine. |
| `gymnasium` | `0.29.1` | Modern standard Farama Foundation environment interface. |
| `pettingzoo` | `1.24.3` | De-facto multi-agent reinforcement learning (MARL) API standard. |
| `shimmy` | `1.3.0` | Bidirectional translation layer between Gym, Gymnasium, and PettingZoo environments. |
| `statsbombpy` | `1.16.0` | Access to open-source real-world football event and spatial tracking data. |
| `wandb` | `0.19.11` | Weights & Biases experiment tracking, hyperparameter tuning, and metric logging. |
| `hydra-core` | `1.3.2` | Hierarchical configuration management for complex MARL scenarios and architectures. |
| `omegaconf` | `2.3.0` | Flexible YAML configuration backend for Hydra. |
| `einops` | `0.8.0` | Expressive and readable tensor dimension reshaping for multi-agent policy networks. |
| `scikit-learn` | `1.5.2` | Clustering, baseline ML models, dimensionality reduction, and evaluation metrics. |

---

## Running in Google Colab

1. Upload `reinforcement.ipynb` and `requirements.txt` to Google Drive or open via GitHub.
2. In Google Colab, enable GPU acceleration:
   - Go to **Runtime > Change runtime type**.
   - Select **T4 GPU** (or A100/L4).
   - Click **Save**.
3. Run all cells:
   - **Step 1:** Installs Linux `apt-get` packages (`cmake`, `libsdl2-dev`, `libboost-all-dev`, `xvfb`, etc.) needed to compile GRF C++ binaries.
   - **Step 2 & 3:** Installs all pinned dependencies from `requirements.txt`.
   - **Step 4:** Inspects GPU name, CUDA version, and all installed packages, printing them to the console and saving to `environment_manifest.json`.
   - **Step 5:** Executes a 3-agent cooperative scenario (`academy_3_vs_1_with_keeper`) smoke test under headless virtual display (`Xvfb`).
   - **Step 6:** Validates data analytics (`statsbombpy`), config management (`hydra`), and tensor manipulation (`einops`).
