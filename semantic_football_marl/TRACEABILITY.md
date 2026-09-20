# Research Manuscript Traceability Matrix

This document provides a strict, one-to-one mapping between all empirical claims, hypotheses, and reported statistics in the research manuscript and their underlying computational artifacts (**Hydra configuration**, **model checkpoint weights**, **immutable raw evaluation CSV**, **processed aggregation table**, and **rendered figure**).

---

## Traceability Policy & Rules

1. **Immutable Raw Data:** All numbers cited in the manuscript originate from `results/raw/*.csv` files generated exclusively by `scripts/evaluate.py`. No numbers are handwritten or manually adjusted.
2. **Aggregated Summaries:** Aggregations in `results/processed/*.csv` are produced strictly by executing `scripts/aggregate_results.py`.
3. **Analysis Isolation:** Notebooks under `notebooks/` only query `results/processed/` and never generate or hardcode metrics.

---

## Manuscript Claims Mapping Matrix

| Claim ID | Manuscript Section / Target | Claim Summary & Benchmark Hypothesis | Hydra Configuration | Checkpoint Weights | Immutable Raw Evaluation CSV | Processed CSV / Figure | W&B Run ID / Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Section 4.1, Table 1 | *Baseline MAPPO with sparse rewards achieves <30% win rate on Academy 3v1.* | [`configs/experiment/exp01_baseline_mappo.yaml`](configs/experiment/exp01_baseline_mappo.yaml) | `results/checkpoints/exp01_baseline_mappo_final.pt` | `results/raw/eval_exp01_baseline_mappo_step50000_*.csv` | `results/processed/benchmark_leaderboard.csv` | `exp01_baseline_mappo_seed42` |
| **CLM-02** | Section 4.2, Figure 3 | *Dense semantic reward shaping accelerates convergence by 3.2x and raises win rate to >75%.* | [`configs/experiment/exp02_semantic_mappo.yaml`](configs/experiment/exp02_semantic_mappo.yaml) | `results/checkpoints/exp02_semantic_mappo_final.pt` | `results/raw/eval_exp02_semantic_mappo_step50000_*.csv` | `results/figures/learning_curves.png` | `exp02_semantic_mappo_seed42` |
| **CLM-03** | Section 4.3, Figure 4 | *Centralized value critic in MAPPO outperforms decentralized QMIX in passing coordination.* | [`configs/algo/mappo.yaml`](configs/algo/mappo.yaml) vs [`configs/algo/qmix.yaml`](configs/algo/qmix.yaml) | `results/checkpoints/exp02_semantic_mappo_final.pt` | `results/raw/eval_exp02_semantic_mappo_step50000_*.csv` | `results/figures/winrate_comparison.png` | `marl_cooperative_comparison` |
| **CLM-04** | Section 4.4, Table 2 | *Passing corridor clearance and xG proxies correlate positively (r > 0.68) with goal conversion.* | [`configs/features/semantic.yaml`](configs/features/semantic.yaml) | `results/checkpoints/exp02_semantic_mappo_step_50000.pt` | `results/raw/eval_exp02_semantic_mappo_*.csv` | `results/processed/summary_metrics.csv` | `feature_ablation_xg` |

---

## Artifact Audit Log Directory (`results/logs/`)

For every completed run, a JSON audit receipt is written to `results/logs/run_<run_id>.json`. Each receipt specifies:
- Exact Git commit hash and execution timestamp
- Full resolved Hydra configuration tree
- Deterministic random seed
- Pointers to generated `.pt` checkpoint files
- Pointers to generated `.csv` evaluation files

### Example Run Audit JSON (`results/logs/run_exp02_semantic_mappo_seed42.json`):
```json
{
  "run_id": "exp02_semantic_mappo_seed42",
  "created_at_utc": "2026-09-19T10:15:00Z",
  "status": "COMPLETED",
  "checkpoints": [
    {
      "step": 50000,
      "path": "results/checkpoints/exp02_semantic_mappo_final.pt",
      "timestamp_utc": "2026-09-19T10:15:00Z"
    }
  ],
  "raw_eval_csvs": [
    {
      "step": 50000,
      "csv_path": "results/raw/eval_exp02_semantic_mappo_step50000_20260919_101500.csv",
      "num_episodes": 20
    }
  ]
}
```

---

## Verification & Reproduction Command

To reproduce all evaluations and regenerate the figures referenced in the manuscript:

```bash
# 1. Run evaluation for Exp 01 (Baseline)
python scripts/evaluate.py --config-name=exp01_baseline_mappo

# 2. Run evaluation for Exp 02 (Semantic Rewards)
python scripts/evaluate.py --config-name=exp02_semantic_mappo

# 3. Aggregate all raw CSVs and regenerate figures
python scripts/aggregate_results.py
```
