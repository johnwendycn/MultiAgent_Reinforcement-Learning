"""Standardized policy evaluation engine producing immutable raw CSV files."""

import os
import csv
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.envs.grf_wrapper import FootballMultiAgentEnv
from src.eval.metrics import FootballMetricsTracker
from src.features.semantic_extractor import SemanticFeatureExtractor
from src.models.actor_critic import MAPPOPolicy


class PolicyEvaluator:
    """Evaluates multi-agent policies and records raw, immutable CSV outcomes.

    Strictly complies with the immutability protocol: each evaluation produces
    a timestamped CSV under `results/raw/` that is never modified after generation.
    """

    CSV_HEADER = [
        "episode_idx",
        "run_id",
        "checkpoint_step",
        "scenario",
        "outcome",
        "goals_home",
        "goals_away",
        "episode_length",
        "cumulative_reward",
        "passes_attempted",
        "passes_completed",
        "shots",
        "xg_sum",
        "home_possession_pct",
        "timestamp_utc",
    ]

    def __init__(
        self,
        env: FootballMultiAgentEnv,
        raw_output_dir: str = "results/raw",
        feature_extractor: Optional[SemanticFeatureExtractor] = None,
    ) -> None:
        """Initializes the PolicyEvaluator.

        Args:
            env: Initialized FootballMultiAgentEnv instance.
            raw_output_dir: Directory where raw CSV logs are stored.
            feature_extractor: Optional semantic feature extractor for xG calculation.
        """
        self.env = env
        self.raw_output_dir = Path(raw_output_dir)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = feature_extractor or SemanticFeatureExtractor()
        self.metrics_tracker = FootballMetricsTracker()

    def evaluate(
        self,
        policy: MAPPOPolicy,
        num_episodes: int = 20,
        run_id: str = "eval_run",
        checkpoint_step: int = 0,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> Tuple[str, Dict[str, float]]:
        """Runs evaluation rollouts and writes raw CSV entries.

        Args:
            policy: Policy model to evaluate.
            num_episodes: Number of test episodes to simulate.
            run_id: Unique run ID for traceability.
            checkpoint_step: Associated training step of policy weights.
            deterministic: Whether to take argmax greedy actions.
            device: Computing device ('cpu' or 'cuda').

        Returns:
            Tuple of (generated_csv_path, summary_metrics_dict).
        """
        policy.eval()
        torch_device = torch.device(device)
        policy.to(torch_device)
        self.metrics_tracker.reset()

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_filename = f"eval_{run_id}_step{checkpoint_step}_{timestamp_str}.csv"
        csv_path = self.raw_output_dir / csv_filename

        rows: List[List[Any]] = []

        for ep_idx in range(num_episodes):
            obs_dict, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            goals_home = 0
            goals_away = 0
            passes_attempted = 0
            passes_completed = 0
            shots = 0
            xg_sum = 0.0
            home_poss_steps = 0
            away_poss_steps = 0

            while not done:
                ep_steps += 1
                actions_dict: Dict[str, int] = {}

                # Action selection for each active agent
                with torch.no_grad():
                    for agent_id in self.env.agents:
                        obs_tensor = torch.tensor(
                            obs_dict[agent_id], dtype=torch.float32, device=torch_device
                        ).unsqueeze(0)
                        action, _ = policy.act(obs_tensor, deterministic=deterministic)
                        actions_dict[agent_id] = int(action.item())

                next_obs_dict, rewards_dict, terminations, truncations, infos = self.env.step(
                    actions_dict
                )

                # Compute domain indicators from ego observation
                if len(obs_dict) > 0:
                    first_agent = list(obs_dict.keys())[0]
                    ego_obs = obs_dict[first_agent]
                    feats = self.feature_extractor.extract_features(ego_obs)
                    xg_sum += feats.get("xg_proxy", 0.0)

                    poss = feats.get("ball_possession_team", 0)
                    if poss == 1:
                        home_poss_steps += 1
                    elif poss == 2:
                        away_poss_steps += 1

                # Tally step rewards
                step_rew = sum(rewards_dict.values())
                ep_reward += step_rew

                # Check goal scoring (GRF reward +1 for home goal, -1 for away goal)
                for r in rewards_dict.values():
                    if r >= 1.0:
                        goals_home += 1
                    elif r <= -1.0:
                        goals_away += 1

                done = any(terminations.values()) or any(truncations.values())
                obs_dict = next_obs_dict

            # Determine match outcome
            if goals_home > goals_away:
                outcome = "WIN"
            elif goals_home == goals_away:
                outcome = "DRAW"
            else:
                outcome = "LOSS"

            total_poss = max(1, home_poss_steps + away_poss_steps)
            home_poss_pct = float(home_poss_steps / total_poss) * 100.0

            self.metrics_tracker.record_episode(
                goals_home=goals_home,
                goals_away=goals_away,
                steps=ep_steps,
                passes_attempted=passes_attempted,
                passes_completed=passes_completed,
                shots=shots,
                xg_sum=xg_sum,
                home_possession_steps=home_poss_steps,
                away_possession_steps=away_poss_steps,
            )

            row = [
                ep_idx + 1,
                run_id,
                checkpoint_step,
                self.env.scenario_name,
                outcome,
                goals_home,
                goals_away,
                ep_steps,
                round(ep_reward, 4),
                passes_attempted,
                passes_completed,
                shots,
                round(xg_sum, 4),
                round(home_poss_pct, 2),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ]
            rows.append(row)

        # Write immutable CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)
            writer.writerows(rows)

        # Set read-only permissions if platform supports it (immutability enforcement)
        try:
            os.chmod(csv_path, 0o444)
        except Exception:
            pass

        summary = self.metrics_tracker.compute_summary()
        return str(csv_path), summary
