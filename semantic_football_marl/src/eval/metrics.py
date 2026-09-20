"""Domain metrics tracker for football simulation evaluations."""

from typing import Any, Dict, List
import numpy as np


class FootballMetricsTracker:
    """Accumulates match events and computes statistical performance indicators."""

    def __init__(self) -> None:
        """Initializes the metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Clears accumulators for a new match or evaluation round."""
        self.episode_count = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.goals_scored = 0
        self.goals_conceded = 0
        self.total_passes_attempted = 0
        self.total_passes_completed = 0
        self.total_shots = 0
        self.total_xg = 0.0
        self.possession_steps_home = 0
        self.possession_steps_away = 0
        self.total_steps = 0

    def record_episode(
        self,
        goals_home: int,
        goals_away: int,
        steps: int,
        passes_attempted: int = 0,
        passes_completed: int = 0,
        shots: int = 0,
        xg_sum: float = 0.0,
        home_possession_steps: int = 0,
        away_possession_steps: int = 0,
    ) -> None:
        """Records the outcome of a single evaluation episode.

        Args:
            goals_home: Goals scored by the evaluated team.
            goals_away: Goals scored by the opponent team.
            steps: Number of simulation steps in the episode.
            passes_attempted: Number of pass attempts.
            passes_completed: Number of successfully received passes.
            shots: Total shot actions dispatched.
            xg_sum: Cumulative expected goal proxy value.
            home_possession_steps: Steps where home team had ball ownership.
            away_possession_steps: Steps where away team had ball ownership.
        """
        self.episode_count += 1
        self.goals_scored += goals_home
        self.goals_conceded += goals_away
        self.total_steps += steps
        self.total_passes_attempted += passes_attempted
        self.total_passes_completed += passes_completed
        self.total_shots += shots
        self.total_xg += xg_sum
        self.possession_steps_home += home_possession_steps
        self.possession_steps_away += away_possession_steps

        if goals_home > goals_away:
            self.wins += 1
        elif goals_home == goals_away:
            self.draws += 1
        else:
            self.losses += 1

    def compute_summary(self) -> Dict[str, float]:
        """Calculates aggregated summary statistics across all recorded episodes.

        Returns:
            Dictionary containing win rate, goal differential, pass completion rate, etc.
        """
        n = max(1, self.episode_count)
        total_poss = max(1, self.possession_steps_home + self.possession_steps_away)

        return {
            "episodes": float(self.episode_count),
            "win_rate": float(self.wins / n),
            "draw_rate": float(self.draws / n),
            "loss_rate": float(self.losses / n),
            "avg_goals_scored": float(self.goals_scored / n),
            "avg_goals_conceded": float(self.goals_conceded / n),
            "goal_difference": float((self.goals_scored - self.goals_conceded) / n),
            "pass_completion_rate": float(
                self.total_passes_completed / max(1, self.total_passes_attempted)
            ),
            "avg_shots_per_game": float(self.total_shots / n),
            "avg_xg_per_game": float(self.total_xg / n),
            "possession_share_home": float(self.possession_steps_home / total_poss),
            "avg_episode_length": float(self.total_steps / n),
        }
