"""Structured logging utility linking evaluation CSVs, checkpoints, and run metadata."""

import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """Manages experiment metadata, linking evaluation CSVs to model checkpoints.

    Attributes:
        run_id: Unique identifier for the training or evaluation run.
        logs_dir: Directory where run manifest JSON files are persisted.
        metadata: In-memory dictionary tracking run events, paths, and metrics.
    """

    def __init__(self, run_id: str, logs_dir: str = "results/logs") -> None:
        """Initializes the RunLogger.

        Args:
            run_id: Unique run identifier.
            logs_dir: Path to the logs directory. Defaults to 'results/logs'.
        """
        self.run_id = run_id
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.logs_dir / f"run_{self.run_id}.json"

        self.metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoints": [],
            "raw_eval_csvs": [],
            "metrics_history": [],
            "config_snapshot": {},
            "status": "INITIALIZED"
        }
        self.save()

    def set_config(self, config_dict: Dict[str, Any]) -> None:
        """Stores a snapshot of the experiment configuration.

        Args:
            config_dict: Dictionary representation of the Hydra configuration.
        """
        self.metadata["config_snapshot"] = config_dict
        self.save()

    def register_checkpoint(self, checkpoint_path: str, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """Records a saved model checkpoint.

        Args:
            checkpoint_path: Path to the saved PyTorch weights file.
            step: Timestep at which the checkpoint was captured.
            metrics: Optional evaluation metrics associated with the checkpoint.
        """
        entry = {
            "step": step,
            "path": str(checkpoint_path),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metrics": metrics or {}
        }
        self.metadata["checkpoints"].append(entry)
        self.save()

    def register_raw_csv(self, csv_path: str, evaluation_step: int, num_episodes: int) -> None:
        """Records an immutable raw evaluation CSV output.

        Args:
            csv_path: Relative or absolute path to the generated CSV.
            evaluation_step: Timestep at which evaluation took place.
            num_episodes: Number of episodes evaluated in the CSV.
        """
        entry = {
            "step": evaluation_step,
            "csv_path": str(csv_path),
            "num_episodes": num_episodes,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        self.metadata["raw_eval_csvs"].append(entry)
        self.save()

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Logs scalar metrics for a given step.

        Args:
            step: Current training or evaluation step.
            metrics: Dictionary of metric names and float values.
        """
        entry = {
            "step": step,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metrics": metrics
        }
        self.metadata["metrics_history"].append(entry)
        self.save()

    def set_status(self, status: str) -> None:
        """Updates the operational status of the run.

        Args:
            status: Status string (e.g. 'RUNNING', 'COMPLETED', 'FAILED').
        """
        self.metadata["status"] = status
        self.metadata["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.save()

    def save(self) -> None:
        """Persists the metadata dictionary to JSON."""
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
