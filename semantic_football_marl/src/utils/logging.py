"""Production-grade experiment logger with Weights & Biases (W&B) integration,
cryptographic provenance tracking, system telemetry, and zero-crash fallbacks.
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Prevent self-shadowing of standard library 'logging' when executed directly as a script
_script_dir = None
if sys.path and os.path.dirname(__file__) in sys.path[0]:
    _script_dir = sys.path.pop(0)

import logging

if _script_dir is not None:
    sys.path.insert(0, _script_dir)

logger = logging.getLogger("WandBLogger")

# Graceful WandB import fallback
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

# Graceful PyTorch import
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

# Graceful OmegaConf import
try:
    from omegaconf import DictConfig, OmegaConf
    _OMEGACONF_AVAILABLE = True
except ImportError:
    OmegaConf = None
    DictConfig = None
    _OMEGACONF_AVAILABLE = False


def compute_config_hash(config: Any) -> str:
    """Computes a deterministic SHA-256 hash of a Hydra/OmegaConf config or dictionary.

    Args:
        config: Dictionary, OmegaConf DictConfig, or string representation of config.

    Returns:
        64-character hexadecimal SHA-256 hash string.
    """
    if config is None:
        return "none"

    if _OMEGACONF_AVAILABLE and isinstance(config, (DictConfig,)):
        config_dict = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        config_dict = config
    elif isinstance(config, str):
        return hashlib.sha256(config.strip().encode("utf-8")).hexdigest()
    else:
        try:
            config_dict = dict(config)
        except Exception:
            return hashlib.sha256(str(config).encode("utf-8")).hexdigest()

    # Canonical JSON serialization: sorted keys, no extraneous whitespace
    try:
        canonical_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        canonical_json = str(sorted(config_dict.items()))

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file efficiently in streaming chunks.

    Prevents high memory spikes when hashing multi-gigabyte PyTorch checkpoint files.

    Args:
        file_path: Path to the target file.
        chunk_size: Read buffer size in bytes. Defaults to 64KB.

    Returns:
        64-character hexadecimal SHA-256 hash, or 'none' if file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        return "none"

    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute SHA-256 for checkpoint {file_path}: {e}")
        return "error_reading_checkpoint"


def get_git_commit_hash(search_dir: Optional[Union[str, Path]] = None) -> str:
    """Retrieves the current Git commit hash with multi-stage fallback.

    Checks environment variables, git command-line tool, and direct .git/HEAD parsing.

    Args:
        search_dir: Starting directory to locate git information. Defaults to CWD.

    Returns:
        Git commit hash string or 'unversioned_workspace' if unavailable.
    """
    # 1. Environment variable inspection (CI/CD, Slurm, Docker, Cloud Run)
    for env_key in ("GIT_COMMIT", "GIT_SHA", "GITHUB_SHA", "WANDB_GIT_COMMIT", "VCS_REVISION"):
        commit = os.environ.get(env_key)
        if commit:
            return commit.strip()

    # 2. Direct subprocess query via git CLI
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(search_dir) if search_dir else None,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # 3. Direct file inspection of .git/HEAD
    try:
        curr = Path(search_dir).resolve() if search_dir else Path.cwd()
        for parent in [curr] + list(curr.parents):
            git_head = parent / ".git" / "HEAD"
            if git_head.is_file():
                content = git_head.read_text(encoding="utf-8").strip()
                if content.startswith("ref:"):
                    ref_path = parent / ".git" / content.split(" ", 1)[1].strip()
                    if ref_path.is_file():
                        return ref_path.read_text(encoding="utf-8").strip()
                else:
                    return content
    except Exception:
        pass

    return "unversioned_workspace"


class WandBLogger:
    """Production-grade experiment logger with Weights & Biases (W&B) integration,
    immutable cryptographic provenance hashing, system telemetry, and zero-crash fallbacks.

    Every log entry guarantees inclusion of:
    - run_id (linking to config and checkpoint)
    - config_hash (SHA-256 of serialized Hydra config)
    - checkpoint_sha256 (SHA-256 of the active model checkpoint)
    - git_commit_hash
    - seed
    """

    def __init__(
        self,
        project: str = "football-marl",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        seed: int = 42,
        checkpoint_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "results/logs",
        mode: str = "auto",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Initializes the WandBLogger with full provenance metadata.

        Args:
            project: W&B project name.
            entity: W&B entity or organization name.
            group: W&B experiment group name.
            job_type: Type of job ('train', 'eval', 'ablation').
            run_id: Unique run ID. If None, generated automatically.
            run_name: Human-readable run display name.
            config: Hydra DictConfig or dict of experiment hyperparameters.
            seed: Random seed used across the run.
            checkpoint_path: Path to initial model checkpoint if resuming.
            output_dir: Local directory for JSONL provenance audit trails.
            mode: WandB run mode ('online', 'offline', 'disabled', 'auto').
            tags: List of experiment tag strings.
        """
        self.run_id = run_id or f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.run_name = run_name or self.run_id
        self.project = project
        self.entity = entity
        self.group = group
        self.job_type = job_type
        self.seed = int(seed)

        # Cryptographic provenance
        self.config_raw = config
        self.config_hash = compute_config_hash(config)
        self.git_commit_hash = get_git_commit_hash()
        self.active_checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = (
            compute_file_sha256(self.active_checkpoint_path)
            if self.active_checkpoint_path
            else "none"
        )

        # Local audit storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.output_dir / f"{self.run_id}.jsonl"
        self.manifest_path = self.output_dir / f"{self.run_id}_manifest.json"
        self.history: List[Dict[str, Any]] = []

        # WandB Backend Initialization
        self.wandb_run = None
        self._init_wandb(mode=mode, tags=tags)

        # Write initial manifest linking run_id, config, and checkpoint
        self._write_manifest()

    def _init_wandb(self, mode: str, tags: Optional[List[str]]) -> None:
        """Initializes the W&B run or sets up local fallback."""
        if mode == "disabled" or not _WANDB_AVAILABLE:
            if not _WANDB_AVAILABLE and mode not in ("disabled", "auto"):
                logger.warning("wandb is not installed. Falling back to local structured JSONL logging.")
            self.wandb_run = None
            return

        resolved_mode = "online" if mode == "auto" else mode
        # If no API key is found and mode is auto, fall back to offline mode
        if resolved_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            resolved_mode = "offline"

        try:
            config_payload = {
                "seed": self.seed,
                "config_hash": self.config_hash,
                "checkpoint_sha256": self.checkpoint_sha256,
                "git_commit": self.git_commit_hash,
            }
            if isinstance(self.config_raw, dict):
                config_payload.update(self.config_raw)
            elif _OMEGACONF_AVAILABLE and isinstance(self.config_raw, DictConfig):
                config_payload.update(OmegaConf.to_container(self.config_raw, resolve=True))

            self.wandb_run = wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.group,
                job_type=self.job_type,
                id=self.run_id,
                name=self.run_name,
                resume="allow",
                mode=resolved_mode,
                tags=tags,
                config=config_payload,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb run ({e}). Falling back to local logging.")
            self.wandb_run = None

    def _write_manifest(self) -> None:
        """Persists the experiment manifest linking run_id, config_hash, and checkpoint_sha256."""
        manifest_data = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "project": self.project,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_path": self.active_checkpoint_path,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ACTIVE",
        }
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting manifest: {e}")

    def update_checkpoint(self, checkpoint_path: Union[str, Path], step: Optional[int] = None) -> str:
        """Updates the active checkpoint path, computes its SHA-256 hash, and logs registration.

        Args:
            checkpoint_path: Path to the newly saved model checkpoint.
            step: Optional training or evaluation step.

        Returns:
            The calculated SHA-256 hash of the checkpoint.
        """
        self.active_checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = compute_file_sha256(self.active_checkpoint_path)

        # Update manifest on disk
        self._write_manifest()

        # Log checkpoint provenance event
        self.log(
            {
                "checkpoint/path": self.active_checkpoint_path,
                "checkpoint/sha256": self.checkpoint_sha256,
            },
            step=step,
        )
        return self.checkpoint_sha256

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> Dict[str, Any]:
        """Core logging method. Guarantees that EVERY log entry contains run_id and provenance.

        Args:
            metrics: Dictionary of metrics to log.
            step: Integer environment step or optimization epoch.

        Returns:
            The complete logged entry with attached provenance metadata.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Build fully qualified entry with mandatory provenance fields
        entry: Dict[str, Any] = {
            # Mandatory provenance metadata attached to every single entry
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
            "git_commit": self.git_commit_hash,
            "seed": self.seed,
            "timestamp_utc": timestamp,
        }

        if step is not None:
            entry["step"] = int(step)

        # Add metric values
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                entry[k] = v
            elif hasattr(v, "item"):
                entry[k] = v.item()
            else:
                entry[k] = v

        # 1. Forward to WandB if active
        if self.wandb_run is not None:
            try:
                wandb.log(entry, step=step)
            except Exception as e:
                logger.warning(f"wandb.log failed: {e}")

        # 2. Append to local JSONL audit trail
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to local log: {e}")

        # 3. Store in-memory
        self.history.append(entry)
        return entry

    def log_training_metrics(
        self,
        step: int,
        reward: float,
        win_rate: float,
        goal_diff: float,
        entropy: float,
        value_loss: float,
        policy_loss: float,
        kl: float,
        **extra_metrics: Any,
    ) -> Dict[str, Any]:
        """Logs all primary multi-agent RL training metrics.

        Args:
            step: Current environment step or episode.
            reward: Mean episodic or discounted return.
            win_rate: Fraction of games won (0.0 to 1.0).
            goal_diff: Average goal difference per match.
            entropy: Policy entropy for exploration monitoring.
            value_loss: Critic value function loss.
            policy_loss: Actor policy loss.
            kl: Policy Kullback-Leibler divergence (approx_kl).
            **extra_metrics: Additional optional training metrics.
        """
        payload = {
            "train/reward": float(reward),
            "train/win_rate": float(win_rate),
            "train/goal_diff": float(goal_diff),
            "train/entropy": float(entropy),
            "train/value_loss": float(value_loss),
            "train/policy_loss": float(policy_loss),
            "train/kl": float(kl),
        }
        for k, v in extra_metrics.items():
            payload[f"train/{k}"] = v

        return self.log(payload, step=step)

    def log_evaluation_metrics(
        self,
        step: int,
        tpca: float,
        obmq: float,
        pcr: float,
        cohens_kappa: float,
        **extra_metrics: Any,
    ) -> Dict[str, Any]:
        """Logs all specialized domain evaluation metrics for Google Research Football.

        Args:
            step: Current evaluation step.
            tpca: Tactical Phase / Possession Completion Accuracy.
            obmq: Off-Ball Movement Quality.
            pcr: Pass Completion Rate (0.0 to 1.0).
            cohens_kappa: Inter-agent role alignment / decision consistency metric.
            **extra_metrics: Additional evaluation metrics.
        """
        payload = {
            "eval/TPCA": float(tpca),
            "eval/OBMQ": float(obmq),
            "eval/PCR": float(pcr),
            "eval/cohens_kappa": float(cohens_kappa),
        }
        for k, v in extra_metrics.items():
            payload[f"eval/{k}"] = v

        return self.log(payload, step=step)

    def log_system_metrics(
        self,
        step: int,
        fps: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        inference_latency_ms: Optional[float] = None,
        **extra_metrics: Any,
    ) -> Dict[str, Any]:
        """Logs runtime, hardware, and performance metrics.

        Automatically samples CUDA memory allocation if PyTorch with GPU is active.

        Args:
            step: Current execution step.
            fps: Simulation / rollout frames per second.
            gpu_memory_mb: GPU VRAM currently allocated in MB.
            inference_latency_ms: Mean neural network forward pass latency in milliseconds.
            **extra_metrics: Additional system metrics.
        """
        payload: Dict[str, Any] = {}

        if fps is not None:
            payload["system/fps"] = float(fps)

        if gpu_memory_mb is not None:
            payload["system/gpu_memory_mb"] = float(gpu_memory_mb)
        elif _TORCH_AVAILABLE and torch.cuda.is_available():
            # Automatically query allocated GPU memory
            payload["system/gpu_memory_mb"] = float(torch.cuda.memory_allocated() / (1024 * 1024))
            payload["system/gpu_memory_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024 * 1024))

        if inference_latency_ms is not None:
            payload["system/inference_latency_ms"] = float(inference_latency_ms)

        for k, v in extra_metrics.items():
            payload[f"system/{k}"] = v

        return self.log(payload, step=step)

    def finish(self) -> None:
        """Flushes logs, finalizes manifest, and closes the W&B run."""
        # Update manifest status
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r+", encoding="utf-8") as f:
                    manifest = json.load(f)
                    manifest["status"] = "COMPLETED"
                    manifest["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    manifest["total_logged_entries"] = len(self.history)
                    f.seek(0)
                    json.dump(manifest, f, indent=2)
                    f.truncate()
            except Exception as e:
                logger.error(f"Error finalizing manifest: {e}")

        # Close WandB run
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")
            finally:
                self.wandb_run = None

    def __enter__(self) -> "WandBLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()


# ==============================================================================
# Unit Test Suite for WandBLogger & Provenance Integrity
# ==============================================================================
class TestWandBLogger(unittest.TestCase):
    """Verifies that WandBLogger adheres to strict provenance and logging requirements."""

    def setUp(self) -> None:
        self.test_dir = Path("results/test_logs")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.mock_config = {
            "env": {"name": "11_vs_11_kaggle", "representation": "simple115v2"},
            "train": {"lr": 3e-4, "gamma": 0.993, "batch_size": 256},
        }

    def tearDown(self) -> None:
        # Cleanup test directory files
        if self.test_dir.exists():
            for f in self.test_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def test_every_log_entry_has_run_id_and_provenance(self) -> None:
        """Verifies that EVERY log entry contains run_id, config_hash, checkpoint_sha256, and seed."""
        logger_instance = WandBLogger(
            run_id="test_run_prov_001",
            config=self.mock_config,
            seed=1234,
            output_dir=self.test_dir,
            mode="disabled",
        )

        # 1. Log training metrics
        entry_train = logger_instance.log_training_metrics(
            step=100,
            reward=2.5,
            win_rate=0.75,
            goal_diff=1.2,
            entropy=0.85,
            value_loss=0.012,
            policy_loss=0.045,
            kl=0.003,
        )

        # 2. Log evaluation metrics
        entry_eval = logger_instance.log_evaluation_metrics(
            step=100,
            tpca=0.82,
            obmq=0.74,
            pcr=0.88,
            cohens_kappa=0.67,
        )

        # 3. Log system metrics
        entry_sys = logger_instance.log_system_metrics(
            step=100,
            fps=1250.0,
            gpu_memory_mb=1024.0,
            inference_latency_ms=2.1,
        )

        # Assert specific metric values logged
        self.assertEqual(entry_train["train/reward"], 2.5)
        self.assertEqual(entry_train["train/win_rate"], 0.75)
        self.assertEqual(entry_train["train/goal_diff"], 1.2)
        self.assertEqual(entry_train["train/entropy"], 0.85)
        self.assertEqual(entry_train["train/value_loss"], 0.012)
        self.assertEqual(entry_train["train/policy_loss"], 0.045)
        self.assertEqual(entry_train["train/kl"], 0.003)

        self.assertEqual(entry_eval["eval/TPCA"], 0.82)
        self.assertEqual(entry_eval["eval/OBMQ"], 0.74)
        self.assertEqual(entry_eval["eval/PCR"], 0.88)
        self.assertEqual(entry_eval["eval/cohens_kappa"], 0.67)

        self.assertEqual(entry_sys["system/fps"], 1250.0)
        self.assertEqual(entry_sys["system/gpu_memory_mb"], 1024.0)
        self.assertEqual(entry_sys["system/inference_latency_ms"], 2.1)

        # Assert mandatory provenance invariants on every entry
        for entry in (entry_train, entry_eval, entry_sys):
            self.assertIn("run_id", entry, "run_id must be present in every log entry!")
            self.assertEqual(entry["run_id"], "test_run_prov_001")
            self.assertIn("config_hash", entry, "config_hash must be present in every log entry!")
            self.assertEqual(entry["config_hash"], logger_instance.config_hash)
            self.assertIn("checkpoint_sha256", entry, "checkpoint_sha256 must be present in every log entry!")
            self.assertEqual(entry["checkpoint_sha256"], "none")
            self.assertIn("git_commit", entry, "git_commit must be present in every log entry!")
            self.assertIn("seed", entry, "seed must be present in every log entry!")
            self.assertEqual(entry["seed"], 1234)

        logger_instance.finish()

    def test_checkpoint_sha256_update_and_linking(self) -> None:
        """Verifies checkpoint hashing, SHA-256 updating, and linking to run_id."""
        # Create a mock checkpoint file
        ckpt_path = self.test_dir / "checkpoint_step_500.pt"
        ckpt_bytes = b"PYTORCH_MOCK_CHECKPOINT_WEIGHTS_VERSION_1"
        ckpt_path.write_bytes(ckpt_bytes)
        expected_sha = hashlib.sha256(ckpt_bytes).hexdigest()

        with WandBLogger(
            run_id="test_run_ckpt_002",
            config=self.mock_config,
            seed=42,
            output_dir=self.test_dir,
            mode="disabled",
        ) as log_inst:
            updated_sha = log_inst.update_checkpoint(ckpt_path, step=500)
            self.assertEqual(updated_sha, expected_sha)
            self.assertEqual(log_inst.checkpoint_sha256, expected_sha)

            # Subsequent log entry should now carry the new checkpoint SHA-256
            entry = log_inst.log_training_metrics(
                step=501,
                reward=3.0,
                win_rate=0.80,
                goal_diff=2.0,
                entropy=0.70,
                value_loss=0.01,
                policy_loss=0.03,
                kl=0.002,
            )
            self.assertEqual(entry["checkpoint_sha256"], expected_sha)
            self.assertEqual(entry["run_id"], "test_run_ckpt_002")

    def test_config_hash_determinism(self) -> None:
        """Verifies that config SHA-256 hashing is stable and canonical."""
        cfg_a = {"lr": 1e-4, "gamma": 0.99, "batch_size": 64}
        cfg_b = {"batch_size": 64, "lr": 1e-4, "gamma": 0.99}  # Permuted key order
        cfg_c = {"lr": 2e-4, "gamma": 0.99, "batch_size": 64}  # Different value

        hash_a = compute_config_hash(cfg_a)
        hash_b = compute_config_hash(cfg_b)
        hash_c = compute_config_hash(cfg_c)

        self.assertEqual(hash_a, hash_b, "Hash should be invariant to dictionary key ordering!")
        self.assertNotEqual(hash_a, hash_c, "Distinct configurations must yield distinct hashes!")


if __name__ == "__main__":
    print("[TEST] Running WandBLogger test suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWandBLogger)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("[SUCCESS] All WandBLogger tests passed successfully.")
    else:
        sys.exit(1)
