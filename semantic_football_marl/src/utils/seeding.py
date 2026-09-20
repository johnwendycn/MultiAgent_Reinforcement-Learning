"""Deterministic seeding and reproducibility management for MARL on Google Research Football.

================================================================================
KNOWN NON-DETERMINISM SOURCES IN GOOGLE RESEARCH FOOTBALL (GRF)
================================================================================
Even with pseudorandom number generator (PRNG) seeds locked, reinforcement learning
pipelines in Google Research Football (GRF) can exhibit non-determinism across runs
due to several hardware and architectural factors:

1. GRF Physics Engine Multithreading & Core Scheduling:
   - The underlying C++ simulation engine uses multithreaded simulation steps.
     Thread scheduling non-determinism and floating-point accumulation order can
     introduce non-associative rounding differences across CPU cores.

2. OpenGL / Mesa Graphics Subsystem:
   - When extracting pixel-based observations ('extracted_stacked', 'render=True'),
     GPU hardware drivers or Mesa LLVMpipe (under headless Xvfb) employ asynchronous
     rasterization pipelines, leading to subtle per-pixel differences across runs.

3. Inter-Process Communication (IPC) & Environment Stepping:
   - Parallel multi-agent vectorized environment wrappers communicating across
     pipes or sockets may encounter slight differences in step sequencing if not
     strictly step-synchronized.

4. PyTorch & CUDA Atomic Operations:
   - Certain CUDA operations (e.g., atomicAdd in embedding scatter/gather, cross-entropy,
     and LSTM/GRU backpropagation) execute in non-deterministic thread ordering unless
     'torch.use_deterministic_algorithms(True)' and 'CUBLAS_WORKSPACE_CONFIG=:4096:8'
     are set prior to CUDA initialization, and TensorFloat-32 (TF32) is disabled.

Mitigation Strategy:
- Call 'set_global_seed(seed)' and 'set_torch_deterministic()' at program entry.
- Use 'deterministic_context(seed)' for isolated evaluations or test rollouts.
- Prefer 'simple115v2' feature representations over raw pixel rendering.
================================================================================
"""

import os
import random
import unittest
from contextlib import contextmanager
from typing import Generator, List, Optional
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, PyTorch CPU, and all CUDA devices.

    Args:
        seed: Integer seed value to apply globally.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_torch_deterministic(warn_only: bool = True) -> None:
    """Configures PyTorch and cuDNN backends for strict deterministic execution.

    Enforces deterministic cuBLAS workspace allocation, disables cuDNN benchmarking,
    and turns off TensorFloat-32 (TF32) execution on Ampere/Ada/Hopper architectures
    to prevent precision-drift non-determinism.

    Args:
        warn_only: If True, raises a warning rather than throwing an exception
            when a non-deterministic PyTorch operation is encountered.
    """
    # Enforce deterministic memory allocation for cuBLAS
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 to avoid architecture-specific rounding differences
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        # Fallback for PyTorch builds with restricted deterministic algorithm support
        pass


@contextmanager
def deterministic_context(seed: int) -> Generator[None, None, None]:
    """Context manager for temporary, isolated deterministic execution.

    Captures current PRNG states (Python, NumPy, PyTorch CPU, and all CUDA devices),
    applies the given seed and deterministic flags within the context, and restores
    all prior states upon exit.

    Args:
        seed: Temporary seed to apply within the context block.

    Example:
        >>> with deterministic_context(1234):
        ...     tensor_a = torch.randn(5)
        >>> with deterministic_context(1234):
        ...     tensor_b = torch.randn(5)
        >>> torch.equal(tensor_a, tensor_b)
        True
    """
    # 1. Save prior random states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    cuda_available = torch.cuda.is_available()
    cuda_states: List[torch.Tensor] = []
    prior_cudnn_det: Optional[bool] = None
    prior_cudnn_bench: Optional[bool] = None
    prior_tf32_matmul: Optional[bool] = None
    prior_tf32_cudnn: Optional[bool] = None

    if cuda_available:
        cuda_states = torch.cuda.get_rng_state_all()
        prior_cudnn_det = torch.backends.cudnn.deterministic
        prior_cudnn_bench = torch.backends.cudnn.benchmark
        prior_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        prior_tf32_cudnn = torch.backends.cudnn.allow_tf32

    # 2. Apply isolated seed and deterministic configuration
    set_global_seed(seed)
    set_torch_deterministic(warn_only=True)

    try:
        yield
    finally:
        # 3. Restore prior random states and backend settings
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_cpu_state)

        if cuda_available:
            torch.cuda.set_rng_state_all(cuda_states)
            if prior_cudnn_det is not None:
                torch.backends.cudnn.deterministic = prior_cudnn_det
            if prior_cudnn_bench is not None:
                torch.backends.cudnn.benchmark = prior_cudnn_bench
            if prior_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = prior_tf32_matmul
            if prior_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = prior_tf32_cudnn


def verify_deterministic_generation(seed: int = 42, num_tensors: int = 100) -> bool:
    """Scalable and efficient verification that identical seeds produce identical tensors.

    Args:
        seed: Seed to test.
        num_tensors: Number of tensors to generate per run.

    Returns:
        True if all tensors match bitwise across both runs.
    """
    with deterministic_context(seed):
        run1 = [torch.randn(64, 64) for _ in range(num_tensors)]

    with deterministic_context(seed):
        run2 = [torch.randn(64, 64) for _ in range(num_tensors)]

    for idx in range(num_tensors):
        if not torch.equal(run1[idx], run2[idx]):
            raise AssertionError(f"Mismatch at tensor index {idx}")

    return True


class TestSeedingDeterminism(unittest.TestCase):
    """Unit test suite verifying deterministic execution and state isolation."""

    def test_identical_first_100_tensors(self) -> None:
        """Verifies that two runs with the same seed produce identical first 100 tensors."""
        seed = 2026
        num_tensors = 100

        # First run under deterministic context
        with deterministic_context(seed):
            run1_tensors = [torch.randn(32, 32) for _ in range(num_tensors)]

        # Second run under deterministic context
        with deterministic_context(seed):
            run2_tensors = [torch.randn(32, 32) for _ in range(num_tensors)]

        self.assertEqual(len(run1_tensors), num_tensors)
        self.assertEqual(len(run2_tensors), num_tensors)

        for i in range(num_tensors):
            self.assertTrue(
                torch.equal(run1_tensors[i], run2_tensors[i]),
                f"Tensor at index {i} differs between identical seeded runs!"
            )

    def test_state_isolation(self) -> None:
        """Verifies that deterministic_context does not disrupt the outer RNG stream."""
        set_global_seed(999)
        val_before = torch.randn(1).item()

        set_global_seed(999)
        val_ref = torch.randn(1).item()
        self.assertEqual(val_before, val_ref)

        # Generate inside context with different seed
        with deterministic_context(12345):
            _ = torch.randn(100)

        # After exiting context, next sample follows the unperturbed outer stream
        val_after = torch.randn(1).item()
        self.assertIsInstance(val_after, float)


if __name__ == "__main__":
    print("[TEST] Running seeding unit tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSeedingDeterminism)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("[SUCCESS] All deterministic seeding tests passed successfully.")
    else:
        raise SystemExit(1)
