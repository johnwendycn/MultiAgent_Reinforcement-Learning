"""Unit tests for deterministic seeding utilities."""

import torch
from src.utils.seeding import (
    set_global_seed,
    set_torch_deterministic,
    deterministic_context,
    verify_deterministic_generation,
)


def test_first_100_tensors_identical_with_same_seed() -> None:
    """Unit test verifying that two runs with the same seed produce identical first 100 tensors."""
    assert verify_deterministic_generation(seed=42, num_tensors=100) is True


def test_deterministic_context_isolation() -> None:
    """Verifies that deterministic_context restores prior state and does not leak."""
    set_global_seed(999)
    baseline_val = torch.randn(1).item()

    set_global_seed(999)
    with deterministic_context(123):
        inner_val = torch.randn(1).item()

    # After exiting context, RNG state should continue from seed 999
    post_val = torch.randn(1).item()
    assert baseline_val == post_val
    assert inner_val != baseline_val
