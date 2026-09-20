"""Comprehensive test suite for the SemanticFeaturePipeline.

Validates:
1. Shape contract: [B, 10, 115] -> [B, 10, 139] for various batch sizes.
2. Differentiability: torch.autograd.gradcheck passes on double precision inputs.
3. No hard-coded coordinates: source code contains no np.array coordinate literals.
4. Non-constant variation: all 24 added dimensions have variance > 1e-4 across 100 random GRF states.
5. Parameter learnability: kappa1, kappa2, kappa3, sigma0, lambda_ttr are nn.Parameter objects and receive gradients.
"""

import os
import re
import unittest
import torch
import torch.nn as nn
from src.features.semantic import SemanticFeaturePipeline


class TestSemanticFeaturePipeline(unittest.TestCase):
    """Test suite covering all manuscript requirements and acceptance criteria."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.pipeline = SemanticFeaturePipeline()

    def test_output_shape(self) -> None:
        """Requirement 1 & 2: Output shape must be [B, 10, 139]."""
        for B in [1, 2, 8]:
            raw_obs = torch.randn(B, 10, 115)
            # Ensure valid pitch coordinate ranges roughly [-1, 1]
            raw_obs = torch.clamp(raw_obs, -1.0, 1.0)
            augmented = self.pipeline(raw_obs)
            self.assertEqual(
                augmented.shape,
                (B, 10, 139),
                f"Expected shape ({B}, 10, 139), got {augmented.shape}"
            )
            # Check raw 115 dimensions are preserved exactly
            self.assertTrue(torch.allclose(augmented[..., :115], raw_obs, atol=1e-6))

    def test_gradcheck(self) -> None:
        """Requirement 4: Gradient check passes with torch.autograd.gradcheck."""
        pipeline_double = SemanticFeaturePipeline().double()
        B, N, D = 1, 10, 115
        # Set distinct non-zero coordinates to avoid exact singular configurations
        raw_obs = torch.randn(B, N, D, dtype=torch.float64) * 0.5
        raw_obs.requires_grad_(True)

        def func(inputs: torch.Tensor) -> torch.Tensor:
            return pipeline_double(inputs)

        # fast_mode=True tests VJP and JVP analytically
        check_passed = torch.autograd.gradcheck(
            func,
            (raw_obs,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
            fast_mode=True
        )
        self.assertTrue(check_passed, "Gradcheck failed for SemanticFeaturePipeline.")

    def test_parameter_learnability(self) -> None:
        """Requirement 5: Parameters must be nn.Parameter and receive gradients."""
        expected_params = ["kappa1", "kappa2", "kappa3", "sigma0", "lambda_ttr"]
        for p_name in expected_params:
            param = getattr(self.pipeline, p_name, None)
            self.assertIsNotNone(param, f"Missing parameter {p_name}")
            self.assertIsInstance(
                param,
                nn.Parameter,
                f"Parameter {p_name} is not an instance of nn.Parameter"
            )

        raw_obs = torch.randn(2, 10, 115, requires_grad=True)
        augmented = self.pipeline(raw_obs)
        loss = augmented.sum()
        loss.backward()

        for p_name in expected_params:
            param = getattr(self.pipeline, p_name)
            self.assertIsNotNone(param.grad, f"Gradient for {p_name} is None after backward()")
            self.assertFalse(torch.isnan(param.grad), f"Gradient for {p_name} is NaN")
            self.assertNotEqual(param.grad.item(), 0.0, f"Gradient for {p_name} is zero")

    def test_no_hardcoded_coordinates(self) -> None:
        """Requirement 6: Source file must contain no literal coordinate arrays like np.array([...])."""
        semantic_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "features", "semantic.py")
        )
        self.assertTrue(os.path.exists(semantic_path), f"File not found: {semantic_path}")

        with open(semantic_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for np.array pattern
        np_matches = re.findall(r"np\.array\s*\(", content)
        self.assertEqual(len(np_matches), 0, f"Found literal np.array calls: {np_matches}")

        # Check numpy is not even imported
        self.assertNotIn("import numpy", content)
        self.assertNotIn("from numpy", content)

    def test_all_24_dims_non_constant(self) -> None:
        """Acceptance Criteria: All 24 added dimensions must vary across 100 random states (var > 1e-4)."""
        B = 100
        # Generate 100 diverse GRF states
        torch.manual_seed(123)
        raw_states = torch.randn(B, 10, 115) * 0.4
        augmented = self.pipeline(raw_states)
        added_24 = augmented[..., 115:]  # (100, 10, 24)

        # Compute variance across batch B for each of the 24 dimensions across states
        dim_variances = added_24.var(dim=(0, 1))  # (24,)
        min_dim_variance = torch.min(dim_variances).item()
        print(f"\n[Test Info] Minimum dimension variance across 100 states: {min_dim_variance:.6e}")
        self.assertGreater(min_dim_variance, 1e-4, f"Min dimension variance {min_dim_variance} <= 1e-4")

        # Also verify per-agent variance
        agent_variances = torch.var(added_24, dim=0)  # (10, 24)
        for d in range(24):
            avg_agent_var = torch.mean(agent_variances[:, d]).item()
            self.assertGreater(
                avg_agent_var,
                1e-4,
                f"Dimension {d} average agent variance is insufficient: {avg_agent_var:.6e} <= 1e-4"
            )


if __name__ == "__main__":
    unittest.main()
