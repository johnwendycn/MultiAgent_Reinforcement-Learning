"""Unit tests for semantic feature extraction and spatial representations."""

import numpy as np
import torch
from src.features.semantic_extractor import SemanticFeatureExtractor
from src.features.spatial_graph import SpatialGraphTransformer


def test_semantic_feature_extractor() -> None:
    """Verifies feature extraction output keys, types, and value bounds."""
    extractor = SemanticFeatureExtractor()

    # Synthetic simple115 vector
    obs = np.zeros(115, dtype=np.float32)
    # Put ball at (0.8, 0.0) -> attacking third, close to goal
    obs[88] = 0.8
    obs[89] = 0.0
    # Left team owns ball
    obs[95] = 1.0

    features = extractor.extract_features(obs)

    expected_keys = [
        "ball_possession_team",
        "dist_to_goal",
        "angle_to_goal",
        "min_opponent_dist",
        "pressing_intensity",
        "min_teammate_dist",
        "xg_proxy",
        "pitch_zone",
        "pitch_zone_one_hot",
        "corridor_clearance",
    ]
    for key in expected_keys:
        assert key in features, f"Missing key: {key}"

    assert 0.0 <= features["xg_proxy"] <= 1.0
    assert features["pitch_zone"] in ["defensive_third", "middle_third", "attacking_third"]
    assert len(features["pitch_zone_one_hot"]) == 3


def test_spatial_graph_transformer() -> None:
    """Verifies relative coordinate computation and relational feature projection."""
    transformer = SpatialGraphTransformer(embed_dim=16)

    batch_size = 4
    num_agents = 3
    coords = torch.randn(batch_size, num_agents, 2)

    embeddings = transformer(coords)
    assert embeddings.shape == (batch_size, num_agents, num_agents, 16)

    # Relative displacement from agent i to itself must be zero
    disp = transformer.compute_relative_displacements(coords)
    for b in range(batch_size):
        for i in range(num_agents):
            assert torch.allclose(disp[b, i, i], torch.zeros(2), atol=1e-6)
