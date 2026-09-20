"""Spatial coordinate transformation and graph embedding utilities using einops."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    import einops
except ImportError:
    einops = None
from src.utils.registry import FEATURE_REGISTRY


@FEATURE_REGISTRY.register("spatial_graph_transformer")
class SpatialGraphTransformer(nn.Module):
    """Computes pairwise spatial displacement matrices across multi-agent sets.

    Uses einops to compute translation-invariant relative coordinate embeddings
    between all controlled agents and opposing defenders.

    Input shape: (batch_size, num_agents, 2)
    Output shape: (batch_size, num_agents, num_agents, 2)
    """

    def __init__(self, embed_dim: int = 32) -> None:
        """Initializes the spatial embedding projection.

        Args:
            embed_dim: Dimension to project relative coordinate vectors into.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def compute_relative_displacements(self, positions: torch.Tensor) -> torch.Tensor:
        """Computes pairwise relative displacement vectors between agents.

        Args:
            positions: Tensor of shape (batch, num_agents, 2)

        Returns:
            Displacements tensor of shape (batch, num_agents, num_agents, 2)
            where entry [b, i, j] = positions[b, j] - positions[b, i]
        """
        if einops is not None:
            pos_i = einops.repeat(positions, "b n d -> b n m d", m=positions.shape[1])
            pos_j = einops.repeat(positions, "b m d -> b n m d", n=positions.shape[1])
            return pos_j - pos_i
        else:
            # Native PyTorch broadcasting fallback
            # positions: (B, N, 2) -> (B, 1, N, 2) - (B, N, 1, 2) = (B, N, N, 2)
            return positions.unsqueeze(1) - positions.unsqueeze(2)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Projects pairwise relative displacement matrices into feature space.

        Args:
            positions: Tensor of shape (batch, num_agents, 2)

        Returns:
            Embedded relational features of shape (batch, num_agents, num_agents, embed_dim)
        """
        disp = self.compute_relative_displacements(positions)
        return self.proj(disp)
