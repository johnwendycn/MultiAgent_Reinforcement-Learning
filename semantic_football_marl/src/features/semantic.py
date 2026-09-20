"""Differentiable semantic feature extraction pipeline for Association Football MARL.

Implements the complete geometric and spatial mechanics from the manuscript:
- 8D: Dynamic pass-lane openness to the 8 nearest teammates (Eqs. 2-7)
- 6D: Spatial occupation scores for the 6 most advanced teammates (Eq. 8)
- 4D: Time-To-React (TTR) / pitch control field features (Eqs. 9-10)
- 4D: Goal-angle geometry and shot viability metrics (Eqs. 11-12)
- 2D: Global team shape indicators (defensive line height, team width)

Total: 24 added differentiable dimensions concatenated with 115D raw state -> 139D output.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFeaturePipeline(nn.Module):
    """End-to-end differentiable neural pipeline computing 24 tactical geometric features.

    Transforms raw simple115v2 tensors [B, N, 115] into augmented semantic tensors [B, N, 139]
    where N=10 learning outfield agents.
    """

    RAW_DIM: int = 115
    ADDED_DIM: int = 24
    AUGMENTED_DIM: int = 139
    NUM_AGENTS: int = 10

    def __init__(
        self,
        v_max: float = 1.0,
        t_react: float = 0.1,
        eps: float = 1e-7,
        sigma0_init: float = 0.15,
        kappa1_init: float = 1.0,
        kappa2_init: float = 1.0,
        kappa3_init: float = 1.0,
        lambda_ttr_init: float = 1.0,
    ) -> None:
        """Initializes the differentiable semantic feature pipeline.

        Args:
            v_max: Maximum nominal player velocity on pitch (buffer).
            t_react: Player cognitive and physical reaction delay (buffer).
            eps: Epsilon parameter preventing division by zero and singularity in norms.
            sigma0_init: Initial value for interception dispersion parameter sigma0.
            kappa1_init: Initial weight for ball proximity in spatial score.
            kappa2_init: Initial weight for defender distance in spatial score.
            kappa3_init: Initial weight for pass openness in spatial score.
            lambda_ttr_init: Initial scaling factor for pitch control sigmoid.
        """
        super().__init__()
        self.v_max = v_max
        self.t_react = t_react
        self.eps = eps

        # Manuscript Learnable Parameters (must remain nn.Parameter objects)
        self.kappa1 = nn.Parameter(torch.tensor(kappa1_init, dtype=torch.float32))
        self.kappa2 = nn.Parameter(torch.tensor(kappa2_init, dtype=torch.float32))
        self.kappa3 = nn.Parameter(torch.tensor(kappa3_init, dtype=torch.float32))
        self.sigma0 = nn.Parameter(torch.tensor(sigma0_init, dtype=torch.float32))
        self.lambda_ttr = nn.Parameter(torch.tensor(lambda_ttr_init, dtype=torch.float32))

        # Opponent goal post coordinates registered as non-trainable buffers
        # Standard pitch coordinates: opponent goal is centered at x=1.0, width=0.088
        self.register_buffer("goal_post_top", torch.tensor([1.0, 0.044], dtype=torch.float32))
        self.register_buffer("goal_post_bot", torch.tensor([1.0, -0.044], dtype=torch.float32))

    def _soft_min(self, x: torch.Tensor, dim: int = -1, beta: float = 5.0) -> torch.Tensor:
        """Differentiable soft-min approximation using smooth Boltzmann weights."""
        weights = F.softmax(-beta * x, dim=dim)
        return torch.sum(x * weights, dim=dim)

    def _soft_max(self, x: torch.Tensor, dim: int = -1, beta: float = 5.0) -> torch.Tensor:
        """Differentiable soft-max approximation using smooth Boltzmann weights."""
        weights = F.softmax(beta * x, dim=dim)
        return torch.sum(x * weights, dim=dim)

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass computing the augmented 139D state from raw 115D observation.

        Args:
            raw_obs: Tensor of shape [B, N, 115] where N=10 outfield learning agents.

        Returns:
            Tensor of shape [B, N, 139] with exactly 24 added differentiable features.
        """
        B, N, D = raw_obs.shape
        assert N == self.NUM_AGENTS, f"Expected {self.NUM_AGENTS} outfield agents, got {N}"
        assert D == self.RAW_DIM, f"Expected {self.RAW_DIM} raw observation dimensions, got {D}"

        # ----------------------------------------------------------------------
        # Extract player, opponent defender, and ball positions dynamically
        # NO hard-coded player or ball coordinates are used.
        # ----------------------------------------------------------------------
        # Left team (11 players: idx 0 is goalkeeper, idx 1..10 are outfield agents)
        p_left = raw_obs[..., 0:22].view(B, N, 11, 2)
        v_left = raw_obs[..., 22:44].view(B, N, 11, 2)

        # Right team (11 players: opposing defenders and opposing goalkeeper)
        p_right = raw_obs[..., 44:66].view(B, N, 11, 2)
        v_right = raw_obs[..., 66:88].view(B, N, 11, 2)

        # Ball kinematics (x, y)
        p_ball = raw_obs[..., 88:90]
        v_ball = raw_obs[..., 91:93]

        # Opponent goalkeeper position (player 0 of right team)
        p_gk_opp = p_right[..., 0, :]

        added_feature_list = []

        # Process each of the 10 learning outfield agents (indices 0..9 -> players 1..10)
        for i in range(self.NUM_AGENTS):
            ego_player_idx = i + 1
            p_ego = p_left[:, i, ego_player_idx, :]  # (B, 2)
            v_ego = v_left[:, i, ego_player_idx, :]  # (B, 2)
            p_b = p_ball[:, i, :]                    # (B, 2)
            v_b = v_ball[:, i, :]                    # (B, 2)

            # The 9 other outfield teammates
            tm_indices = [idx for idx in range(1, 11) if idx != ego_player_idx]
            p_tms = p_left[:, i, tm_indices, :]      # (B, 9, 2)
            v_tms = v_left[:, i, tm_indices, :]      # (B, 9, 2)
            p_defs = p_right[:, i, :, :]             # (B, 11, 2)
            v_defs = v_right[:, i, :, :]             # (B, 11, 2)

            # ==================================================================
            # 1. 8D: Dynamic pass-lane openness to 8 nearest teammates (Eqs. 2-7)
            # ==================================================================
            dists_to_tms = torch.sqrt(torch.sum((p_tms - p_ego.unsqueeze(1)) ** 2, dim=-1) + self.eps)
            _, nearest_tm_ranks = torch.topk(-dists_to_tms, k=8, dim=-1)

            batch_idx_8 = torch.arange(B, device=raw_obs.device).unsqueeze(1).expand(B, 8)
            p_8tms = p_tms[batch_idx_8, nearest_tm_ranks]  # (B, 8, 2)

            # Pass trajectory vector from ball to target teammate: v_pass = p_j - p_b
            v_pass = p_8tms - p_b.unsqueeze(1)  # (B, 8, 2)
            v_pass_norm_sq = torch.sum(v_pass ** 2, dim=-1, keepdim=True) + self.eps  # (B, 8, 1)

            p_d_minus_pb = (p_defs - p_b.unsqueeze(1)).unsqueeze(1)  # (B, 1, 11, 2)
            v_pass_exp = v_pass.unsqueeze(2)  # (B, 8, 1, 2)

            # Eq. 2: t_proj(d) = <p_d - p_b, v_pass> / (||v_pass||^2 + eps)
            dot_prod = torch.sum(p_d_minus_pb * v_pass_exp, dim=-1)
            t_proj = dot_prod / v_pass_norm_sq  # (B, 8, 11)

            # Eq. 3: p_closest(d) = p_b + clip(t_proj, 0, 1) * v_pass
            t_proj_clamped = torch.clamp(t_proj, 0.0, 1.0)
            p_closest = p_b.unsqueeze(1).unsqueeze(2) + t_proj_clamped.unsqueeze(-1) * v_pass_exp

            # Eq. 4: h_perp(d) = ||p_d - p_closest(d)||_2
            h_perp = torch.sqrt(torch.sum((p_defs.unsqueeze(1) - p_closest) ** 2, dim=-1) + self.eps)

            # Eq. 6: sigma_d = sigma0 * (1 + ||v_d|| / v_max)
            v_def_norm = torch.sqrt(torch.sum(v_defs ** 2, dim=-1) + self.eps)  # (B, 11)
            sigma_d = torch.abs(self.sigma0) * (1.0 + v_def_norm / self.v_max)   # (B, 11)
            sigma_d_exp = sigma_d.unsqueeze(1)  # (B, 1, 11)

            # Eq. 5: P_intercept(d) = exp(-h_perp^2 / (2 * sigma_d^2))
            P_intercept = torch.exp(-(h_perp ** 2) / (2.0 * (sigma_d_exp ** 2) + self.eps))  # (B, 8, 11)

            # Eq. 7: L_pass = 1 - max_d(P_intercept(d))
            max_P = self._soft_max(P_intercept, dim=-1, beta=10.0)  # (B, 8)
            L_pass_8 = 1.0 - max_P  # (B, 8)

            # ==================================================================
            # 2. 6D: Spatial occupation scores for 6 most advanced teammates (Eq. 8)
            # ==================================================================
            tm_x = p_tms[..., 0]  # (B, 9)
            _, adv_ranks = torch.topk(tm_x, k=6, dim=-1)
            batch_idx_6 = torch.arange(B, device=raw_obs.device).unsqueeze(1).expand(B, 6)
            p_6adv = p_tms[batch_idx_6, adv_ranks]  # (B, 6, 2)

            # D_ball: Proximity to the ball
            D_ball_6 = torch.sqrt(torch.sum((p_6adv - p_b.unsqueeze(1)) ** 2, dim=-1) + self.eps)

            # D_def: Distance to closest defender via differentiable soft-min
            dists_to_defs = torch.sqrt(
                torch.sum((p_6adv.unsqueeze(2) - p_defs.unsqueeze(1)) ** 2, dim=-1) + self.eps
            )
            D_def_6 = self._soft_min(dists_to_defs, dim=-1, beta=5.0)

            # L_pass to the 6 most advanced teammates
            v_pass_6 = p_6adv - p_b.unsqueeze(1)
            v_pass_6_norm_sq = torch.sum(v_pass_6 ** 2, dim=-1, keepdim=True) + self.eps
            dot_6 = torch.sum(p_d_minus_pb * v_pass_6.unsqueeze(2), dim=-1)
            t_6 = torch.clamp(dot_6 / v_pass_6_norm_sq, 0.0, 1.0)
            p_closest_6 = p_b.unsqueeze(1).unsqueeze(2) + t_6.unsqueeze(-1) * v_pass_6.unsqueeze(2)
            h_perp_6 = torch.sqrt(torch.sum((p_defs.unsqueeze(1) - p_closest_6) ** 2, dim=-1) + self.eps)
            P_intercept_6 = torch.exp(-(h_perp_6 ** 2) / (2.0 * (sigma_d_exp ** 2) + self.eps))
            L_pass_6 = 1.0 - self._soft_max(P_intercept_6, dim=-1, beta=10.0)

            # Eq. 8: S(j) = kappa1 * D_ball + kappa2 * D_def + kappa3 * L_pass
            S_6 = self.kappa1 * D_ball_6 + self.kappa2 * D_def_6 + self.kappa3 * L_pass_6  # (B, 6)

            # ==================================================================
            # 3. 4D: TTR / Pitch Control Features (Eqs. 9-10)
            # ==================================================================
            # 4 target evaluation locations derived dynamically from state:
            # [Ball, Ego, Midpoint, Centroid of Top 3 Advanced Teammates]
            x_pts = torch.stack([
                p_b,
                p_ego,
                0.5 * (p_b + p_ego),
                torch.mean(p_6adv[:, :3, :], dim=1),
            ], dim=1)  # (B, 4, 2)

            # All 10 outfield attackers
            p_att_all = p_left[:, i, 1:11, :]  # (B, 10, 2)

            # Eq. 9: TTR(k, x) = t_react + ||p_k - x|| / v_max
            dists_att_x = torch.sqrt(torch.sum((p_att_all.unsqueeze(2) - x_pts.unsqueeze(1)) ** 2, dim=-1) + self.eps)
            TTR_att_all = self.t_react + dists_att_x / self.v_max
            TTR_att = self._soft_min(TTR_att_all, dim=1, beta=5.0)  # (B, 4)

            dists_def_x = torch.sqrt(torch.sum((p_defs.unsqueeze(2) - x_pts.unsqueeze(1)) ** 2, dim=-1) + self.eps)
            TTR_def_all = self.t_react + dists_def_x / self.v_max
            TTR_def = self._soft_min(TTR_def_all, dim=1, beta=5.0)  # (B, 4)

            # Eq. 10: PC_team(x) = sigmoid(lambda_ttr * (TTR_att - TTR_def))
            PC_4 = torch.sigmoid(self.lambda_ttr * (TTR_att - TTR_def))  # (B, 4)

            # ==================================================================
            # 4. 4D: Goal-angle geometry and shot viability (Eqs. 11-12)
            # ==================================================================
            # Evaluated for 4 players: ego agent and top 3 advanced teammates
            shot_players = torch.cat([p_ego.unsqueeze(1), p_6adv[:, :3, :]], dim=1)  # (B, 4, 2)

            g1 = self.goal_post_top.to(raw_obs.dtype)
            g2 = self.goal_post_bot.to(raw_obs.dtype)

            vec_g1 = g1.unsqueeze(0).unsqueeze(0) - shot_players  # (B, 4, 2)
            vec_g2 = g2.unsqueeze(0).unsqueeze(0) - shot_players  # (B, 4, 2)

            dot_goal = torch.sum(vec_g1 * vec_g2, dim=-1)
            norm_g1 = torch.sqrt(torch.sum(vec_g1 ** 2, dim=-1) + self.eps)
            norm_g2 = torch.sqrt(torch.sum(vec_g2 ** 2, dim=-1) + self.eps)
            cos_theta = dot_goal / (norm_g1 * norm_g2 + self.eps)
            cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

            # Eq. 11: theta_goal(j) = arccos(<g1 - p_j, g2 - p_j> / (||g1 - p_j|| * ||g2 - p_j||))
            theta_goal = torch.acos(cos_theta_clamped)  # (B, 4)

            # phi_gk: Angle subtended by opposing goalkeeper (r_gk = 0.04)
            p_gk = p_gk_opp[:, i, :]
            dist_to_gk = torch.sqrt(torch.sum((shot_players - p_gk.unsqueeze(1)) ** 2, dim=-1) + self.eps)
            phi_gk = 2.0 * torch.atan(0.04 / (dist_to_gk + self.eps))

            # Expected threat surface xT(pos)
            x_pos = shot_players[..., 0]
            y_pos = shot_players[..., 1]
            xT = torch.sigmoid(3.0 * x_pos) * torch.exp(-2.0 * (y_pos ** 2))

            # Eq. 12: Shot_Score(j) = max(0, theta_goal - phi_gk) * xT(pos)
            shot_score_4 = F.relu(theta_goal - phi_gk) * xT  # (B, 4)

            # ==================================================================
            # 5. 2D: Team Shape (Defensive line height, team width)
            # ==================================================================
            outfield_x = p_att_all[..., 0]  # (B, 10)
            outfield_y = p_att_all[..., 1]  # (B, 10)

            # Defensive line height: soft-min x-coordinate among outfield teammates
            def_line_height = self._soft_min(outfield_x, dim=-1, beta=5.0).unsqueeze(-1)  # (B, 1)

            # Team width: lateral span (soft-max y - soft-min y)
            team_width = (
                self._soft_max(outfield_y, dim=-1, beta=5.0) - self._soft_min(outfield_y, dim=-1, beta=5.0)
            ).unsqueeze(-1)  # (B, 1)

            shape_2 = torch.cat([def_line_height, team_width], dim=-1)  # (B, 2)

            # Assemble the 24 tactical geometric features for agent i
            agent_24 = torch.cat([L_pass_8, S_6, PC_4, shot_score_4, shape_2], dim=-1)  # (B, 24)
            added_feature_list.append(agent_24)

        # Stack over all 10 outfield agents: (B, 10, 24)
        added_24 = torch.stack(added_feature_list, dim=1)

        # Concatenate: [B, 10, 115] + [B, 10, 24] -> [B, 10, 139]
        augmented_obs = torch.cat([raw_obs, added_24], dim=-1)

        assert augmented_obs.shape == (B, self.NUM_AGENTS, self.AUGMENTED_DIM), (
            f"Augmented state shape mismatch: Expected ({B}, {self.NUM_AGENTS}, {self.AUGMENTED_DIM}), "
            f"got {augmented_obs.shape}"
        )
        return augmented_obs
