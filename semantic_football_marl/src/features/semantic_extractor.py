"""Semantic feature extraction for Google Research Football representations.

Transforms raw 115-dimensional simple115v2 state vectors into tactical domain features
and augmented 139-dimensional semantic observation vectors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from src.utils.registry import FEATURE_REGISTRY
except (ImportError, ValueError):
    try:
        from ..utils.registry import FEATURE_REGISTRY
    except Exception:
        class _DummyRegistry:
            def register(self, name: str):
                return lambda cls: cls
        FEATURE_REGISTRY = _DummyRegistry()


@FEATURE_REGISTRY.register("semantic_extractor")
class SemanticFeatureExtractor:
    """Extracts tactical and domain-specific semantic features from simple115v2 vectors.

    Produces a 24-dimensional semantic tactical vector which, when concatenated with
    the 115D raw simple115v2 representation, yields the full 139D semantic observation:
    
    115D (Raw simple115v2) + 24D (Tactical Semantics) = 139D (Augmented Semantic Obs)

    The 24 semantic tactical dimensions are:
    [0]  dist_to_goal: Euclidean distance from agent to opponent goal center (1.0, 0.0)
    [1]  angle_to_goal: Angle in radians to opponent goal
    [2]  goal_dir_x: Normalized direction vector x to goal
    [3]  goal_dir_y: Normalized direction vector y to goal
    [4]  min_opponent_dist: Proximity of the closest opposing defender
    [5]  pressing_intensity: Normalized opponent pressure intensity in [0, 1]
    [6]  nearest_opp_rel_x: Relative x displacement to nearest opponent
    [7]  nearest_opp_rel_y: Relative y displacement to nearest opponent
    [8]  min_teammate_dist: Proximity of the closest outfield teammate
    [9]  nearest_tm_rel_x: Relative x displacement to nearest teammate
    [10] nearest_tm_rel_y: Relative y displacement to nearest teammate
    [11] xg_proxy: Exponential decay shot quality / expected goals score in [0, 1]
    [12] pitch_zone_defensive: One-hot indicator for defensive third (x < -0.33)
    [13] pitch_zone_middle: One-hot indicator for middle third (-0.33 <= x <= 0.33)
    [14] pitch_zone_attacking: One-hot indicator for attacking third (x > 0.33)
    [15] corridor_clearance: Clearance distance along passing lane to open teammate
    [16] ball_dist_to_agent: Euclidean distance from agent to the ball
    [17] ball_rel_x: Relative x displacement to the ball
    [18] ball_rel_y: Relative y displacement to the ball
    [19] ball_rel_vx: Relative velocity vx between agent and ball
    [20] ball_rel_vy: Relative velocity vy between agent and ball
    [21] ball_dist_to_goal: Euclidean distance from ball to opponent goal
    [22] is_agent_closest_to_ball: Binary flag (1.0 if agent is nearest teammate to ball)
    [23] pass_angle_to_open_tm: Angular bearing to the nearest open passing target
    """

    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    SEMANTIC_FEATURE_DIM = 24
    RAW_OBS_DIM = 115
    TOTAL_SEMANTIC_OBS_DIM = 139  # 115 + 24

    def __init__(
        self,
        use_pitch_zones: bool = True,
        use_xg_proxy: bool = True,
        use_passing_corridors: bool = True,
        opponent_influence_radius: float = 0.15,
    ) -> None:
        """Initializes the semantic feature extractor.

        Args:
            use_pitch_zones: Whether to compute one-hot pitch zones.
            use_xg_proxy: Whether to compute shot threat / xG proxy score.
            use_passing_corridors: Whether to estimate passing corridor clearances.
            opponent_influence_radius: Radius in pitch coordinates for pressing calculation.
        """
        self.use_pitch_zones = use_pitch_zones
        self.use_xg_proxy = use_xg_proxy
        self.use_passing_corridors = use_passing_corridors
        self.opponent_influence_radius = opponent_influence_radius

    def compute_semantic_vector(self, obs: np.ndarray, agent_idx: Optional[int] = None) -> np.ndarray:
        """Computes the 24-dimensional tactical semantic feature vector.

        Args:
            obs: 1D array of length >= 115 (GRF simple115 format).
            agent_idx: Optional index of the agent in the team (0..10). If None,
                inferred from active player one-hot encoding or defaults to 0.

        Returns:
            np.ndarray: 1D float32 array of shape (24,).
        """
        if len(obs) < self.RAW_OBS_DIM:
            return self._fallback_semantic_vector()

        # Parse simple115v2 segments
        left_coords = obs[0:22].reshape(11, 2)
        left_vels = obs[22:44].reshape(11, 2)
        right_coords = obs[44:66].reshape(11, 2)
        right_vels = obs[66:88].reshape(11, 2)
        ball_pos = obs[88:91]  # (x, y, z)
        ball_vel = obs[91:94]  # (vx, vy, vz)
        ball_owned_team = np.argmax(obs[94:97])  # 0: None, 1: Left, 2: Right

        # Determine ego position and velocity
        if agent_idx is not None and 0 <= agent_idx < 11:
            ego_idx = agent_idx
        else:
            active_player_one_hot = obs[97:108]
            ego_idx = int(np.argmax(active_player_one_hot)) if np.any(active_player_one_hot > 0.5) else 0

        ego_pos = left_coords[ego_idx]
        ego_vel = left_vels[ego_idx]

        # 1-4. Goal orientation and unit directions
        vec_to_goal = self.GOAL_POS - ego_pos
        dist_to_goal = float(np.linalg.norm(vec_to_goal))
        angle_to_goal = float(np.arctan2(vec_to_goal[1], vec_to_goal[0]))
        goal_dir = vec_to_goal / (dist_to_goal + 1e-6)

        # 5-8. Opponent pressing and relative displacement
        opp_diffs = right_coords - ego_pos
        opp_dists = np.linalg.norm(opp_diffs, axis=1)
        nearest_opp_idx = int(np.argmin(opp_dists)) if len(opp_dists) > 0 else 0
        min_opp_dist = float(opp_dists[nearest_opp_idx]) if len(opp_dists) > 0 else 1.0
        pressing_intensity = float(np.clip(1.0 - (min_opp_dist / self.opponent_influence_radius), 0.0, 1.0))
        nearest_opp_rel = opp_diffs[nearest_opp_idx] if len(opp_diffs) > 0 else np.zeros(2, dtype=np.float32)

        # 9-11. Teammate support and relative displacement
        other_tm_mask = np.arange(11) != ego_idx
        tm_coords = left_coords[other_tm_mask]
        tm_diffs = tm_coords - ego_pos
        tm_dists = np.linalg.norm(tm_diffs, axis=1)
        nearest_tm_idx = int(np.argmin(tm_dists)) if len(tm_dists) > 0 else 0
        min_tm_dist = float(tm_dists[nearest_tm_idx]) if len(tm_dists) > 0 else 1.0
        nearest_tm_rel = tm_diffs[nearest_tm_idx] if len(tm_diffs) > 0 else np.zeros(2, dtype=np.float32)

        # 12. Expected Goals (xG) Proxy
        cos_angle = max(0.0, np.cos(angle_to_goal))
        xg_proxy = float(np.clip(np.exp(-3.5 * dist_to_goal) * cos_angle, 0.0, 1.0))

        # 13-15. Pitch zone one-hot classification
        x_pos = ego_pos[0]
        if x_pos < -0.33:
            zone_one_hot = [1.0, 0.0, 0.0]  # Defensive
        elif x_pos <= 0.33:
            zone_one_hot = [0.0, 1.0, 0.0]  # Middle
        else:
            zone_one_hot = [0.0, 0.0, 1.0]  # Attacking

        # 16. Passing corridor clearance
        corridor_clearance = self._compute_corridor_clearance(ego_pos, tm_coords, right_coords)

        # 17-21. Ball relative kinematics
        ball_xy = ball_pos[:2]
        ball_diff = ball_xy - ego_pos
        ball_dist_to_agent = float(np.linalg.norm(ball_diff))
        ball_rel_vel = ball_vel[:2] - ego_vel

        # 22. Ball distance to goal
        ball_dist_to_goal = float(np.linalg.norm(self.GOAL_POS - ball_xy))

        # 23. Is agent closest teammate to the ball
        all_tm_to_ball = np.linalg.norm(left_coords - ball_xy, axis=1)
        is_closest_to_ball = 1.0 if np.argmin(all_tm_to_ball) == ego_idx else 0.0

        # 24. Pass bearing to nearest open teammate
        target_tm_diff = nearest_tm_rel
        pass_angle_to_open_tm = float(np.arctan2(target_tm_diff[1], target_tm_diff[0]))

        # Assemble the 24 tactical dimensions
        features_24 = np.array([
            dist_to_goal,               # [0]
            angle_to_goal,              # [1]
            float(goal_dir[0]),         # [2]
            float(goal_dir[1]),         # [3]
            min_opp_dist,               # [4]
            pressing_intensity,         # [5]
            float(nearest_opp_rel[0]),  # [6]
            float(nearest_opp_rel[1]),  # [7]
            min_tm_dist,                # [8]
            float(nearest_tm_rel[0]),   # [9]
            float(nearest_tm_rel[1]),   # [10]
            xg_proxy,                   # [11]
            zone_one_hot[0],            # [12]
            zone_one_hot[1],            # [13]
            zone_one_hot[2],            # [14]
            corridor_clearance,         # [15]
            ball_dist_to_agent,         # [16]
            float(ball_diff[0]),        # [17]
            float(ball_diff[1]),        # [18]
            float(ball_rel_vel[0]),     # [19]
            float(ball_rel_vel[1]),     # [20]
            ball_dist_to_goal,          # [21]
            is_closest_to_ball,         # [22]
            pass_angle_to_open_tm,      # [23]
        ], dtype=np.float32)

        assert features_24.shape == (self.SEMANTIC_FEATURE_DIM,), (
            f"Semantic vector shape mismatch: expected ({self.SEMANTIC_FEATURE_DIM},), got {features_24.shape}"
        )
        return features_24

    def build_semantic_observation(self, obs: np.ndarray, agent_idx: Optional[int] = None) -> np.ndarray:
        """Constructs the full 139-dimensional semantic observation vector.

        Concatenates the base 115D simple115v2 raw observation with the 24D tactical
        semantic feature vector: [115D raw, 24D semantic] -> 139D total.

        Args:
            obs: 1D array of length >= 115 (GRF simple115 format).
            agent_idx: Optional agent index within the team (0..10).

        Returns:
            np.ndarray: 1D float32 array of shape (139,).
        """
        raw_115 = np.asarray(obs[:self.RAW_OBS_DIM], dtype=np.float32)
        if len(raw_115) < self.RAW_OBS_DIM:
            # Pad if synthetic or truncated
            padded = np.zeros(self.RAW_OBS_DIM, dtype=np.float32)
            padded[:len(raw_115)] = raw_115
            raw_115 = padded

        sem_24 = self.compute_semantic_vector(raw_115, agent_idx=agent_idx)
        obs_139 = np.concatenate([raw_115, sem_24], axis=0).astype(np.float32)

        assert obs_139.shape == (self.TOTAL_SEMANTIC_OBS_DIM,), (
            f"Full semantic observation shape mismatch: expected ({self.TOTAL_SEMANTIC_OBS_DIM},), got {obs_139.shape}"
        )
        return obs_139

    def extract_features(self, obs: np.ndarray) -> Dict[str, Any]:
        """Extracts structured semantic indicators from a 115-dim observation vector.

        Args:
            obs: 1D array of length >= 115.

        Returns:
            Dictionary containing computed semantic scalars and vectors.
        """
        sem_vec = self.compute_semantic_vector(obs)
        zone_str = "defensive_third" if sem_vec[12] == 1.0 else ("middle_third" if sem_vec[13] == 1.0 else "attacking_third")
        possession = int(np.argmax(obs[94:97])) if len(obs) >= 97 else 1
        return {
            "ball_possession_team": possession,
            "dist_to_goal": float(sem_vec[0]),
            "angle_to_goal": float(sem_vec[1]),
            "min_opponent_dist": float(sem_vec[4]),
            "pressing_intensity": float(sem_vec[5]),
            "min_teammate_dist": float(sem_vec[8]),
            "xg_proxy": float(sem_vec[11]),
            "pitch_zone": zone_str,
            "pitch_zone_one_hot": [float(sem_vec[12]), float(sem_vec[13]), float(sem_vec[14])],
            "corridor_clearance": float(sem_vec[15]),
            "ball_dist_to_agent": float(sem_vec[16]),
            "is_closest_to_ball": float(sem_vec[22]),
        }

    def _compute_corridor_clearance(
        self, ego_pos: np.ndarray, teammates: np.ndarray, opponents: np.ndarray
    ) -> float:
        """Estimates passing corridor clearance to nearest open teammate."""
        if len(teammates) == 0 or len(opponents) == 0:
            return 1.0

        target_tm = teammates[0]
        pass_vec = target_tm - ego_pos
        pass_len = np.linalg.norm(pass_vec)
        if pass_len < 1e-6:
            return 1.0

        unit_pass = pass_vec / pass_len
        min_clearance = 1.0
        for opp in opponents:
            opp_vec = opp - ego_pos
            proj_dist = np.dot(opp_vec, unit_pass)
            if 0 < proj_dist < pass_len:
                perp_dist = np.linalg.norm(opp_vec - proj_dist * unit_pass)
                min_clearance = min(min_clearance, float(perp_dist))

        return float(min_clearance)

    def _fallback_semantic_vector(self) -> np.ndarray:
        """Provides default values when observation vector is synthetic or truncated."""
        vec = np.zeros(self.SEMANTIC_FEATURE_DIM, dtype=np.float32)
        vec[0] = 0.5   # dist_to_goal
        vec[2] = 1.0   # goal_dir_x
        vec[4] = 0.3   # min_opponent_dist
        vec[8] = 0.25  # min_teammate_dist
        vec[11] = 0.15 # xg_proxy
        vec[13] = 1.0  # middle third
        vec[15] = 0.5  # corridor_clearance
        vec[16] = 0.4  # ball_dist_to_agent
        vec[21] = 0.5  # ball_dist_to_goal
        return vec
