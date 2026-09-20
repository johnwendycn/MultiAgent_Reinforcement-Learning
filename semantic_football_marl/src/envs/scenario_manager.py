"""Scenario definitions and configuration helpers for Google Research Football."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ScenarioInfo:
    """Metadata describing a football scenario.

    Attributes:
        name: Name of the scenario in GRF.
        num_left_players: Total players on the left (home) team.
        num_right_players: Total players on the right (away) team.
        num_controlled: Number of players directly controlled by MARL policy.
        has_keeper: Whether a goalkeeper is active.
        description: Textual description of scenario goals.
    """
    name: str
    num_left_players: int
    num_right_players: int
    num_controlled: int
    has_keeper: bool
    description: str


class ScenarioManager:
    """Manages preset scenario configurations for football MARL experiments."""

    SCENARIOS: Dict[str, ScenarioInfo] = {
        "11_vs_11_stochastic": ScenarioInfo(
            name="11_vs_11_stochastic",
            num_left_players=11,
            num_right_players=11,
            num_controlled=10,
            has_keeper=True,
            description="Full 11 vs 11 regulation match with stochastic engine mechanics and 10 learning outfield agents."
        ),
        "11_vs_11_kaggle": ScenarioInfo(
            name="11_vs_11_kaggle",
            num_left_players=11,
            num_right_players=11,
            num_controlled=10,
            has_keeper=True,
            description="Full 11 vs 11 regulation football match."
        ),
        "academy_3_vs_1_with_keeper": ScenarioInfo(
            name="academy_3_vs_1_with_keeper",
            num_left_players=3,
            num_right_players=2,
            num_controlled=3,
            has_keeper=True,
            description="3 attackers versus 1 defender and 1 goalkeeper."
        ),
        "academy_pass_and_shoot_with_keeper": ScenarioInfo(
            name="academy_pass_and_shoot_with_keeper",
            num_left_players=2,
            num_right_players=2,
            num_controlled=2,
            has_keeper=True,
            description="2 attackers coordinate to pass and score past a defender and goalkeeper."
        ),
        "academy_counterattack_easy": ScenarioInfo(
            name="academy_counterattack_easy",
            num_left_players=4,
            num_right_players=2,
            num_controlled=4,
            has_keeper=True,
            description="4 attackers counterattack against 1 defender and 1 goalkeeper."
        ),
        "counterattack_3_vs_2": ScenarioInfo(
            name="counterattack_3_vs_2",
            num_left_players=3,
            num_right_players=3,
            num_controlled=3,
            has_keeper=True,
            description="3 attackers counterattacking against 2 outfield defenders and 1 goalkeeper."
        ),
    }

    @classmethod
    def get_info(cls, scenario_name: str) -> ScenarioInfo:
        """Retrieves scenario metadata.

        Args:
            scenario_name: Identifier of the football scenario.

        Returns:
            ScenarioInfo dataclass instance.

        Raises:
            ValueError: If the scenario is unknown.
        """
        if scenario_name not in cls.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Available: {list(cls.SCENARIOS.keys())}"
            )
        return cls.SCENARIOS[scenario_name]

    @classmethod
    def list_scenarios(cls) -> List[str]:
        """Returns the list of supported scenario identifiers."""
        return list(cls.SCENARIOS.keys())
