"""Football simulation environments and wrappers."""

try:
    from .grf_wrapper import FootballMultiAgentEnv
    from .scenario_manager import ScenarioManager
    from .scenarios import Academy3v1Env, Counterattack3v2Env, ValidationConstraintError
except (ImportError, ValueError):
    from src.envs.grf_wrapper import FootballMultiAgentEnv
    from src.envs.scenario_manager import ScenarioManager
    from src.envs.scenarios import Academy3v1Env, Counterattack3v2Env, ValidationConstraintError

__all__ = [
    "FootballMultiAgentEnv",
    "ScenarioManager",
    "Academy3v1Env",
    "Counterattack3v2Env",
    "ValidationConstraintError",
]
