"""
Unified adapter for Catanatron agents.

Expose a small, stable surface for multi-agent systems to:
- Inspect game state
- Enumerate legal actions
- Execute hypothetical moves (with/without validation)
- Expand chance outcomes (dice, dev cards, robber)
- Use pruning helpers
- Build/evaluate heuristics

Everything here is a thin re-export or trivial wrapper from catanatron & friends.
"""

from typing import Any, Iterable, List, Tuple, Dict, Optional

# === Core types & enums =======================================================
from catanatron.game import Game  # has .state, .copy(), .execute(), .winning_color()
from catanatron.models.player import Player, Color
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    CITY,
    Action,        # Action(color, action_type, value?)
    ActionType,    # END_TURN, BUILD_SETTLEMENT, ROLL, MOVE_ROBBER, ...
)

# === State helpers (commonly used by search / heuristics) ====================
from catanatron.state_functions import (
    get_player_buildings,   # (state, color, BUILDING_TYPE) -> Iterable[node_id]
    get_dev_cards_in_hand,  # (state, color, card) -> int
    get_player_freqdeck,    # (state, color) -> frequency deck (hand histogram)
    get_enemy_colors,       # (colors, my_color) -> Iterable[color]
)

# === Map / probability utilities ============================================
from catanatron.models.map import number_probability  # P(roll sum)

# === Features & value functions =============================================
from catanatron_gym.features import build_production_features
from catanatron_experimental.machine_learning.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,          # (builder_name, params, custom_value_fn?) -> callable(game, pov_color)->float
    value_production,      # features -> scalar
)

# === Tree-search spectrum & pruning (chance and action reduction) ============
# Bring in the exact logic AlphaBeta uses
from catanatron_experimental.machine_learning.players.tree_search_utils import (
    DETERMINISTIC_ACTIONS,
    execute_deterministic,
    execute_spectrum,      # (game, action) -> List[(game_copy, proba)]
    expand_spectrum,       # (game, actions) -> Dict[action, List[(game_copy, proba)]]
    list_prunned_actions,  # (game) -> List[action]
    prune_robber_actions,  # (current_color, game, actions) -> filtered actions
)

# === Thin convenience wrappers ===============================================

def current_color(game: Game) -> Color:
    """Color to move at current state."""
    return game.state.current_color()

def playable_actions(game: Game) -> List[Action]:
    """Legal actions in the current state."""
    return list(game.state.playable_actions)

def is_initial_build_phase(game: Game) -> bool:
    return bool(game.state.is_initial_build_phase)

def colors(game: Game) -> List[Color]:
    return list(game.state.colors)

def board(game: Game) -> Any:
    """Board object (tiles, ports, map). Use sparingly to keep adapter stable."""
    return game.state.board

def copy_game(game: Game) -> Game:
    """Deep copy so search can branch safely."""
    return game.copy()

def execute(game: Game, action: Action, *, validate: bool = False) -> None:
    """Apply action to game. Use validate=False for speed in lookahead."""
    game.execute(action, validate_action=validate)

def winning_color(game: Game) -> Optional[Color]:
    """None if no winner yet; else Color."""
    return game.winning_color()

# === Action helpers ==========================================================

def make_action(color: Color, action_type: ActionType, value: Any = None) -> Action:
    """Uniform action constructor MAS can call without importing enums directly."""
    return Action(color, action_type, value)

# === Chance expansion (expectation over outcomes) ============================

def chance_children_for_action(game: Game, action: Action) -> List[Tuple[Game, float]]:
    """
    Mirror AlphaBeta's stochastic modeling:
    - Deterministic actions -> single (next_state, 1.0)
    - BUY_DEVELOPMENT_CARD -> all possible dev cards w/ deck-informed probabilities
    - ROLL -> all dice outcomes 2..12 with number_probability
    - MOVE_ROBBER -> all resource steals (if any) with uniform p=1/5
    """
    return execute_spectrum(game, action)

def chance_children(game: Game, actions: Iterable[Action]) -> Dict[Action, List[Tuple[Game, float]]]:
    """Batch version (action -> [(state, p), ...]) exactly like AlphaBeta uses."""
    return expand_spectrum(game, list(actions))

# === Pruning shortcuts =======================================================

def pruned_actions(game: Game) -> List[Action]:
    """Smart subset of legal actions (roads/settlements filtering, trades, robber)."""
    return list_prunned_actions(game)

# === Heuristic builders ======================================================

def make_value_fn(
    builder_name: str,
    params: Dict[str, float] = DEFAULT_WEIGHTS,
    custom_value_fn = None,
):
    """
    Build a positional value function.
      builder_name: "base_fn" or "contender_fn" (AlphaBeta uses these names)
      params: weight dict
      custom_value_fn: optional callable(game, pov_color)->float
    Returns: callable(game, pov_color)->float
    """
    return get_value_fn(builder_name, params, custom_value_fn)

def production_features_sampler(include_variety: bool = True):
    """
    Returns a callable features_fn(game, color)->feature_vector,
    matching what value_production expects.
    """
    return build_production_features(include_variety)

# === Common state queries used by heuristics / pruning =======================

def player_build_nodes(game: Game, color: Color, building_type: Any) -> Iterable[int]:
    return get_player_buildings(game.state, color, building_type)

def dev_cards_in_hand(game: Game, color: Color, card: Any) -> int:
    return get_dev_cards_in_hand(game.state, color, card)

def enemy_colors_of(game: Game, my_color: Color) -> Iterable[Color]:
    return get_enemy_colors(game.state.colors, my_color)

def opponent_freqdeck(game: Game, color: Color) -> Any:
    """Histogram-like representation of opponent resources (used by robber logic)."""
    return get_player_freqdeck(game.state, color)

# === Probability helper ======================================================

def p_roll(total: int) -> float:
    """P(sum of two fair dice == total), 2..12."""
    return number_probability(total)

# === Tiny debug helpers ======================================================

def state_action_count(game: Game) -> int:
    """Approximate ply index for labeling debug nodes (matches AlphaBeta usage)."""
    return len(game.state.actions)

def num_turns(game: Game) -> int:
    """Optional parity with AlphaBeta's commented debug prints."""
    return getattr(game.state, "num_turns", 0)
