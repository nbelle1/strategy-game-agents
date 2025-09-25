"""Unified adapter for Catanatron agents.

Expose a small, stable surface for multi-agent systems to:
- Inspect game state
- Enumerate legal actions
- Execute hypothetical moves (with/without validation)
- Expand chance outcomes (dice, dev cards, robber)
- Use pruning helpers
- Build/evaluate heuristics

Everything here is a thin re-export or trivial wrapper from catanatron & friends.
"""

from typing import Callable, List, Optional, Tuple, Dict, Any

# CORE RE-EXPORTS
from catanatron.game import Game  # Game instance with .state, .copy(), .execute(action), .winning_color()
from catanatron.models.player import Player, Color  # Player and Color types
from catanatron.models.enums import Action, ActionType  # Action = namedtuple("Action", ["color", "action_type", "value"]) 

# Player and debug node classes (re-exported so consumers can import them from adapters)
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,  # Player that executes an AlphaBeta search with expected value calculation
    SameTurnAlphaBetaPlayer,  # AlphaBeta constrained to the same turn
    DebugStateNode,  # Node for debugging the AlphaBeta search tree
    DebugActionNode,  # Node representing an action in the AlphaBeta search tree
)
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,  # Player using heuristic value functions
    DEFAULT_WEIGHTS,  # Default weight set for value functions
)

# Underlying implementation imports (underscore aliases to avoid recursion)
from catanatron_experimental.machine_learning.players.tree_search_utils import (
    execute_deterministic as _execute_deterministic,
    execute_spectrum as _execute_spectrum,
    expand_spectrum as _expand_spectrum,
    list_prunned_actions as _list_prunned_actions,  # spelling verified in source
    prune_robber_actions as _prune_robber_actions,
)
from catanatron_experimental.machine_learning.players.minimax import render_debug_tree as _render_debug_tree

from catanatron_experimental.machine_learning.players.value import (
    base_fn as _base_fn,
    contender_fn as _contender_fn,
    value_production as _value_production,
    get_value_fn as _get_value_fn,
)

# Public API
__all__ = [
    "Game",
    "Player",
    "Color",
    "Action",
    "ActionType",
    "AlphaBetaPlayer",
    "SameTurnAlphaBetaPlayer",
    "ValueFunctionPlayer",
    "DebugStateNode",
    "DebugActionNode",
    "copy_game",
    "execute_deterministic",
    "execute_spectrum",
    "expand_spectrum",
    "list_prunned_actions",
    "prune_robber_actions",
    "render_debug_tree",
    "base_fn",
    "contender_fn",
    "value_production",
    "get_value_fn",
]

# THIN CONVENIENCE WRAPPERS
def copy_game(game: Game) -> Game:
    '''Create a deep copy of the game state.'''
    return game.copy()

def execute_deterministic(game: Game, action: Action) -> List[Tuple[Game, float]]:
    '''Execute a deterministic action and return the resulting game state with probability 1.'''
    return _execute_deterministic(game, action)

def execute_spectrum(game: Game, action: Action) -> List[Tuple[Game, float]]:
    '''Return a list of (game_copy, probability) tuples for all possible outcomes of an action.'''
    return _execute_spectrum(game, action)

def expand_spectrum(game: Game, actions: List[Action]) -> Dict[Action, List[Tuple[Game, float]]]:
    '''Expand a game state into all possible outcomes for a list of actions.'''
    return _expand_spectrum(game, actions)

def list_prunned_actions(game: Game) -> List[Action]:
    '''Returns a pruned list of actions to reduce the search space.'''
    return _list_prunned_actions(game)

def prune_robber_actions(current_color: Color, game: Game, actions: List[Action]) -> List[Action]:
    '''Prunes robber actions to keep only the most impactful ones.'''
    return _prune_robber_actions(current_color, game, actions)

def render_debug_tree(node: DebugStateNode) -> str:
    '''Renders the AlphaBeta search tree using Graphviz.'''
    return _render_debug_tree(node)

# HEURISTIC BUILDERS
def base_fn(params=DEFAULT_WEIGHTS) -> Callable[[Game, Color], float]:
    '''Base value function factory for evaluating game states.'''
    return _base_fn(params)

def contender_fn(params) -> Callable[[Game, Color], float]:
    '''Alternative value function factory with tuned weights.'''
    return _contender_fn(params)

def value_production(sample, player_name: str = "P0", include_variety: bool = True) -> float:
    '''Compute the production value of a player's state.'''
    return _value_production(sample, player_name, include_variety)

def get_value_fn(name: str, params, value_function=None) -> Callable[[Game, Color], float]:
    '''Factory that returns a value function by name and parameters.'''
    return _get_value_fn(name, params, value_function)
