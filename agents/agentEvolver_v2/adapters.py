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
### KEEP THESE IMPORTS BELOW THIS LINE ###
from catanatron.game import Game  # has .state, .copy(), .execute(), .winning_color()
from catanatron.models.player import Player, Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.minimax import SameTurnAlphaBetaPlayer
from catanatron_experimental.machine_learning.players.value import DEFAULT_WEIGHTS
from catanatron_experimental.machine_learning.players.tree_search_utils import list_pruned_actions
from catanatron_experimental.machine_learning.players.tree_search_utils import execute_deterministic
from catanatron_experimental.machine_learning.players.tree_search_utils import execute_spectrum
from catanatron_experimental.machine_learning.players.tree_search_utils import expand_spectrum
from catanatron_experimental.machine_learning.players.tree_search_utils import prune_robber_actions
from catanatron_experimental.machine_learning.players.tree_search_utils import list_pruned_actions as _list_pruned_actions
### KEEP THESE IMPORTS ABOVE THIS LINE ###

# =============================================
# Thin Wrappers for Core Functions
# =============================================

# (game: Game, action: Action) -> Game
from catanatron_experimental.machine_learning.players.tree_search_utils import execute_deterministic as _execute_deterministic
def execute_deterministic(game, action):
    """Execute an action deterministically, returning the resulting game state."""
    return _execute_deterministic(game, action)

# (game: Game, action: Action) -> List[Tuple[Game, float]]
from catanatron_experimental.machine_learning.players.tree_search_utils import execute_spectrum as _execute_spectrum
def execute_spectrum(game, action):
    """Execute an action, returning all possible outcomes and their probabilities."""
    return _execute_spectrum(game, action)

# (game: Game, actions: List[Action]) -> List[Tuple[Game, float]]
from catanatron_experimental.machine_learning.players.tree_search_utils import expand_spectrum as _expand_spectrum
def expand_spectrum(game, actions):
    """Expand all chance outcomes (dice, dev cards, robber) for the current game state and actions."""
    return _expand_spectrum(game, actions)

# (game: Game) -> List[Action]
from catanatron_experimental.machine_learning.players.tree_search_utils import list_pruned_actions as _list_pruned_actions
def list_pruned_actions(game):
    """List all actions after pruning unlikely or suboptimal moves."""
    return _list_pruned_actions(game)

# (current_color: str, game: Game, actions: List[Action]) -> List[Action]
from catanatron_experimental.machine_learning.players.tree_search_utils import prune_robber_actions as _prune_robber_actions
def prune_robber_actions(current_color, game, actions):
    """Prune robber placement actions to only the most relevant tiles."""
    return _prune_robber_actions(current_color, game, actions)

# (params: dict) -> Callable[[Game], float]
from catanatron_experimental.machine_learning.players.value import base_fn as _base_fn
def base_fn(params=DEFAULT_WEIGHTS):
    """Build a base heuristic value function for evaluating game states."""
    return _base_fn(params)

# (sample: dict, player_name: str, include_variety: bool) -> float
from catanatron_experimental.machine_learning.players.value import value_production as _value_production
def value_production(sample, player_name="P0", include_variety=True):
    """Evaluate game state based on resource production potential."""
    return _value_production(sample, player_name, include_variety)

# (params: dict) -> Callable[[Game], float]
from catanatron_experimental.machine_learning.players.value import contender_fn as _contender_fn
def contender_fn(params):
    """Build a value function based on contender status (e.g., longest road, largest army)."""
    return _contender_fn(params)

# (name: str, params: dict, value_function: Callable) -> Callable[[Game], float]
from catanatron_experimental.machine_learning.players.value import get_value_fn as _get_value_fn
def get_value_fn(name, params, value_function=None):
    """Get a value function tailored to the specified name and parameters."""
    return _get_value_fn(name, params, value_function)
