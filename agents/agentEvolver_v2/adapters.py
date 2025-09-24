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
from catanatron.game import Game  # Game = main game object; exposes .state, .copy(), .execute(), .winning_color()
from catanatron.models.player import Player, Color
### KEEP THESE IMPORTS ABOVE THIS LINE ###


# Thin convenience wrappers -------------------------------------------------
# (game: Game) -> Game
def copy_game(game: Game) -> Game:
    """Return an independent clone of `game` for use by search algorithms (minimax/MCTS). Mirrors Game.copy() semantics."""
    return game.copy()

# Note: this wrapper currently delegates to the instance method game.copy().
# If RESEARCHER later reports a canonical top-level function (e.g., catanatron.utils.copy_game(game)),
# update this wrapper to delegate to that function instead (one-line change).
