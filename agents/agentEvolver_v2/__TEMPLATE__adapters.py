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
### KEEP THESE IMPORTS ABOVE THIS LINE ###