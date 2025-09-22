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
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,  # Tuple[str]: All development card types (e.g., 'knight', 'victory_point', 'monopoly').
    RESOURCES,         # Tuple[str]: All resource types (e.g., 'brick', 'wood', 'sheep', 'wheat', 'ore').
    SETTLEMENT,        # str: Constant representing a settlement building type.
    CITY,             # str: Constant representing a city building type.
    Action,           # Enum: Represents all possible actions in the game (e.g., 'build_settlement', 'play_knight').
    ActionType,       # Enum: Categorizes actions (e.g., 'build', 'play_dev_card', 'trade').
)
from catanatron.state_functions import (
    get_player_buildings,  # (state, color, building_type) -> List[int]: Returns a LIST of node IDs where the player has buildings of the specified type.
    get_dev_cards_in_hand, # (state, color) -> Dict[str, int]: Returns a DICTIONARY of development cards in the player's hand, mapped to their counts.
    get_player_freqdeck,   # (state, color) -> Dict[str, int]: Returns a DICTIONARY of resource cards in the player's hand, mapped to their counts.
    get_enemy_colors,      # (state, color) -> List[Color]: Returns a LIST of enemy player colors.
)
from catanatron_gym.features import (
    build_production_features,  # (state, color) -> np.ndarray: Returns a NUMPY ARRAY of production features for the player, used in ML models.
)
from catanatron_experimental.machine_learning.players.value import (
    value_production,  # (state, color) -> float: Returns a FLOAT score representing the production value of the player's current state.
)
from catanatron.models.map import (
    number_probability,  # (number: int) -> float: Returns the PROBABILITY of rolling a specific number (2-12) in Catan.
)
from catanatron.game import Game  # has .state, .copy(), .execute(), .winning_color()
from catanatron.models.player import Player, Color
### KEEP THESE IMPORTS ABOVE THIS LINE ###