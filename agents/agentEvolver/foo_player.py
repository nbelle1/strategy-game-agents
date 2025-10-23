import os
import random
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import ActionType


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        # Define action weights for weighted random selection
        self.ACTION_WEIGHTS = {
            ActionType.BUILD_SETTLEMENT: 0.6,
            ActionType.BUILD_ROAD: 0.3,
            ActionType.END_TURN: 0.1
        }

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.
        
        # Args:
        #     game (Game): complete game state. read-only.
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        print(f"Total playable actions: {len(playable_actions)}")  # Debug log

        # ===== WEIGHTED ACTION SELECTION =====
        # Filter playable actions to only include those with defined weights
        weighted_actions = [
            action for action in playable_actions 
            if action.action_type in self.ACTION_WEIGHTS  # Corrected attribute
        ]

        if not weighted_actions:
            print("Warning: No weighted actions available. Defaulting to first action.")
            return playable_actions[0]

        # Select an action based on weights
        selected_action = random.choices(
            weighted_actions,
            weights=[self.ACTION_WEIGHTS[action.action_type] for action in weighted_actions],  # Corrected attribute
            k=1
        )[0]

        print(f"Selected action: {selected_action.action_type}")  # Corrected attribute
        return selected_action
        # ===== END WEIGHTED ACTION SELECTION =====