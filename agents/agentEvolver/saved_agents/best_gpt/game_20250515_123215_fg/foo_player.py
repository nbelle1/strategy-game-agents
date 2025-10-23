import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Args:
        #     game (Game): complete game state. read-only. 
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions

        # ===== YOUR CODE HERE =====
        # Settlement Expansion Strategy Implementation
        print("Evaluating actions for Settlement Expansion Strategy")
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                print(f"Chosen Action: {action}")
                return action

        # Road Building Strategy Implementation
        print("Evaluating actions for Road Building Strategy")
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_ROAD:
                print(f"Chosen Action: {action}")
                return action

        # Default to the first action as a fallback
        print("Choosing First Action on Default")
        return playable_actions[0]
        # ===== END YOUR CODE =====