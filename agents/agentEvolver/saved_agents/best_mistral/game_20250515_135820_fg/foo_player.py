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
        #         Defined in in \"catanatron/catanatron_core/catanatron/game.py\"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        # ===== YOUR CODE HERE =====
        # Iterate through all playable actions
        for action in playable_actions:
            # Prioritize building settlements
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                print('Choosing to build a settlement')
                return action
        # If no preferred action is found, fall back to the first action
        print('Choosing First Action on Default')
        return playable_actions[0]
        # ===== END YOUR CODE =====