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
        # Initialize variables to track the best action and its score
        best_action = None
        best_score = -1

        # Iterate through all playable actions
        for action in playable_actions:
            # Assign scores based on the type of action
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                score = 3
                print('Choosing to build a settlement')
            elif action.action_type == ActionType.BUILD_CITY:
                score = 2
                print('Choosing to build a city')
            elif action.action_type == ActionType.BUILD_ROAD:
                score = 1
                print('Choosing to build a road')
            else:
                score = 0

            # Update the best action if the current one has a higher score
            if score > best_score:
                best_action = action
                best_score = score

        # If no preferred action is found, fall back to the first action
        if best_action is None:
            best_action = playable_actions[0]
            print('Choosing First Action on Default')

        return best_action
        # ===== END YOUR CODE =====