import os
from .adapters import (
    Game, Player, Color, Action, ActionType,
    get_player_freqdeck, value_production, get_player_buildings,
    get_dev_cards_in_hand, get_enemy_colors, build_production_features,
    number_probability, RESOURCES, DEVELOPMENT_CARDS, SETTLEMENT, CITY
)

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.
        #
        # Args:
        #     game (Game): complete game state. read-only.
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        # ===== YOUR CODE HERE =====
        # Implement a 1-ply value lookahead to evaluate actions
        best_action = None
        best_value = -float('inf')
        
        for action in playable_actions:
            # Create a copy of the game to simulate the action
            game_copy = game.copy()
            game_copy.execute(action)
            
            # Evaluate the value of the resulting state
            features = build_production_features(game_copy.state, self.color)
            current_value = value_production(features, self.color)
            
            # Update the best action if the current value is higher
            if current_value > best_value:
                best_value = current_value
                best_action = action
        
        if best_action is not None:
            print(f"Choosing action with value: {best_value}")
            return best_action
        else:
            # Fallback to the first action if no action is found
            print("Choosing First Action on Default")
            return playable_actions[0]
        # ===== END YOUR CODE =====