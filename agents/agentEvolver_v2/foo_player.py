import os
from .adapters import (
    Game, Player, Color, Action, ActionType,
    playable_actions, pruned_actions, chance_children,
    make_value_fn, DEFAULT_WEIGHTS, value_production,
    production_features_sampler, winning_color, copy_game
)


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.value_fn = make_value_fn("base_fn", DEFAULT_WEIGHTS)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.
        
        # Args:
        #     game (Game): complete game state. read-only. 
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        best_action = None
        best_value = float('-inf')
        
        for action in playable_actions:
            # Create a copy of the game to simulate the action
            game_copy = copy_game(game)
            # Execute the action on the copied game
            game_copy.execute(action)
            # Evaluate the value of the resulting game state
            value = self.value_fn(game_copy, self.color)
            
            # Update the best action if the current action has a higher value
            if value > best_value:
                best_value = value
                best_action = action
        
        # If no action was found, default to the first action
        if best_action is None:
            print("Choosing First Action on Default")
            return playable_actions[0]
        
        # Return the best action found
        return best_action
        # ===== END YOUR CODE =====
