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

        # Helper function for evaluating trades
        def evaluate_trade(current_resources, target_cost):
            needed_resources = {r: target_cost.get(r, 0) - current_resources.get(r, 0)
                                for r in target_cost if target_cost.get(r, 0) > current_resources.get(r, 0)}
            surplus_resources = [r for r, qty in current_resources.items() if qty > 3]
            if needed_resources and surplus_resources:
                return (surplus_resources[0], list(needed_resources.keys())[0])  # Trade surplus for needed
            return None

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

        # Resource Utilization Strategy Implementation
        settlement_cost = {'Brick': 1, 'Wood': 1, 'Wheat': 1, 'Sheep': 1}
        road_cost = {'Brick': 1, 'Wood': 1}
        city_cost = {'Wheat': 2, 'Ore': 3}

        if game.current_player.can_build('settlement'):
            target_cost = settlement_cost
        elif game.current_player.can_build('road'):
            target_cost = road_cost
        elif game.current_player.can_build('city'):
            target_cost = city_cost
        else:
            target_cost = None

        for action in playable_actions:
            if action.action_type == ActionType.TRADE_RESOURCE and target_cost:
                trade = evaluate_trade(game.current_player.resources, target_cost)
                if trade:
                    print(f"Trading {trade[0]} for {trade[1]}")
                    return action

        # Prevent Resource Stagnation
        def prevent_stagnation(resources):
            for resource, qty in resources.items():
                if qty > 4:  # Excess threshold
                    return resource
            return None

        stagnant_resource = prevent_stagnation(game.current_player.resources)
        if stagnant_resource:
            print(f"Using or trading surplus resource: {stagnant_resource}")

        # Default to the first action as a fallback
        print("Choosing First Action on Default")
        return playable_actions[0]
        
        # ===== END YOUR CODE =====