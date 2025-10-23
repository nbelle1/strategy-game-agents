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
        #         Defined in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions

        # ===== YOUR CODE HERE =====

        # Refactored helper function for fetching current player resources
        def get_current_player_resources(state):
            current_color = state.current_player().color
            player_prefix = f"P{state.color_to_index[current_color]}"
            resources = {
                "wood": state.player_state[f"{player_prefix}_WOOD_IN_HAND"],
                "brick": state.player_state[f"{player_prefix}_BRICK_IN_HAND"],
                "sheep": state.player_state[f"{player_prefix}_SHEEP_IN_HAND"],
                "wheat": state.player_state[f"{player_prefix}_WHEAT_IN_HAND"],
                "ore": state.player_state[f"{player_prefix}_ORE_IN_HAND"],
            }
            return resources

        # Helper function for evaluating trades
        def evaluate_trade(current_resources, target_cost):
            needed_resources = {r: target_cost.get(r, 0) - current_resources.get(r, 0)
                                for r in target_cost if target_cost.get(r, 0) > current_resources.get(r, 0)}
            surplus_resources = [r for r, qty in current_resources.items() if qty > 3]
            if needed_resources and surplus_resources:
                return (surplus_resources[0], list(needed_resources.keys())[0])  # Trade surplus for needed
            return None

        # Fetch current player resources using the helper function
        current_resources = get_current_player_resources(game.state)

        print(f"Current Player Resources: {current_resources}")

        # Settlement Expansion Strategy Implementation
        settlement_cost = {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1}
        can_build_settlement = all(
            current_resources.get(resource, 0) >= quantity
            for resource, quantity in settlement_cost.items()
        )
        if can_build_settlement:
            print("Evaluating actions for Settlement Expansion Strategy")
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    print(f"Chosen Action: {action}")
                    return action

        # Road Building Strategy Implementation
        road_cost = {'wood': 1, 'brick': 1}
        can_build_road = all(
            current_resources.get(resource, 0) >= quantity
            for resource, quantity in road_cost.items()
        )
        if can_build_road:
            print("Evaluating actions for Road Building Strategy")
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_ROAD:
                    print(f"Chosen Action: {action}")
                    return action

        # Resource Utilization Strategy Implementation
        city_cost = {'wheat': 2, 'ore': 3}

        if can_build_settlement:
            target_cost = settlement_cost
        elif can_build_road:
            target_cost = road_cost
        elif all(current_resources.get(resource, 0) >= quantity for resource, quantity in city_cost.items()):
            target_cost = city_cost
        else:
            target_cost = None

        for action in playable_actions:
            if action.action_type == ActionType.TRADE_RESOURCE and target_cost:
                trade = evaluate_trade(current_resources, target_cost)
                if trade:
                    print(f"Trading {trade[0]} for {trade[1]}")
                    return action

        # Prevent Resource Stagnation
        def prevent_stagnation(resources):
            for resource, qty in resources.items():
                if qty > 4:  # Excess threshold
                    return resource
            return None

        stagnant_resource = prevent_stagnation(current_resources)
        if stagnant_resource:
            print(f"Using or trading surplus resource: {stagnant_resource}")

        # Default to the first action as a fallback
        print("Choosing First Action on Default")
        return playable_actions[0]

        # ===== END YOUR CODE =====