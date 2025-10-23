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

        # Helper function for evaluating surplus resource trades
        def evaluate_trade_surplus(current_resources, surplus_limit=4):
            surplus_resources = [resource for resource, qty in current_resources.items() if qty > surplus_limit]
            if surplus_resources:
                return surplus_resources[0]  # Trade the largest surplus for needed resources
            return None

        # Helper function for evaluating resource deficits
        def evaluate_trade_deficit(current_resources, target_cost):
            needed_resources = {
                resource: target_cost[resource] - current_resources.get(resource, 0)
                for resource in target_cost
                if target_cost[resource] > current_resources.get(resource, 0)
            }
            return needed_resources

        # Helper function for prioritizing high-value actions based on game phase
        def prioritize_actions_by_phase(playable_actions, game_state, phase):
            priority_per_phase = {
                "early": [ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD],
                "mid": [ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT],
                "late": [ActionType.BUY_DEVELOPMENT_CARD, ActionType.BUILD_CITY],
            }
            for action_type in priority_per_phase[phase]:
                for action in playable_actions:
                    if action.action_type == action_type:
                        print(f"Prioritized based on phase ({phase}): {action.action_type}")
                        return action
            return None

        # Helper function to determine current game phase
        def determine_game_phase(game_state):
            num_turns = game_state.num_turns  # Fixed the invalid turn_count reference
            print(f"Number of completed turns: {num_turns}")
            if game_state.is_initial_build_phase:
                return "early"
            elif num_turns < 50:
                return "mid"
            else:
                return "late"

        # Fetch current player resources
        current_resources = get_current_player_resources(game.state)
        print(f"Current Player Resources: {current_resources}")

        # Determine the current game phase and prioritize actions accordingly
        phase = determine_game_phase(game.state)
        prioritized_action = prioritize_actions_by_phase(playable_actions, game.state, phase)
        if prioritized_action:
            return prioritized_action

        # Settlement Expansion Strategy Implementation
        settlement_cost = {"wood": 1, "brick": 1, "wheat": 1, "sheep": 1}
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
        road_cost = {"wood": 1, "brick": 1}
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

        # City Building Strategy
        city_cost = {"wheat": 2, "ore": 3}
        can_build_city = all(
            current_resources.get(resource, 0) >= quantity
            for resource, quantity in city_cost.items()
        )
        if can_build_city:
            print("Evaluating actions for City Building Strategy")
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_CITY:
                    print(f"Chosen Action: {action}")
                    return action

        # Resource Utilization Strategy Implementation
        target_cost = None
        if can_build_settlement:
            target_cost = settlement_cost
        elif can_build_road:
            target_cost = road_cost
        elif can_build_city:
            target_cost = city_cost

        for action in playable_actions:
            if action.action_type == ActionType.MARITIME_TRADE and target_cost:
                surplus_resource = evaluate_trade_surplus(current_resources)
                needed_resources = evaluate_trade_deficit(current_resources, target_cost)
                if surplus_resource and needed_resources:
                    print(f"Executing Maritime Trade: Trading {surplus_resource} for {list(needed_resources.keys())[0]}")
                    return action
            elif action.action_type == ActionType.OFFER_TRADE and target_cost:
                surplus_resource = evaluate_trade_surplus(current_resources)
                needed_resources = evaluate_trade_deficit(current_resources, target_cost)
                if surplus_resource and needed_resources:
                    print(f"Proposing Domestic Trade: Offering {surplus_resource} for {list(needed_resources.keys())[0]}")
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