import os
import sys
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType

# Adjust the path dynamically to ensure access to required modules
sys.path.append(os.path.abspath("./catanatron_core"))

try:
    from catanatron.catanatron.state_functions import settlement_possibilities, road_building_possibilities
except ModuleNotFoundError:
    print("Error: The module 'catanatron_core' could not be found. Ensure it is installed or available in the specified path.")
    settlement_possibilities = None
    road_building_possibilities = None

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Helper function for fetching current player resources
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

        # Helper function for finding valid settlement locations
        def find_valid_settlement_spots(game):
            if settlement_possibilities:
                current_color = game.state.current_player().color
                return settlement_possibilities(game.state, current_color)
            else:
                print("Settlement possibilities function is not available.")
                return []

        # Helper function for road-building towards settlements
        def find_road_to_settlement(game, playable_actions):
            if road_building_possibilities:
                current_color = game.state.current_player().color
                valid_road_spots = road_building_possibilities(game.state, current_color)
                settlement_spots = find_valid_settlement_spots(game)
                for action in playable_actions:
                    if action.action_type == ActionType.BUILD_ROAD and action.value in valid_road_spots:
                        # Check if the road leads to a valid settlement location
                        for node in game.state.board.edges[action.value]:
                            if node in settlement_spots:
                                return action
                return None
            else:
                print("Road building possibilities function is not available.")
                return None

        # Weighted fallback mechanism
        def rank_fallback_actions(playable_actions):
            action_scores = {}
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    action_scores[action] = 10
                elif action.action_type == ActionType.BUILD_CITY:
                    action_scores[action] = 15
                elif action.action_type == ActionType.BUILD_ROAD:
                    action_scores[action] = 8
                elif action.action_type in [ActionType.OFFER_TRADE, ActionType.MARITIME_TRADE]:
                    action_scores[action] = 6
                else:
                    action_scores[action] = 1
            return max(action_scores, key=action_scores.get) if action_scores else None

        # Determine game phase
        def determine_game_phase(game_state):
            num_turns = game_state.num_turns
            print(f"Number of completed turns: {num_turns}")
            if game_state.is_initial_build_phase:
                return "early"
            elif num_turns < 50:
                return "mid"
            else:
                return "late"

        # Fetch current resources
        current_resources = get_current_player_resources(game.state)
        print(f"Current Player Resources: {current_resources}")

        # Determine the current game phase
        phase = determine_game_phase(game.state)

        # Priority-based action selection
        if phase == "early":
            road_action = find_road_to_settlement(game, playable_actions)
            if road_action:
                print(f"Prioritized Road-Building Action: {road_action}")
                return road_action

        # Settlement Expansion Strategy
        settlement_spots = find_valid_settlement_spots(game)
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT and action.value in settlement_spots:
                print(f"Chosen Settlement Action: {action}")
                return action

        # Resource Utilization Strategy
        target_cost = {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}  # Default Settlement cost
        for action in playable_actions:
            if action.action_type == ActionType.MARITIME_TRADE:
                surplus_resource = next(
                    (res for res, qty in current_resources.items() if qty > 4), None
                )
                if surplus_resource:
                    print(f"Executing Maritime Trade with surplus: {surplus_resource}")
                    return action

        # Fallback Mechanism
        fallback_action = rank_fallback_actions(playable_actions)
        if fallback_action:
            print(f"Chosen Fallback Action Based on Ranking: {fallback_action}")
            return fallback_action

        # Default to the first action
        print("Choosing First Action as Default Fallback")
        return playable_actions[0]