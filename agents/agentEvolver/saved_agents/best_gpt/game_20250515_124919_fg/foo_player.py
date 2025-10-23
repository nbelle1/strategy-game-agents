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

        # Replace invalid `get_opponent_hotspot_edges` logic
        def get_opponent_hotspot_edges(state, board, player_color):
            opponent_hotspot_edges = []
            for opponent_color in state.colors:
                if opponent_color == player_color:
                    continue

                buildable_edges = board.buildable_edges(opponent_color)
                connected_nodes = set.union(*board.connected_components[opponent_color])

                for edge in buildable_edges:
                    node1, node2 = edge
                    if node1 in connected_nodes or node2 in connected_nodes:
                        opponent_hotspot_edges.append(edge)

            return opponent_hotspot_edges

        # Scoring settlement spots dynamically
        def score_settlement_spots(game, player_color):
            settlement_spots = game.state.board.buildable_node_ids(player_color)
            spot_scores = {}
            for spot in settlement_spots:
                resources = [tile.resource for tile in game.state.board.tiles if spot in tile.nodes]
                score = len(set(resources))  # Resource diversity score
                if "ore" in resources or "wheat" in resources:
                    score += 2  # Bonus for high-value resources
                if spot in game.state.board.get_road_ends(player_color):
                    score += 3  # Accessibility bonus
                spot_scores[spot] = score
            return max(spot_scores, key=spot_scores.get) if spot_scores else None

        # Dynamic Road Evaluation for prioritizing actions
        def score_road_targets(game, player_color, playable_actions):
            settlement_targets = [spot for spot in game.state.board.buildable_node_ids(player_color)]
            road_scores = {}
            opponent_hotspot_edges = get_opponent_hotspot_edges(game.state, game.state.board, player_color)
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_ROAD:
                    road_end = action.value
                    if road_end in settlement_targets:
                        road_scores[action] = 10  # Favor roads leading to settlements
                    elif road_end in opponent_hotspot_edges:
                        road_scores[action] = 5  # Discourage opponent expansion
                    else:
                        road_scores[action] = 3  # General connectivity bonus
            return max(road_scores, key=road_scores.get) if road_scores else None

        # Trade for needed resources dynamically
        def trade_for_settlement_or_road(game, current_resources, playable_actions):
            settlement_cost = {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}
            road_cost = {"wood": 1, "brick": 1}

            # Use determine_game_phase instead of relying on invalid attributes
            current_phase = determine_game_phase(game.state)

            target_cost = settlement_cost if current_phase == "early" else road_cost
            needed_resources = {res: max(0, qty - current_resources.get(res, 0)) for res, qty in target_cost.items()}
            surplus_resources = [res for res, qty in current_resources.items() if qty > 4]

            for action in playable_actions:
                if action.action_type in [ActionType.MARITIME_TRADE, ActionType.OFFER_TRADE]:
                    # Match surplus with needed resources dynamically
                    if surplus_resources and needed_resources:
                        print(f"Trading surplus {surplus_resources[0]} for needed resource {list(needed_resources.keys())[0]}")
                        return action
            return None

        # Determine game phase
        def determine_game_phase(game_state):
            num_turns = game_state.num_turns
            print(f"Number of completed turns: {num_turns}")
            if game_state.is_initial_build_phase:
                return "early"
            elif num_turns < 50 and game_state.num_settlements_built < 2:
                return "early-mid"
            elif num_turns < 50:
                return "mid"
            else:
                return "late"

        # Fallback mechanism to rank playable actions
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

        # Fetch current resources
        current_resources = get_current_player_resources(game.state)
        print(f"Current Player Resources: {current_resources}")

        # Determine the current game phase
        phase = determine_game_phase(game.state)

        # Priority-based action selection
        if phase == "early":
            road_action = score_road_targets(game, game.state.current_player().color, playable_actions)
            if road_action:
                print(f"Prioritized Road-Building Action: {road_action}")
                return road_action

        # Settlement Expansion Strategy
        highest_scoring_spot = score_settlement_spots(game, game.state.current_player().color)
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT and action.value == highest_scoring_spot:
                print(f"Chosen Settlement Action Based on Scoring: {action}")
                return action

        # Resource Utilization Strategy
        trade_action = trade_for_settlement_or_road(game, current_resources, playable_actions)
        if trade_action:
            print(f"Chosen Trade Action to Meet Resource Needs: {trade_action}")
            return trade_action

        # Fallback Mechanism
        fallback_action = rank_fallback_actions(playable_actions)
        if fallback_action:
            print(f"Chosen Ranked Fallback Action: {fallback_action}")
            return fallback_action

        # Default to the first action
        print("Choosing First Action as Default Fallback")
        return playable_actions[0]