import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.state_functions import (
    get_player_freqdeck, player_num_resource_cards,
    get_player_buildings, player_key
)
from catanatron.models.map import number_probability
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM
from catanatron.models.board import STATIC_GRAPH

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # Use self.llm.query_llm(str prompt) to query the LLM

    def decide(self, game, playable_actions):
        if not playable_actions:
            print("No playable actions available.")
            return None

        prompt = self._build_prompt(game, playable_actions)
        try:
            llm_response = self.llm.query_llm(prompt)
            print(f"LLM Response: {llm_response}")

            chosen_action = self._parse_llm_response(llm_response, playable_actions)
            if chosen_action:
                return chosen_action
            else:
                print("LLM response ambiguous or invalid. Defaulting to fallback mechanism.")
                return self._fallback_action(game, playable_actions)
        except Exception as e:
            print(f"Error querying LLM: {e}. Defaulting to fallback mechanism.")
            return self._fallback_action(game, playable_actions)

    def _build_prompt(self, game, playable_actions):
        state = game.state
        player_resources = get_player_freqdeck(state, Color.BLUE)
        settlements = get_player_buildings(state, Color.BLUE, "SETTLEMENT")
        cities = get_player_buildings(state, Color.BLUE, "CITY")
        player_key_blue = player_key(state, Color.BLUE)
        victory_points = state.player_state[f"{player_key_blue}_ACTUAL_VICTORY_POINTS"]
        roads_available = state.player_state[f"{player_key_blue}_ROADS_AVAILABLE"]

        heuristic_scores = self._compute_heuristic_scores(game, playable_actions)

        # Advanced and structured LLM prompt with priorities
        prompt = f"""
        You are playing the Catanatron Minigame.
        Your goal is to maximize Victory Points (VP) efficiently by following these priorities:
        1. Build settlements and roads early to expand for resource diversity.
        2. Upgrade settlements to cities later for higher VP returns.
        3. Block opponent progress when strategically advantageous.

        Current Game State:
        - Resources: {player_resources}
        - Settlements: {settlements}
        - Cities: {cities}
        - Victory Points (VP): {victory_points}
        - Roads Available: {roads_available}

        Available Actions:
        {chr(10).join([str(action) for action in playable_actions])}

        Heuristic Scores for Actions (strategic breakdown by priority):
        {chr(10).join([f"{action}: {score}" for action, score in heuristic_scores.items()])}

        Avoid repetitive or low-impact actions like ending turns unnecessarily. Instead, choose impactful actions that optimize VP growth.

        Respond with:
        Action: {{Chosen Action}}
        Justification: {{Reasoning behind the optimal choice}}
        """
        return prompt

    def _compute_heuristic_scores(self, game, playable_actions):
        heuristic_scores = {}
        state = game.state
        player_resources = get_player_freqdeck(state, Color.BLUE)
        roads_available = state.player_state[f"{player_key(state, Color.BLUE)}_ROADS_AVAILABLE"]

        for action in playable_actions:
            score = 0

            if action.action_type == ActionType.BUILD_CITY:
                # Stage-sensitive city building prioritization
                settlements = get_player_buildings(state, Color.BLUE, "SETTLEMENT")
                score = 15 if len(settlements) >= 3 else 7  # Requires foundational settlements

            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                # Calculate bonuses based on adjacent tile probabilities and diversity
                node_id = action.value
                adjacent_tiles = game.state.board.map.adjacent_tiles[node_id]
                resource_bonus = sum(
                    number_probability(tile.number) for tile in adjacent_tiles if tile.resource is not None
                )

                # Penalize low-value locations
                if any(number_probability(tile.number) < 4 for tile in adjacent_tiles):
                    resource_bonus -= 3
                
                score = 10 + resource_bonus  # Higher base score for settlements

            elif action.action_type == ActionType.BUILD_ROAD:
                # Incentivize roads enabling settlement expansion
                if roads_available > 0 and self._adjacent_empty_node(action.value, game.state.board):
                    score = 15  # High priority if road leads to settlement opportunities
                else:
                    score = 10

            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                # Encourage development cards during resource surpluses
                if player_resources[4] >= 1 and player_resources[3] >= 1:  # Ore and Wheat
                    score = 7  
                else:
                    score = 3

            else:
                score = 1  # Low-priority actions like "ROLL" and "END_TURN"

            # Resource bottleneck handling (e.g., surplus materials)
            if player_resources[1] >= 3 and player_resources[0] >= 3:  # Brick and Wood
                if action.action_type == ActionType.BUILD_ROAD:
                    score += 5  # Spend surplus on roads

            heuristic_scores[action] = score

        return heuristic_scores

    def _adjacent_empty_node(self, edge_id, board):
        """
        Finds empty nodes adjacent to the given edge.
        
        Args:
            edge_id (Tuple[int, int]): The edge defined by two node IDs.
            board (Board): Current game board state.
        
        Returns:
            bool: True if adjacent empty nodes exist, False otherwise.
        """
        adjacent_nodes = []
        for node in edge_id:
            neighbors = STATIC_GRAPH.neighbors(node)
            # Check if the neighbor node is empty (no building present)
            empty_neighbors = [
                neighbor for neighbor in neighbors 
                if neighbor not in board.buildings
            ]
            adjacent_nodes.extend(empty_neighbors)

        print(f"Adjacent empty nodes for edge {edge_id}: {adjacent_nodes}")
        return len(adjacent_nodes) > 0

    def _parse_llm_response(self, llm_response, playable_actions):
        for action in playable_actions:
            if str(action) in llm_response:
                return action
        print("No valid action found in LLM response.")
        return None

    def _fallback_action(self, game, playable_actions):
        state = game.state
        priority_order = [ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD, ActionType.BUY_DEVELOPMENT_CARD, ActionType.END_TURN]

        for action_type in priority_order:
            for action in playable_actions:
                if action.action_type == action_type:
                    print(f"Fallback: Chose action {action} based on priority order.")
                    return action

        print("Fallback mechanism applied. Defaulting to first available action.")
        return playable_actions[0]