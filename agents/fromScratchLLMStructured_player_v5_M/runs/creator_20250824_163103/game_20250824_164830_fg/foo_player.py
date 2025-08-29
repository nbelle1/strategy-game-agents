import os
import re
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES, ActionType
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM

    def decide(self, game, playable_actions):
        """
        Make a decision about which action to take based on LLM evaluation.
        
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # If no actions available, return None
        if not playable_actions:
            print("No playable actions available")
            return None
        
        # For debugging - show how many actions are available
        print(f"Evaluating {len(playable_actions)} possible actions")
        
        # Create game state representation
        game_state = self._get_game_state_representation(game)
        
        # Create action descriptions with safe handling of potentially problematic actions
        action_descriptions = []
        for i, action in enumerate(playable_actions):
            try:
                description = self._describe_action(action, game)
                action_descriptions.append(f"Action {i}: {description}")
            except Exception as e:
                # Fallback description if the detailed description fails
                print(f"Error describing action {i}: {str(e)}")
                action_descriptions.append(f"Action {i}: {str(action)}")
        
        # Create the prompt for LLM
        prompt = f"""
        Current Game State:
        {game_state}
        
        Available Actions:
        {chr(10).join(action_descriptions)}
        
        Evaluate the above actions considering:
        1. Immediate resource gain
        2. Strategic positioning
        3. Blocking opponents
        4. Victory point potential
        
        Rank the top 3 actions from best to worst and explain your reasoning.
        Finally, output just the number of the best action (e.g., "Best Action: 2").
        """
        
        # Query the LLM
        print("Querying LLM for decision...")
        response = self.llm.query_llm(prompt)
        print(f"LLM Response Summary: {response[:100]}...")
        
        # Parse the response to get the chosen action
        match = re.search(r"Best Action: (\d+)", response)
        if match:
            chosen_index = int(match.group(1))
            if 0 <= chosen_index < len(playable_actions):
                print(f"LLM chose action {chosen_index}")
                return playable_actions[chosen_index]
            else:
                print(f"Invalid action index {chosen_index}, falling back to first action")
        else:
            print("Could not parse LLM response for best action")
        
        # Fallback to first action if parsing fails
        print("Falling back to first action")
        return playable_actions[0]

    def _get_game_state_representation(self, game):
        """
        Generate a string representation of the current game state.
        
        Args:
            game (Game): The current game state
        Returns:
            str: A human-readable representation of the game state
        """
        # Get player index and player object
        player_index = game.state.color_to_index[self.color]
        
        # Get resource counts using state_functions
        key = f"P{player_index}"
        wood_count = game.state.player_state.get(f"{key}_WOOD_IN_HAND", 0)
        brick_count = game.state.player_state.get(f"{key}_BRICK_IN_HAND", 0)
        sheep_count = game.state.player_state.get(f"{key}_SHEEP_IN_HAND", 0)
        wheat_count = game.state.player_state.get(f"{key}_WHEAT_IN_HAND", 0)
        ore_count = game.state.player_state.get(f"{key}_ORE_IN_HAND", 0)
        
        # Get victory points
        victory_points = game.state.player_state.get(f"{key}_VICTORY_POINTS", 0)
        
        # Get buildings
        settlements = game.state.buildings_by_color.get(self.color, {}).get("SETTLEMENT", [])
        cities = game.state.buildings_by_color.get(self.color, {}).get("CITY", [])
        roads = game.state.buildings_by_color.get(self.color, {}).get("ROAD", [])
        
        # Get development cards (if available in state)
        dev_cards_knights = game.state.player_state.get(f"{key}_KNIGHT_PLAYED", 0)
        dev_cards_victory = game.state.player_state.get(f"{key}_PLAYED_VICTORY_POINT", 0)
        
        # Create string representation of game state
        state_str = f"Your Color: {self.color.value}\n"
        
        # Resources
        state_str += "Your Resources:\n"
        state_str += f"- WOOD: {wood_count}\n"
        state_str += f"- BRICK: {brick_count}\n"
        state_str += f"- SHEEP: {sheep_count}\n"
        state_str += f"- WHEAT: {wheat_count}\n"
        state_str += f"- ORE: {ore_count}\n"
        
        # Victory points
        state_str += f"Your Victory Points: {victory_points}\n"
        
        # Buildings
        state_str += f"Your Buildings: {len(settlements)} settlements, {len(cities)} cities, {len(roads)} roads\n"
        
        # Development cards
        state_str += f"Your Development Cards: Knights played: {dev_cards_knights}, Victory points: {dev_cards_victory}\n"
        
        # Opponent summary
        state_str += "Opponents:\n"
        for color in game.state.colors:
            if color != self.color:
                opp_index = game.state.color_to_index[color]
                opp_key = f"P{opp_index}"
                opp_points = game.state.player_state.get(f"{opp_key}_VICTORY_POINTS", 0)
                opp_settlements = game.state.buildings_by_color.get(color, {}).get("SETTLEMENT", [])
                opp_cities = game.state.buildings_by_color.get(color, {}).get("CITY", [])
                opp_roads = game.state.buildings_by_color.get(color, {}).get("ROAD", [])
                state_str += f"- {color.value}: {opp_points} points, {len(opp_settlements)} settlements, {len(opp_cities)} cities, {len(opp_roads)} roads\n"
        
        # Game turn
        state_str += f"Current Turn: {game.state.num_turns}\n"
        
        return state_str

    def _describe_action(self, action, game):
        """
        Generate a human-readable description of an action.
        
        Args:
            action (Action): The action to describe
            game (Game): The current game state
        Returns:
            str: A human-readable description of the action
        """
        action_type = action.action_type
        
        # Safe value access with fallbacks
        def safe_value_str(value):
            if value is None:
                return "unknown"
            try:
                return str(value)
            except:
                return "unprintable value"
        
        if action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build settlement at node {safe_value_str(action.value)}"
        
        elif action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {safe_value_str(action.value)}"
        
        elif action_type == ActionType.BUILD_ROAD:
            return f"Build road at edge {safe_value_str(action.value)}"
        
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buy development card"
        
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            # Safely handle Knight card action
            try:
                if action.value is None:
                    return "Play Knight card (target not specified)"
                elif isinstance(action.value, tuple) and len(action.value) > 0:
                    return f"Play Knight card and move robber to {safe_value_str(action.value[0])}"
                else:
                    return f"Play Knight card with value {safe_value_str(action.value)}"
            except Exception as e:
                return "Play Knight card (error extracting details)"
        
        elif action_type == ActionType.END_TURN:
            return "End your turn"
        
        elif action_type == ActionType.ROLL:
            return "Roll the dice"
        
        elif action_type == ActionType.MOVE_ROBBER:
            try:
                location = "unknown location"
                target = "no player"
                
                if action.value is not None:
                    if isinstance(action.value, tuple) and len(action.value) > 0:
                        location = safe_value_str(action.value[0])
                        
                        if len(action.value) > 1:
                            target_color = action.value[1]
                            target = safe_value_str(target_color) if target_color else "no player"
                
                return f"Move robber to {location} and steal from {target}"
            except Exception as e:
                return f"Move robber (error extracting details: {str(e)})"
        
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            try:
                if action.value is None:
                    return "Play Year of Plenty card"
                else:
                    return f"Play Year of Plenty card and take resources {safe_value_str(action.value)}"
            except:
                return "Play Year of Plenty card (error extracting details)"
        
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            try:
                if action.value is None:
                    return "Play Road Building card"
                else:
                    return f"Play Road Building card and build roads at {safe_value_str(action.value)}"
            except:
                return "Play Road Building card (error extracting details)"
        
        elif action_type == ActionType.PLAY_MONOPOLY:
            try:
                if action.value is None:
                    return "Play Monopoly card"
                else:
                    return f"Play Monopoly card and take all {safe_value_str(action.value)} resources"
            except:
                return "Play Monopoly card (error extracting details)"
        
        elif action_type == ActionType.MARITIME_TRADE:
            try:
                if action.value is None:
                    return "Trade resources"
                elif isinstance(action.value, tuple) and len(action.value) >= 2:
                    return f"Trade {safe_value_str(action.value[0])} for {safe_value_str(action.value[1])}"
                else:
                    return f"Trade with values {safe_value_str(action.value)}"
            except:
                return "Trade resources (error extracting details)"
        
        elif action_type == ActionType.DISCARD:
            try:
                return f"Discard cards {safe_value_str(action.value)}"
            except:
                return "Discard cards (error extracting details)"
        
        elif action_type == ActionType.INITIAL_SETTLEMENT:
            return f"Build initial settlement at node {safe_value_str(action.value)}"
        
        elif action_type == ActionType.INITIAL_ROAD:
            return f"Build initial road at edge {safe_value_str(action.value)}"
        
        # Fallback for any other action types
        else:
            try:
                return f"{action_type.name} - {safe_value_str(action.value)}"
            except:
                return str(action_type)