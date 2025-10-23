import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.state_functions import (
    get_player_freqdeck, player_num_resource_cards,
    get_player_buildings, player_key
)
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM() # use self.llm.query_llm(str prompt) to query the LLM

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Args:
        #     game (Game): complete game state. read-only. 
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions

        if not playable_actions:
            print("No playable actions available.")
            return None

        # Prepare prompt for the LLM based on the game state and playable actions
        prompt = self._build_prompt(game, playable_actions)
        try:
            # Query the LLM for the best action
            llm_response = self.llm.query_llm(prompt)
            print(f"LLM Response: {llm_response}")

            # Try to parse the chosen action from the LLM response
            chosen_action = self._parse_llm_response(llm_response, playable_actions)
            if chosen_action:
                return chosen_action
            else:
                print("LLM response ambiguous or invalid. Defaulting to fallback mechanism.")
                return self._fallback_action(playable_actions)
        except Exception as e:
            # Handle any exceptions from the LLM
            print(f"Error querying LLM: {e}. Defaulting to fallback mechanism.")
            return self._fallback_action(playable_actions)

    def _build_prompt(self, game, playable_actions):
        """
        Constructs a prompt for the LLM to evaluate actions.

        Args:
            game (Game): The current game state.
            playable_actions (Iterable[Action]): The actions available to the player.

        Returns:
            str: The prompt to send to the LLM.
        """
        state = game.state
        player_resources = get_player_freqdeck(state, Color.BLUE)
        settlements = get_player_buildings(state, Color.BLUE, "SETTLEMENT")
        cities = get_player_buildings(state, Color.BLUE, "CITY")
        player_key_blue = player_key(state, Color.BLUE)
        victory_points = state.player_state[f"{player_key_blue}_ACTUAL_VICTORY_POINTS"]
        roads_available = state.player_state[f"{player_key_blue}_ROADS_AVAILABLE"]

        heuristic_scores = self._compute_heuristic_scores(game, playable_actions)

        # Build a structured and informative prompt
        prompt = f"""
        You are playing the Catanatron Minigame.
        Here is the current game state:
        Player Stats:
        - Resources: {player_resources}
        - Settlements: {settlements}
        - Cities: {cities}
        - Victory Points: {victory_points}
        - Roads Available: {roads_available}

        Available Actions:
        {chr(10).join([str(action) for action in playable_actions])}

        Heuristic Scores for Actions:
        {chr(10).join([f"{action}: {score}" for action, score in heuristic_scores.items()])}

        Please respond with the optimal action and briefly explain why it was chosen. Format:
        Action: {{Chosen Action}}
        Justification: {{Reasoning behind the decision}}
        """

        return prompt

    def _compute_heuristic_scores(self, game, playable_actions):
        """
        Computes heuristic scores for the available actions.

        Args:
            game (Game): The current game state.
            playable_actions (Iterable[Action]): The actions available to the player.

        Returns:
            dict: A dictionary mapping actions to heuristic scores.
        """
        heuristic_scores = {}
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_CITY:
                heuristic_scores[action] = 10  # Example score for building a city
            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                heuristic_scores[action] = 7  # Example score for building a settlement
            elif action.action_type == ActionType.BUILD_ROAD:
                heuristic_scores[action] = 5  # Example score for building a road
            else:
                heuristic_scores[action] = 1  # Lower scores for other actions
        return heuristic_scores

    def _parse_llm_response(self, llm_response, playable_actions):
        """
        Parses the LLM's response to select an action.

        Args:
            llm_response (str): The response from the LLM.
            playable_actions (Iterable[Action]): The list of actions available.

        Returns:
            Action or None: The chosen action, or None if the response is invalid.
        """
        for action in playable_actions:
            if str(action) in llm_response:  # Basic check for action matching response
                return action
        return None

    def _fallback_action(self, playable_actions):
        """
        Implements a ranked fallback mechanism when LLM responses are invalid.

        Args:
            playable_actions (Iterable[Action]): The list of actions available.

        Returns:
            Action: The chosen fallback action based on priority.
        """
        # Define fallback priority order
        priority_order = [ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD, ActionType.END_TURN]
        for action_type in priority_order:
            for action in playable_actions:
                if action.action_type == action_type:
                    return action
        # Default to first available action
        print("Fallback mechanism applied. Defaulting to first available action.")
        return playable_actions[0]