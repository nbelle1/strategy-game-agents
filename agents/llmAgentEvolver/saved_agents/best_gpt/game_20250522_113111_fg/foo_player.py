import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
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

        # ===== YOUR CODE HERE =====
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
                print("LLM response ambiguous or invalid. Defaulting to first action.")
                return playable_actions[0]
        except Exception as e:
            # Handle any exceptions from the LLM
            print(f"Error querying LLM: {e}. Defaulting to first action.")
            return playable_actions[0]
        # ===== END YOUR CODE =====

    def _build_prompt(self, game, playable_actions):
        """
        Constructs a prompt for the LLM to evaluate actions.

        Args:
            game (Game): The current game state.
            playable_actions (Iterable[Action]): The actions available to the player.

        Returns:
            str: The prompt to send to the LLM.
        """
        # Include relevant game state and action options in the prompt
        prompt = """
        You are playing the Catanatron Minigame.
        Here is the current game state:
        {game_state}

        Here are the actions you can choose from:
        {actions_list}

        Please provide the best action to take based on maximizing Victory Points (VP) and achieving strategic goals.
        """.format(
            game_state=str(game),  # Convert game state to string for the prompt
            actions_list="\n".join([str(action) for action in playable_actions])
        )
        return prompt

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