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
        # As an example we simply return the first action:
        print("Choosing First Action on Default")
        return playable_actions[0]
        # ===== END YOUR CODE =====

