import os
from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.game import Game
from catanatron.models.player import Color
import random


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name or "FooPlayer")

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only. 
                Defined in in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # If there's only one possible action, take it
        if len(playable_actions) == 1:
            return playable_actions[0]
            
        # Evaluate each action and find the best one
        best_action = None
        best_score = float('-inf')
        
        for action in playable_actions:
            score = self.evaluate_action(game, action)
            if score > best_score:
                best_score = score
                best_action = action
        
        print(f"Selected action: {best_action.action_type} with score {best_score}")
        return best_action
    
    def evaluate_action(self, game, action):
        """
        Assign a score to an action based on its type and impact.
        Higher score means better action.
        """
        # Basic action type priorities based on strategic importance
        if action.action_type == ActionType.BUILD_CITY:
            # Cities are highest priority as they provide 2 VPs and double resource production
            return 100
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            # Settlements are next - they provide VPs and new resource sources
            return 50
        elif action.action_type == ActionType.BUILD_ROAD:
            # Roads don't give VPs but allow expansion
            return 20
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # Development cards can provide VPs or special abilities
            return 15
        elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
            # Playing knight cards helps get largest army
            return 10
        elif action.action_type == ActionType.MOVE_ROBBER:
            # Blocking opponent resources can be strategic
            return 5
        elif action.action_type == ActionType.MARITIME_TRADE:
            # Trading is useful but not as good as building
            return 3
        elif action.action_type == ActionType.ROLL:
            # Need to roll dice to proceed with turn
            return 30
        elif action.action_type == ActionType.END_TURN:
            # Lowest priority, only if nothing better to do
            return -10
        
        # Default score for other actions
        return 0