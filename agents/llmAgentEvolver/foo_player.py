import random
from catanatron.models.player import Player, Color
from catanatron.models.enums import ActionType

class FooPlayer(Player):
    """Ultra-minimal Catanatron player matching the successful Evolution 13 pattern."""
    
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
    
    def decide(self, game, playable_actions):
        """Simple priority-based decision making with minimal logic."""
        try:
            # Handle required actions first
            for action in playable_actions:
                if action.action_type == ActionType.ROLL:
                    return action
                    
                if action.action_type == ActionType.DISCARD:
                    return action
                
                if action.action_type == ActionType.MOVE_ROBBER:
                    return action
            
            # Fixed priority list with correct enum values from catanatron.models.enums
            priority_order = [
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD,
                ActionType.BUY_DEVELOPMENT_CARD,
                ActionType.PLAY_KNIGHT_CARD,  # Fixed: This is the correct enum value
                ActionType.PLAY_YEAR_OF_PLENTY,
                ActionType.PLAY_MONOPOLY,
                ActionType.PLAY_ROAD_BUILDING,
                ActionType.MARITIME_TRADE,    # Added trading actions to help get resources for cities
                ActionType.END_TURN
            ]
            
            # Choose action based on priority with randomization for same type
            for action_type in priority_order:
                matching_actions = [a for a in playable_actions if a.action_type == action_type]
                if matching_actions:
                    # Choose randomly among actions of the same type
                    return random.choice(matching_actions)
            
            # Fallback to first action if nothing matches (should never happen)
            return playable_actions[0]
                
        except Exception:
            # On any error, just return the first action
            if playable_actions:
                return playable_actions[0]
            
            # If we somehow get here with no actions, raise an error
            raise ValueError("No playable actions available")