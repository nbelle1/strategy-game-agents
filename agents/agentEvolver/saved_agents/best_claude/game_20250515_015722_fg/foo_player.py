from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.models.enums import ROAD, SETTLEMENT, CITY
from catanatron.state_functions import (
    get_player_freqdeck, 
    player_num_resource_cards,
    get_visible_victory_points,
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_played_dev_cards,
)
import random
from collections import Counter


class FooPlayer(Player):
    def __init__(self, color):
        super().__init__(color)  # Pass color directly to parent class
        self.name = "FooPlayer"  # Set name after parent initialization

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
            print(f"Evaluating action: {action.action_type} with score {score}")
            if score > best_score:
                best_score = score
                best_action = action
        
        print(f"Selected action: {best_action.action_type} with score {best_score}")
        return best_action
    
    def evaluate_action(self, game, action):
        """
        Assign a score to an action based on its type, impact, and context.
        Higher score means better action.
        """
        try:
            if action.action_type == ActionType.BUILD_CITY:
                # Cities are highest priority but now we evaluate their location too
                node_id = action.value
                # Base score plus production value of the location
                return 100 + self.evaluate_settlement_location(game, node_id, self.color)
                
            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                # Evaluate settlements based on their resource production potential
                node_id = action.value
                location_value = self.evaluate_settlement_location(game, node_id, self.color)
                return 50 + location_value
                
            elif action.action_type == ActionType.BUILD_ROAD:
                # Evaluate roads based on where they can lead
                edge = action.value
                road_value = self.evaluate_road_location(game, edge, self.color)
                return 20 + road_value
                
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                # Evaluate if development card is the best use of resources
                # Check if we're close to victory - save resources for settlements/cities if so
                try:
                    vps = get_visible_victory_points(game.state, self.color)
                    if vps >= 7:  # Close to winning, prioritize direct VP strategies
                        return 5  # Lower priority when close to winning
                except Exception:
                    pass  # Fall back to default value if we can't get VPs
                return 15
                
            elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
                # Evaluate knight card play based on robber placement value
                try:
                    # Higher value if close to largest army achievement
                    knights_played = get_played_dev_cards(game.state, self.color, "KNIGHT")
                    if knights_played >= 2:  # Will have 3+ after playing this one
                        return 20
                except Exception:
                    pass  # Fall back to default value if we can't get knight info
                return 10
                
            elif action.action_type == ActionType.MOVE_ROBBER:
                # Simple robber strategy for now - can be enhanced later
                return 5
                
            elif action.action_type == ActionType.MARITIME_TRADE:
                # Consider what resources we're getting vs giving
                return 3
                
            elif action.action_type == ActionType.ROLL:
                # Need to roll dice to proceed with turn
                return 30
                
            elif action.action_type == ActionType.END_TURN:
                # Lowest priority, only if nothing better to do
                return -10
            
            # Default score for other actions
            return 0
            
        except Exception as e:
            print(f"Error evaluating action {action.action_type}: {e}")
            # Return default scores if evaluation fails
            if action.action_type == ActionType.BUILD_CITY:
                return 100
            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                return 50
            elif action.action_type == ActionType.BUILD_ROAD:
                return 20
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                return 15
            elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
                return 10
            elif action.action_type == ActionType.MOVE_ROBBER:
                return 5
            elif action.action_type == ActionType.MARITIME_TRADE:
                return 3
            elif action.action_type == ActionType.ROLL:
                return 30
            elif action.action_type == ActionType.END_TURN:
                return -10
            return 0
    
    def evaluate_settlement_location(self, game, node_id, player_color):
        """
        Calculate the value of a settlement location based on:
        - Resource production probability
        - Resource diversity
        - Port access
        
        Returns a numeric score where higher is better.
        """
        try:
            # Get production values at this node
            node_production = game.state.board.map.node_production[node_id]
            
            # Calculate basic production value (sum of probabilities)
            production_value = sum(node_production.values())
            
            # Add bonus for resource diversity
            resource_types = len(node_production.keys())
            diversity_bonus = resource_types * 5  # Emphasize resource diversity
            
            # Check for port access
            port_bonus = 0
            for resource, nodes in game.state.board.map.port_nodes.items():
                if node_id in nodes:
                    if resource is None:  # 3:1 port
                        port_bonus = 10
                    else:  # 2:1 port
                        port_bonus = 20
            
            # Combine factors
            total_value = production_value + diversity_bonus + port_bonus
            print(f"Settlement at {node_id} has value: {total_value}")
            return total_value
        except Exception as e:
            # Handle gracefully if we can't evaluate (e.g., in initial placement)
            print(f"Error evaluating settlement location: {e}")
            return 0
    
    def evaluate_road_location(self, game, edge, player_color):
        """
        Evaluate the strategic value of building a road at a specific edge.
        Higher scores for roads that lead to good potential settlement spots.
        """
        try:
            # Unpack the edge (it's a tuple of two node IDs)
            node1, node2 = edge
            
            # Check if either endpoint is or could be a buildable settlement location
            value = 0
            
            # Get buildable nodes
            buildable_nodes = game.state.board.buildable_node_ids(player_color)
            
            # Add value if the road connects directly to a buildable node
            if node1 in buildable_nodes:
                value += 5
                # Add more value if the node has good production
                prod_value = self.evaluate_settlement_location(game, node1, player_color)
                value += prod_value / 10
                
            if node2 in buildable_nodes:
                value += 5
                # Add more value if the node has good production
                prod_value = self.evaluate_settlement_location(game, node2, player_color)
                value += prod_value / 10
                
            return value
        except Exception as e:
            print(f"Error evaluating road location: {e}")
            return 0