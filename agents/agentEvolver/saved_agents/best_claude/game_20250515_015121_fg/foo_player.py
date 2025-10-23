import os
from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.game import Game
from catanatron.models.player import Color
import random
from collections import Counter


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
            vps = game.state.player_state[self.color].victory_points
            if vps >= 7:  # Close to winning, prioritize direct VP strategies
                return 5  # Lower priority when close to winning
            return 15
            
        elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
            # Evaluate knight card play based on robber placement value
            # and whether we're close to largest army
            knights_played = game.state.player_state[self.color].army_size
            # Higher value if close to largest army achievement
            current_largest = max([game.state.player_state[color].army_size 
                                  for color in game.state.player_state.keys()])
            if knights_played >= current_largest - 1:  # We could get largest army
                return 20
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
            board = game.state.board
            
            # Check if the road connects to potential future settlement spots
            value = 0
            
            # Get the nodes connected by this edge
            connected_nodes = board.map.edge_to_nodes[edge]
            
            for node_id in connected_nodes:
                # If the node is empty and buildable in the future
                if (node_id not in board.buildings and 
                    self.is_potential_settlement_spot(game, node_id, player_color)):
                    # Add value based on the potential settlement location
                    potential_value = self.evaluate_settlement_location(game, node_id, player_color)
                    value += min(potential_value / 5, 10)  # Cap the bonus to avoid extreme values
            
            return value
        except Exception as e:
            print(f"Error evaluating road location: {e}")
            return 0
    
    def is_potential_settlement_spot(self, game, node_id, player_color):
        """Check if a node could become a settlement spot in the future."""
        try:
            # Check distance rule - no adjacent settlements
            board = game.state.board
            for adjacent_node in board.map.adjacent_nodes[node_id]:
                if adjacent_node in board.buildings:
                    return False
            
            # Check if we have or could have a road connection
            # This is a simplified check - if any adjacent edge could be our road
            for edge in board.map.node_to_edges[node_id]:
                if (edge in board.roads and board.roads[edge] == player_color):
                    return True
                
                # If no road yet but could potentially build one
                for adjacent_edge in board.map.node_to_edges[node_id]:
                    for node in board.map.edge_to_nodes[adjacent_edge]:
                        if node in board.buildings and board.buildings[node][0] == player_color:
                            return True
            
            return False
        except Exception as e:
            print(f"Error checking potential settlement spot: {e}")
            return False