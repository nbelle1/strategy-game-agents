import os
import re
import traceback
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES, ActionType
from catanatron.models.enums import ActionPrompt  # Added ActionPrompt for reliable initial placement detection
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
        
        # Detect if we're in initial placement phase using the reliable detection method
        is_initial_placement = self.is_initial_placement_phase(game)
        
        if is_initial_placement:
            print("Initial placement phase detected - using specialized analysis")
            game_state = self._get_initial_placement_state(game, playable_actions)
        else:
            # Normal game phase - use regular state representation
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
        
        # Get player's current VP to determine game phase
        player_index = game.state.color_to_index[self.color]
        key = f"P{player_index}"
        victory_points = game.state.player_state.get(f"{key}_VICTORY_POINTS", 0)
        
        # Determine game phase based on victory points
        game_phase = "EARLY"
        if victory_points >= 7:
            game_phase = "LATE"
        elif victory_points >= 4:
            game_phase = "MID"
        
        # Create different prompts based on game phase
        if is_initial_placement:
            prompt = f"""
            INITIAL PLACEMENT PHASE - CRITICAL DECISION
            
            Current Board Analysis:
            {game_state}
            
            Available Actions:
            {chr(10).join(action_descriptions)}
            
            Initial settlement placement is the MOST IMPORTANT decision in the game!
            
            Evaluation Guidelines for Initial Placements:
            1. High probability numbers (6,8,5,9) are most valuable
            2. Diverse resources are critical (aim for at least 3 different resources)
            3. Access to scarce resources (Brick and Wood for early game, Ore and Wheat for late game)
            4. Port access provides long-term trading benefits
            5. Consider your second settlement location (try to complement resources)
            
            For each potential settlement location, calculate:
            - Production Value (0-10): Sum of probability weights of adjacent tiles
            - Resource Diversity (0-5): Number of different resources accessible
            - Scarcity Value (0-5): Access to rare resources on the board
            - Future Expansion (0-5): Available adjacent nodes for roads/settlements
            
            Choose the action with the highest total score.
            Explicitly state your reasoning and numerical scores.
            End with "Best Action: X" where X is the action number.
            """
        else:
            prompt = f"""
            Current Game State:
            {game_state}
            
            Available Actions:
            {chr(10).join(action_descriptions)}
            
            Game Phase: {game_phase}
            
            STRATEGIC PRIORITIES:
            1. EARLY GAME (1-3 VP): Secure Wood/Brick for roads, expand to diverse resources
            2. MID GAME (4-6 VP): Upgrade to cities (wheat/ore) and buy development cards
            3. LATE GAME (7+ VP): Block opponent paths, focus on direct VP gains
            
            Resource Priorities:
            - Road Building: Wood + Brick
            - Settlements: Wood + Brick + Wheat + Sheep 
            - Cities: Wheat + Ore (highest value upgrade - 2 VP for 2 Wheat + 3 Ore)
            - Development Cards: Wheat + Sheep + Ore (25% chance of direct VP)
            
            Action Valuation Guidelines:
            - Building settlements on high-probability tiles (6,8,5,9) gets +2 priority
            - Upgrading settlements to cities gets +3 priority in mid-game
            - Blocking opponent's expansion paths gets +2 priority when they're 2+ VP ahead
            - Maritime trading for scarce resources gets +1 priority when you have 4+ of another resource
            - Development cards get +1 priority when you need defensive options
            
            Evaluate actions with numerical scores considering:
            1. Immediate VP gain (0-5 points)
            2. Resource generation improvement (0-5 points)
            3. Strategic positioning (0-5 points)
            4. Opponent blocking value (0-5 points)
            5. Progress toward next VP (0-5 points)
            
            Choose the action with the highest total score.
            Explicitly state your reasoning and numerical scores.
            End with "Best Action: X" where X is the action number.
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

    def is_initial_placement_phase(self, game):
        """
        Reliable detection of initial placement phase using the ActionPrompt.
        
        Args:
            game (Game): The current game state
        Returns:
            bool: True if in initial placement phase, False otherwise
        """
        # The most reliable way to check initial placement is using the ActionPrompt
        try:
            return (game.state.prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT or
                    game.state.prompt == ActionPrompt.BUILD_INITIAL_ROAD)
        except (AttributeError, NameError):
            # Fallback method if ActionPrompt isn't accessible
            
            # Method 1: Check game turn count vs player count
            # Initial placement happens during first N*2 turns (N=number of players)
            if game.state.num_turns < len(game.state.colors) * 2:
                return True
            
            # Method 2: Check settlement count - in initial phase we have 0-1 settlements
            try:
                buildings = game.state.buildings_by_color.get(self.color, {})
                settlements = buildings.get("SETTLEMENT", [])
                if len(settlements) <= 1:
                    return True
            except Exception:
                pass  # Ignore errors in this check
                
            return False

    def _get_game_state_representation(self, game):
        """
        Generate an enhanced string representation of the current game state with strategic information.
        
        Args:
            game (Game): The current game state
        Returns:
            str: A human-readable representation of the game state with strategic insights
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
        
        # Calculate total resources and resource diversity
        total_resources = wood_count + brick_count + sheep_count + wheat_count + ore_count
        resource_types = sum(1 for count in [wood_count, brick_count, sheep_count, wheat_count, ore_count] if count > 0)
        
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
        
        # Resources with strategic information
        state_str += "Your Resources:\n"
        state_str += f"- WOOD: {wood_count} (needed for roads and settlements)\n"
        state_str += f"- BRICK: {brick_count} (needed for roads and settlements)\n"
        state_str += f"- SHEEP: {sheep_count} (needed for settlements and development cards)\n"
        state_str += f"- WHEAT: {wheat_count} (needed for settlements, cities, and development cards)\n"
        state_str += f"- ORE: {ore_count} (needed for cities and development cards)\n"
        state_str += f"Total Resources: {total_resources}, Resource Diversity: {resource_types}/5\n"
        
        # Resource combinations for strategic planning
        state_str += "Strategic Resource Combinations:\n"
        state_str += f"- Road Potential: {min(wood_count, brick_count)} roads buildable\n"
        state_str += f"- Settlement Potential: {min(wood_count, brick_count, wheat_count, sheep_count)} settlements buildable\n"
        state_str += f"- City Potential: {min(wheat_count // 2, ore_count // 3)} cities upgradable\n"
        state_str += f"- Development Card Potential: {min(wheat_count, sheep_count, ore_count)} cards purchasable\n"
        
        # Victory points with breakdown
        state_str += f"Your Victory Points: {victory_points}\n"
        state_str += f"VP Breakdown: {len(settlements)} from settlements + {len(cities) * 2} from cities + {dev_cards_victory} from VP cards\n"
        
        # Buildings
        state_str += f"Your Buildings: {len(settlements)} settlements, {len(cities)} cities, {len(roads)} roads\n"
        
        # Development cards
        state_str += f"Your Development Cards: Knights played: {dev_cards_knights}, Victory points: {dev_cards_victory}\n"
        
        # Game progress indicator
        state_str += f"Game Progress: "
        if victory_points <= 3:
            state_str += "EARLY GAME - Focus on expansion and resource diversity\n"
        elif victory_points <= 6:
            state_str += "MID GAME - Focus on city upgrades and development cards\n"
        else:
            state_str += "LATE GAME - Focus on direct VP and blocking opponent\n"
        
        # Enhanced opponent summary with VP gap analysis
        state_str += "Opponents:\n"
        max_opponent_vp = 0
        for color in game.state.colors:
            if color != self.color:
                opp_index = game.state.color_to_index[color]
                opp_key = f"P{opp_index}"
                opp_points = game.state.player_state.get(f"{opp_key}_VICTORY_POINTS", 0)
                if opp_points > max_opponent_vp:
                    max_opponent_vp = opp_points
                    
                opp_settlements = game.state.buildings_by_color.get(color, {}).get("SETTLEMENT", [])
                opp_cities = game.state.buildings_by_color.get(color, {}).get("CITY", [])
                opp_roads = game.state.buildings_by_color.get(color, {}).get("ROAD", [])
                state_str += f"- {color.value}: {opp_points} points, {len(opp_settlements)} settlements, {len(opp_cities)} cities, {len(opp_roads)} roads\n"
        
        # VP gap analysis
        vp_gap = max_opponent_vp - victory_points
        if vp_gap > 0:
            state_str += f"STRATEGIC ALERT: You are {vp_gap} VP behind the leading player!\n"
            if vp_gap >= 3:
                state_str += "URGENT: Need high-value moves to catch up quickly!\n"
        elif vp_gap < 0:
            state_str += f"STRATEGIC ADVANTAGE: You are {-vp_gap} VP ahead of all opponents.\n"
        
        # Game turn
        state_str += f"Current Turn: {game.state.num_turns}\n"
        
        return state_str

    def _get_initial_placement_state(self, game, playable_actions):
        """
        Generate a specialized game state representation for initial placement phase.
        Focuses on node values and resource distribution.
        
        Args:
            game (Game): The current game state
            playable_actions (list): Available actions
        Returns:
            str: Analysis for initial settlement placement
        """
        state_str = "INITIAL PLACEMENT ANALYSIS\n\n"
        
        # Filter settlement actions
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        
        if not settlement_actions:
            return "No settlement placement actions available.\n"
        
        # Board resource distribution
        state_str += "BOARD RESOURCE DISTRIBUTION:\n"
        try:
            # Count resources on the board
            resource_counts = {
                "WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0
            }
            
            probability_by_resource = {
                "WOOD": [], "BRICK": [], "SHEEP": [], "WHEAT": [], "ORE": []
            }
            
            # Analyze the board resource distribution
            for tile_coord, resource in game.state.board.map.tiles.items():
                resource_str = str(resource) if resource else "DESERT"
                
                # Skip desert tiles and non-resource tiles
                if resource_str == "DESERT" or resource_str == "None":
                    continue
                
                # Count this resource
                if resource_str in resource_counts:
                    resource_counts[resource_str] += 1
                    
                    # Get probability number
                    number = game.state.board.map.numbers.get(tile_coord)
                    if number:
                        probability_by_resource[resource_str].append(number)
            
            # Show resource distribution with probabilities
            for res, count in resource_counts.items():
                prob_str = str(probability_by_resource[res]) if probability_by_resource[res] else "None"
                state_str += f"- {res}: {count} tiles, Numbers: {prob_str}\n"
                
            # Identify scarce resources (less than 3 tiles)
            scarce = [res for res, count in resource_counts.items() if count < 3]
            if scarce:
                state_str += f"SCARCE RESOURCES: {', '.join(scarce)} - prioritize these!\n"
                
        except Exception as e:
            print(f"Error analyzing board resources: {str(e)}")
            traceback.print_exc()
            state_str += "Could not analyze board resource distribution.\n"
        
        # Define probabilities for reference
        state_str += "\nPLACEMENT GUIDELINES:\n"
        state_str += "- Number Probabilities: 6,8 (5/36), 5,9 (4/36), 4,10 (3/36), 3,11 (2/36), 2,12 (1/36)\n"
        state_str += "- Resource Priority: Brick & Wood for early game, Ore & Wheat for late game\n"
        state_str += "- Aim for at least 3 different resources across your settlements\n"
        
        # Rank settlement locations
        state_str += "\nSETTLEMENT LOCATION RANKINGS:\n"
        try:
            # Evaluate each settlement location
            settlement_values = []
            
            for action in settlement_actions:
                node_id = action.value
                
                # Use the verified node evaluation method
                node_value, resources_str = self.calculate_node_value(game, node_id)
                
                settlement_values.append({
                    "action_index": playable_actions.index(action),
                    "node_id": node_id,
                    "value": node_value,
                    "resources": resources_str
                })
            
            # Sort by value
            settlement_values.sort(key=lambda x: x["value"], reverse=True)
            
            # Show top settlement locations
            for i, settlement in enumerate(settlement_values):
                state_str += f"Rank {i+1}: Action {settlement['action_index']} - "
                state_str += f"Node {settlement['node_id']} - Score: {settlement['value']}\n"
                state_str += f"  Resources: {settlement['resources']}\n"
                
        except Exception as e:
            print(f"Error ranking settlements: {str(e)}")
            traceback.print_exc()
            state_str += "Error calculating settlement rankings. Use basic probabilities instead.\n"
        
        return state_str

    def calculate_node_value(self, game, node_id):
        """
        Calculate the value of a node for initial settlement placement.
        Uses verified Catanatron API calls.
        
        Args:
            game (Game): The current game state
            node_id: Node ID to evaluate
        Returns:
            tuple: (node_value, resources_string)
        """
        value = 0
        resources = []
        seen_resources = set()
        
        # Get adjacent tiles using the verified API method
        adjacent_tiles = game.state.board.map.get_adjacent_tiles(node_id)
        
        # Analyze each adjacent tile
        for tile_coord in adjacent_tiles:
            # Get resource type
            resource = game.state.board.map.tiles.get(tile_coord)
            
            # Skip desert/None resources
            if resource is None:
                continue
            
            resource_str = str(resource)
            if resource_str == "DESERT" or resource_str == "None":
                continue
            
            # Get probability number
            number = game.state.board.map.numbers.get(tile_coord)
            
            # Add to resources list
            resources.append(f"{resource_str}({number})")
            seen_resources.add(resource_str)
            
            # Calculate probability value
            if number:
                # Standard probability weights (6-abs(7-n))
                prob_value = 6 - abs(7 - number)
                value += prob_value
                
                # Bonus for especially good numbers (6,8)
                if number == 6 or number == 8:
                    value += 1
        
        # Resource diversity bonus
        value += len(seen_resources) * 2
        
        # Check for early game resources (wood/brick)
        if "WOOD" in seen_resources:
            value += 1
        if "BRICK" in seen_resources:
            value += 1
            
        # Check for port access
        try:
            for port_edge, port_data in game.state.board.map.ports.items():
                port_resource, port_nodes = port_data
                if node_id in port_nodes:
                    resources.append(f"PORT({port_resource or '3:1'})")
                    value += 1
                    break
        except Exception:
            pass  # Ignore errors in port detection
        
        # Return the value and resource description
        return value, ", ".join(resources)

    def _describe_action(self, action, game):
        """
        Generate a human-readable description of an action with strategic context.
        
        Args:
            action (Action): The action to describe
            game (Game): The current game state
        Returns:
            str: A human-readable description of the action with strategic context
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
            # For initial placements, add detailed node value information
            if self.is_initial_placement_phase(game):
                try:
                    node_id = action.value
                    
                    # Calculate node value using the verified method
                    value, resources_str = self.calculate_node_value(game, node_id)
                    
                    return f"Build INITIAL settlement at node {safe_value_str(node_id)} - Value: {value} - Resources: {resources_str}"
                except Exception as e:
                    print(f"Error describing initial settlement: {str(e)}")
                    return f"Build INITIAL settlement at node {safe_value_str(action.value)}"
            else:
                return f"Build settlement at node {safe_value_str(action.value)} (Grants 1 VP and resource production)"
        
        elif action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {safe_value_str(action.value)} (Adds 1 VP and doubles resource production)"
        
        elif action_type == ActionType.BUILD_ROAD:
            # For initial road placements, add context about it being initial
            if self.is_initial_placement_phase(game):
                return f"Build INITIAL road at edge {safe_value_str(action.value)}"
            else:
                return f"Build road at edge {safe_value_str(action.value)} (Enables future expansion)"
        
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buy development card (Could be Victory Point, Knight, Road Building, Year of Plenty, or Monopoly)"
        
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            # Safely handle Knight card action
            try:
                if action.value is None:
                    return "Play Knight card (Increases army size for Largest Army bonus)"
                elif isinstance(action.value, tuple) and len(action.value) > 0:
                    return f"Play Knight card and move robber to {safe_value_str(action.value[0])} (Increases army size and disrupts opponent)"
                else:
                    return f"Play Knight card with value {safe_value_str(action.value)} (Increases army size for Largest Army bonus)"
            except Exception:
                return "Play Knight card (error extracting details) (Increases army size for Largest Army bonus)"
        
        elif action_type == ActionType.END_TURN:
            return "End your turn (Pass to next player)"
        
        elif action_type == ActionType.ROLL:
            return "Roll the dice (Generate resources or trigger robber on 7)"
        
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
                
                return f"Move robber to {location} and steal from {target} (Blocks resource production and steals one resource)"
            except Exception:
                return "Move robber (error extracting details) (Blocks resource production)"
        
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            try:
                if action.value is None:
                    return "Play Year of Plenty card (Take any 2 resources from bank)"
                else:
                    return f"Play Year of Plenty card and take resources {safe_value_str(action.value)} (Gain specific resources)"
            except:
                return "Play Year of Plenty card (error extracting details) (Take any 2 resources from bank)"
        
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            try:
                if action.value is None:
                    return "Play Road Building card (Build 2 roads for free)"
                else:
                    return f"Play Road Building card and build roads at {safe_value_str(action.value)} (Build free roads at specific locations)"
            except:
                return "Play Road Building card (error extracting details) (Build 2 roads for free)"
        
        elif action_type == ActionType.PLAY_MONOPOLY:
            try:
                if action.value is None:
                    return "Play Monopoly card (Take all of one resource type from all players)"
                else:
                    return f"Play Monopoly card and take all {safe_value_str(action.value)} resources (Steal this resource from all players)"
            except:
                return "Play Monopoly card (error extracting details) (Take all of one resource type from all players)"
        
        elif action_type == ActionType.MARITIME_TRADE:
            try:
                if action.value is None:
                    return "Trade resources (Exchange resources with bank)"
                elif isinstance(action.value, tuple) and len(action.value) >= 2:
                    return f"Trade {safe_value_str(action.value[0])} for {safe_value_str(action.value[1])} (Convert excess resources to needed ones)"
                else:
                    return f"Trade with values {safe_value_str(action.value)} (Exchange resources)"
            except:
                return "Trade resources (error extracting details) (Exchange resources with bank)"
        
        elif action_type == ActionType.DISCARD:
            try:
                return f"Discard cards {safe_value_str(action.value)} (Mandatory discard due to robber)"
            except:
                return "Discard cards (error extracting details) (Mandatory discard due to robber)"
        
        # Fallback for any other action types
        else:
            try:
                return f"{action_type.name} - {safe_value_str(action.value)}"
            except:
                return str(action_type)