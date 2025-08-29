import os
import re
import traceback
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
        
        # Detect if we're in initial placement phase using extremely simple, reliable methods
        is_initial_placement = self.is_initial_placement_phase(game)
        
        try:
            if is_initial_placement:
                print("Initial placement phase detected - using simplified approach")
                # Use the simplest, most reliable implementation for initial placement
                return self.handle_initial_placement(game, playable_actions)
            else:
                # Normal game phase - use regular state representation
                game_state = self._get_game_state_representation(game)
        except Exception as e:
            print(f"Error getting game state: {str(e)}")
            traceback.print_exc()
            game_state = "Error getting detailed game state."
        
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
        
        # Create prompt for regular game phase
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

    def handle_initial_placement(self, game, playable_actions):
        """
        Ultra-simple initial settlement placement that works with integer node IDs.
        No complex board analysis, just basic selection.
        """
        # First check if these are settlement actions
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        if settlement_actions:
            print(f"Found {len(settlement_actions)} settlement actions")
            
            # Use LLM for initial placement if we can
            try:
                action_descriptions = []
                for i, action in enumerate(settlement_actions):
                    action_descriptions.append(f"Action {i}: Build settlement at node {action.value}")
                    
                prompt = f"""
                INITIAL SETTLEMENT PLACEMENT DECISION
                
                You are placing an initial settlement in Catan. This is a critical strategic decision.
                Select a location based on these principles:
                1. High-probability numbers (6,8,5,9) are more valuable
                2. Diverse resources are important (aim for 3+ different types)
                3. First settlement: prioritize Wood/Brick for early road building
                4. Second settlement: complement your first settlement's resources
                
                Available settlement locations:
                {chr(10).join(action_descriptions)}
                
                Carefully analyze the options and select the best settlement location.
                End with "Best Action: X" where X is the action number.
                """
                
                response = self.llm.query_llm(prompt)
                print(f"LLM Response for initial placement: {response[:100]}...")
                match = re.search(r"Best Action: (\d+)", response)
                if match:
                    chosen_index = int(match.group(1))
                    if 0 <= chosen_index < len(settlement_actions):
                        print(f"LLM chose settlement at node {settlement_actions[chosen_index].value}")
                        return settlement_actions[chosen_index]
            except Exception as e:
                print(f"Error using LLM for placement: {e}")
                traceback.print_exc()
                # Continue to fallback
            
            # Ultra-simple fallback: pick the first settlement action
            print("Falling back to first settlement action")
            return settlement_actions[0]
        
        # For road actions - just pick the first one
        road_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
        if road_actions:
            print("Choosing first road placement action")
            return road_actions[0]
        
        # Absolute fallback
        print("Falling back to first action")
        return playable_actions[0]

    def is_initial_placement_phase(self, game):
        """
        Ultra-reliable detection of initial placement phase using multiple signals.
        Uses only the most basic API calls that are unlikely to change.
        """
        try:
            # Method 1: Check game turn count (most reliable)
            if hasattr(game, 'state') and hasattr(game.state, 'num_turns'):
                if game.state.num_turns < len(game.state.colors) * 2:
                    print("Initial placement detected based on turn count")
                    return True
            
            # Method 2: Check if player has 0 or 1 settlements (also reliable)
            player_settlements = game.state.buildings_by_color.get(self.color, {}).get("SETTLEMENT", [])
            if len(player_settlements) <= 1:
                print("Initial placement detected based on settlement count")
                return True
            
            # Method 3: Check action types only (fall back method)
            settlement_road_only = True
            for action in game.state.playable_actions:
                if action.action_type not in [ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD]:
                    settlement_road_only = False
                    break
            
            if settlement_road_only and len(game.state.playable_actions) > 0:
                print("Initial placement detected based on available actions")
                return True
            
            return False
        except Exception as e:
            print(f"Error in is_initial_placement_phase: {str(e)}")
            traceback.print_exc()
            # Ultra-safe fallback - if in doubt, assume it's not initial placement
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
            # For initial placements, add INITIAL label
            if self.is_initial_placement_phase(game):
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