import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from base_llm import OpenAILLM, AzureOpenAILLM, MistralLLM, BaseLLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from enum import Enum
from io import StringIO
from datetime import datetime

from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import CatanMap, LandTile, Port
from catanatron.models.decks import freqdeck_count
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
    get_player_freqdeck,
    get_longest_road_length,
    get_largest_army,
)
from catanatron.state import State

# Constants for pretty printing
RESOURCE_EMOJI = {
    "WOOD": "🌲",
    "BRICK": "🧱",
    "SHEEP": "🐑",
    "WHEAT": "🌾",
    "ORE": "⛏️",
    None: "🏜️",
}

BUILDING_EMOJI = {
    "SETTLEMENT": "🏠",
    "CITY": "🏙️",
    "ROAD": "🛣️",
}

COSTS = {
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "WHEAT": 1, "SHEEP": 1},
    "CITY": {"WHEAT": 2, "ORE": 3},
    "DEVELOPMENT_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}

DEV_CARD_DESCRIPTIONS = {
    "KNIGHT": "Move the robber and steal a card from a player adjacent to the new location",
    "YEAR_OF_PLENTY": "Take any 2 resources from the bank",
    "MONOPOLY": "Take all resources of one type from all other players",
    "ROAD_BUILDING": "Build 2 roads for free",
    "VICTORY_POINT": "Worth 1 victory point",
}

class PromptEvolverPlayer(Player):
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    debug_mode = True
    run_dir = None

    def __init__(self, color, name=None, llm=None):
        super().__init__(color, name)
        # Get API key from environment variable
        if llm is None:
            self.llm = AzureOpenAILLM(model_name="gpt-4o")
            #self.llm = MistralLLM(model_name="mistral-large-latest")
            #self.llm = AnthropicLLM()
        else:
            self.llm = llm
        self.is_bot = True
        self.llm_name = self.llm.model

        # Use the shared run directory from the creator agent if available
        if PromptEvolverPlayer.run_dir is None:
            if "CATAN_CURRENT_RUN_DIR" in os.environ:
                # Use the directory passed from the creator agent
                PromptEvolverPlayer.run_dir = os.environ["CATAN_CURRENT_RUN_DIR"]
            else:
                # Create a new run directory if not running through creator agent
                agent_dir = os.path.dirname(os.path.abspath(__file__))
                runs_dir = os.path.join(agent_dir, "runs")
                os.makedirs(runs_dir, exist_ok=True)
                
                # Try to read current_run.txt for the last known run
                try:
                    with open(os.path.join(runs_dir, "current_run.txt"), "r") as f:
                        PromptEvolverPlayer.run_dir = f.read().strip()
                except:
                    # Fallback to creating a new run directory
                    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                    PromptEvolverPlayer.run_dir = os.path.join(runs_dir, run_id)
                    os.makedirs(PromptEvolverPlayer.run_dir, exist_ok=True)

        # Initialize stats for the player
        self.api_calls = 0
        self.api_tokens_used = 0
        self.decision_times = []

        # Initialize resource tracking and planning
        self.resource_history = []
        self.last_resources = None
        self.current_plan = None
        self.last_turn_number = 0

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Use Claude API to analyze game state and choose an action.

        Args:
            game (Game): Complete game state (read-only)
            playable_actions (List[Action]): Available actions to choose from

        Returns:
            action (Action): Chosen action from playable_actions
        """

        start_time = time.time()
        state = game.state

        # Track resources and update resource history
        self._update_resource_tracking(state)

        if len(playable_actions) == 1:
            if self.debug_mode:
                print(f"Turn {state.num_turns}: Only one action available. Automatically selecting: {self._get_action_description(playable_actions[0])}")
            return playable_actions[0]

        # Create a string representation of the game state for Claude
        game_state_text = self._format_game_state_for_llm(game, state, playable_actions)

        # print(game_state_text)

        if self.debug_mode:
            pass
            #print(f"Game state prepared for LLM (length: {len(game_state_text)} chars)")

        # Use LLM to choose an action
        try:
            chosen_action_idx = self._get_llm_decision(game_state_text, len(playable_actions))
            if chosen_action_idx is not None and 0 <= chosen_action_idx < len(playable_actions):
                action = playable_actions[chosen_action_idx]
                if self.debug_mode:
                    print(f"Turn {state.num_turns}: LLM chose action {chosen_action_idx}: {self._get_action_description(action)}")

                # Record decision time
                decision_time = time.time() - start_time
                self.decision_times.append(decision_time)

                return action
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting LLM decision: {e}")
                print("Falling back to rule-based strategy")

        # Fallback to rule-based selection if API call fails or no API key
        return self._select_action(playable_actions, state)

    def _update_resource_tracking(self, state):
        """Track resource changes between turns to provide history to Claude."""
        # Get current resources
        current_resources = get_player_freqdeck(state, self.color)

        # Initialize last_resources on first call
        if self.last_resources is None:
            self.last_resources = current_resources
            return

        # Calculate changes since last turn
        changes = {}
        for i, resource in enumerate(RESOURCES):
            diff = current_resources[i] - self.last_resources[i]
            if diff != 0:
                changes[resource] = diff

        # If there were changes, record them in history
        if changes:
            turn_record = {
                "turn": state.num_turns,
                "changes": changes
            }
            self.resource_history.append(turn_record)

            # Keep history at a reasonable size (last 5 turns)
            if len(self.resource_history) > 5:
                self.resource_history = self.resource_history[-5:]

        # Update for next time
        self.last_resources = current_resources

    def _format_game_state_for_llm(self, game: Game, state: State, playable_actions: List[Action]) -> str:
        """Format game state as a string for the LLM API.

        This creates a concise text representation with only information a human player would have.
        Includes resource history and strategic plan.
        """
        # Use StringIO to build the text efficiently
        from io import StringIO
        output = StringIO()

        # Game header
        output.write(f"TURN {state.num_turns} | Player: {state.current_color().name} | Action: {state.current_prompt.name}\n\n")

        # Include current plan if available
        if self.current_plan:
            output.write("🎯 YOUR CURRENT PLAN:\n")
            output.write(f"  {self.current_plan}\n")
            output.write("  To continue with this plan, select the appropriate action.\n")
            output.write("  To adjust your plan, provide a NEW PLAN in your reasoning with <plan>your new plan</plan>\n\n")

        # Board state (only include information relevant to decision making)
        self._format_board_state(output, state)

        # Player states (showing only public information for other players)
        self._format_player_states(output, state)

        # Available actions (grouped by type for easier decision making)
        self._format_available_actions(output, playable_actions)

        return output.getvalue()

    def _get_llm_decision(self, game_state_text: str, num_actions: int) -> Optional[int]:
        """Send game state to OpenAI LLM and get the selected action index.

        Args:
            game_state_text: Formatted game state text to send to the LLM
            num_actions: Number of available actions (for validation)

        Returns:
            int: Index of the selected action, or None if API call fails
        """

        # Load prompt template from file
        prompt_path = pathlib.Path(__file__).parent / "current_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        # Append the game state and final instruction lines
        added_text = (
            f"\n\n{game_state_text}\n\n"
            "Based on this information, which action number do you choose? "
            "Think step by step about your options, then put the final action number in a box like \\boxed{1}."
        )
        prompt_template += added_text
        prompt = prompt_template
    
        try:
            response = self.llm.query(prompt)

            # Gather plan from the response
            self._extract_plan_from_response(response)

            # Find the current game run directory (should be the most recent one)
            game_run_dirs = [d for d in pathlib.Path(PromptEvolverPlayer.run_dir).glob("game_*") if d.is_dir()]
            if game_run_dirs:
                # Sort by creation time (newest first)
                latest_game_run = max(game_run_dirs, key=lambda d: d.stat().st_mtime)
                log_path = latest_game_run / f"llm_log_{self.llm_name}.txt"
            else:
                # Fallback to main run directory if no game subdirectories exist
                log_path = pathlib.Path(PromptEvolverPlayer.run_dir) / f"llm_log_{self.llm_name}.txt"
                
            with open(log_path, "a") as log_file:
                log_file.write("PROMPT:\n")
                log_file.write(prompt + "\n")
                log_file.write("RESPONSE:\n")
                log_file.write(str(response) + "\n")
                log_file.write("="*40 + "\n")

            # Extract the first integer from a boxed answer or any number
            import re
            boxed_match = re.search(r'\\boxed\{(\d+)\}', str(response))
            if boxed_match:
                idx = int(boxed_match.group(1))
                if 0 <= idx < num_actions:
                    return idx
            # Fallback: look for any number
            numbers = re.findall(r'\d+', str(response))
            if numbers:
                idx = int(numbers[0])
                if 0 <= idx < num_actions:
                    return idx
                
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error calling LLM: {e}")
        return None

    def _select_action(self, playable_actions: List[Action], state: State) -> Action:
        """Fallback strategy when Claude API is not available or fails.

        This is a rule-based approach that prioritizes actions according to
        common Catan strategies.
        """
        # First, handle mandatory actions
        if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            # For initial settlements, prioritize diversity and high probability numbers
            # Just pick first option for simplicity in this fallback
            return playable_actions[0]

        elif state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            # For initial roads, connect to the settlement just placed
            return playable_actions[0]

        elif state.current_prompt == ActionPrompt.DISCARD:
            # For discarding, try to keep a balanced hand
            return playable_actions[0]

        elif state.current_prompt == ActionPrompt.MOVE_ROBBER:
            # For moving the robber, target the leading player
            return playable_actions[0]

        # For regular turns
        elif state.current_prompt == ActionPrompt.PLAY_TURN:
            # Must roll if we haven't yet
            roll_actions = [a for a in playable_actions if a.action_type == ActionType.ROLL]
            if roll_actions:
                return roll_actions[0]

            # Use a simple priority-based strategy
            action_priority = [
                ActionType.BUILD_CITY,         # Cities provide more resources
                ActionType.BUILD_SETTLEMENT,   # New settlements expand resource collection
                ActionType.BUY_DEVELOPMENT_CARD,  # Development cards for versatility
                ActionType.PLAY_KNIGHT_CARD,   # Knights help with largest army
                ActionType.BUILD_ROAD,         # Roads for expansion
                ActionType.MARITIME_TRADE,     # Trading as needed
                ActionType.END_TURN            # End turn as last resort
            ]

            for action_type in action_priority:
                actions = [a for a in playable_actions if a.action_type == action_type]
                if actions:
                    return actions[0]

        # Fallback to random choice if no strategy applies
        if self.debug_mode:
            print("Note: Using random selection as last resort")
        return random.choice(playable_actions)

    def _format_board_state(self, output, state: State) -> None:
        """Format the board state as text for LLM consumption."""
        board = state.board
        COLOR_MARKERS = {
            "RED": "🔴",
            "BLUE": "🔵",
            "WHITE": "⚪",
            "ORANGE": "🟠"
        }

        # Game status
        output.write("📊 GAME STATUS:\n")
        output.write(f"  Longest Road: {board.road_color.name if board.road_color else 'None'} ({board.road_length} segments)\n")
        largest_army_color, largest_army_size = get_largest_army(state)
        output.write(f"  Largest Army: {largest_army_color.name if largest_army_color else 'None'} ({largest_army_size or 0} knights)\n")
        output.write(f"  Robber Location: {board.robber_coordinate}\n\n")

        # Generate programmatic grid representation of the board
        self._format_grid_representation(output, board)

        # Buildings with detailed production information
        output.write("\nBUILDINGS:\n")
        settlements_by_color = {}
        for node_id, (color, building_type) in board.buildings.items():
            if color not in settlements_by_color:
                settlements_by_color[color] = []

            # Find adjacent resources for context
            adjacent_tiles = []
            if node_id in board.map.adjacent_tiles:
                for tile in board.map.adjacent_tiles[node_id]:
                    if tile.resource is not None:  # Skip desert
                        # Add probability stars
                        stars = ""
                        if tile.number in [6, 8]:
                            stars = "[⭐⭐⭐]"  # Highest probability
                        elif tile.number in [5, 9]:
                            stars = "[⭐⭐]"    # Good probability
                        elif tile.number in [4, 10, 3, 11]:
                            stars = "[⭐]"     # Lower probability

                        tile_str = f"{RESOURCE_EMOJI[tile.resource]}{tile.number}{stars}"
                        adjacent_tiles.append(tile_str)

            tiles_str = ", ".join(adjacent_tiles) if adjacent_tiles else "no resources"
            building_str = f"{COLOR_MARKERS[color.name]} {color.name}: {BUILDING_EMOJI[building_type]} at node {node_id} (produces: {tiles_str})"
            settlements_by_color[color].append((building_type, building_str))

        # Add buildings by player
        for color, buildings in settlements_by_color.items():
            for building_type, building_str in buildings:
                output.write(f"- {building_str}\n")

        # Roads (show actual connections for better understanding)
        output.write("\nROADS:\n")
        roads_by_color = {}
        for edge, color in board.roads.items():
            if color not in roads_by_color:
                roads_by_color[color] = []
            roads_by_color[color].append(edge)

        for color, roads in roads_by_color.items():
            output.write(f"- {COLOR_MARKERS[color.name]} {color.name}: ")
            if len(roads) <= 8:  # Show detailed connections if not too many
                road_strs = [f"{edge}" for edge in roads]
                output.write(f"{', '.join(road_strs)}\n")
            else:
                output.write(f"{len(roads)} roads (too many to list)\n")

        # Ports (only include accessible ports that matter for decision-making)
        output.write("\nPORTS:\n")
        for resource, node_ids in board.map.port_nodes.items():
            if resource:
                # Check if any player has access to this port
                access_info = ""
                for node_id in node_ids:
                    if node_id in board.buildings:
                        color, _ = board.buildings[node_id]
                        access_info = f" ({COLOR_MARKERS[color.name]} has access)"
                        break

                output.write(f"- 2:1 {RESOURCE_EMOJI[resource]} {resource} Port: nodes {node_ids}{access_info}\n")
            else:
                output.write(f"- 3:1 General Port: nodes {node_ids}\n")

    def _format_grid_representation(self, output, board):
        """Generate a programmatic grid representation of the Catan board."""
        output.write("🗺️ CATAN RESOURCE & NODE GRID:\n\n")

        # Group tiles by "tiers" or rows for better visualization
        # This simplifies the hexagonal grid into a more readable format
        tiers = {}

        # First, organize tiles by their y-coordinate (roughly corresponds to rows)
        for coord, tile in board.map.land_tiles.items():
            # Extract y from cube coordinates (x,y,z)
            y = coord[1]

            if y not in tiers:
                tiers[y] = []

            tiers[y].append((coord, tile))

        # Sort tiers by y-coordinate
        for y in sorted(tiers.keys()):
            # Sort tiles within each tier by x-coordinate
            tiles_in_tier = sorted(tiers[y], key=lambda item: item[0][0])

            row_output = ""
            nodes_output = ""

            for coord, tile in tiles_in_tier:
                # Mark robber
                robber_marker = " 🔍" if coord == board.robber_coordinate else ""

                # Format tile with resource emoji, number, and coordinates
                number_str = str(tile.number) if tile.number is not None else "-"
                resource_emoji = RESOURCE_EMOJI[tile.resource]

                # Format the tile information with fixed width for alignment
                tile_info = f"{resource_emoji}{number_str} {coord}{robber_marker}".ljust(30)
                row_output += tile_info

                # Find all nodes connected to this tile
                connected_nodes = []
                # In this code, we need to find nodes connected to the current tile
                # We need to use the tile's nodes dictionary directly
                if isinstance(tile, LandTile) or isinstance(tile, Port):
                    connected_nodes = sorted(tile.nodes.values())

                # Format node information with fixed width
                if connected_nodes:
                    nodes_str = f"nodes: {','.join(map(str, sorted(connected_nodes)))}".ljust(30)
                    nodes_output += nodes_str

            # Add the formatted tier to the output
            output.write(f"  {row_output}\n")
            output.write(f"  {nodes_output}\n\n")

    def _format_player_states(self, output, state: State) -> None:
        """Format player states as text for LLM consumption, respecting hidden information."""
        output.write("\n👥 PLAYERS:\n")

        for color in state.colors:
            hand = get_player_freqdeck(state, color)
            key = player_key(state, color)
            vp = state.player_state.get(f"{key}_VICTORY_POINTS", 0)
            longest_road = get_longest_road_length(state, color)
            settlements = get_player_buildings(state, color, SETTLEMENT)
            cities = get_player_buildings(state, color, CITY)

            # Determine if this is the current player
            is_current = color == state.current_color()
            current_marker = "👉 " if is_current else ""

            # Player header
            output.write(f"\n  {current_marker}{color.name} Player:\n")
            output.write(f"    Victory Points: {vp}/10\n")

            # Resource cards - only show for current player
            resource_list = []
            for i, resource in enumerate(RESOURCES):
                count = hand[i] if i < len(hand) else 0
                if count > 0:
                    resource_list.append(f"{count} {RESOURCE_EMOJI[resource]}{resource}")

            # Only show the current player's resources
            if is_current:
                output.write(f"    Resources: {', '.join(resource_list) if resource_list else 'None'}\n")
            else:
                output.write(f"    Resources: {sum(hand)} cards (hidden)\n")

            # Development cards - only show details for current player
            dev_card_count = player_num_dev_cards(state, color)
            if is_current:
                # Show current player's development cards
                dev_cards = {}
                played_dev_cards = {}
                dev_card_types = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]

                # Check for unplayed development cards
                for card_type in dev_card_types:
                    card_key = f"{key}_{card_type}_IN_HAND"
                    count = state.player_state.get(card_key, 0)
                    if count > 0:
                        dev_cards[card_type] = count

                # Check for played cards
                for card_type in dev_card_types:
                    # if card_type == "VICTORY_POINT":
                    #     continue  # VP cards aren't played
                    card_key = f"{key}_PLAYED_{card_type}"
                    count = state.player_state.get(card_key, 0)
                    if count > 0:
                        played_dev_cards[card_type] = count


                # dev_cards = state.player_state.get(f"{key}_DEVELOPMENT_CARDS", {})
                # played_dev_cards = state.player_state.get(f"{key}_PLAYED_DEV_CARDS", {})

                # Format available dev cards
                available_cards = []
                for card_type, count in dev_cards.items():
                    if count > 0:
                        available_cards.append(f"{count} {card_type}")

                # Format played dev cards
                played_cards = []
                for card_type, count in played_dev_cards.items():
                    if count > 0 and card_type != "VICTORY_POINT":  # VP cards aren't visibly played
                        played_cards.append(f"{count} {card_type}")

                output.write(f"    Development Cards: {', '.join(available_cards) if available_cards else 'None'}\n")

                if played_cards:
                    output.write(f"    Played Cards: {', '.join(played_cards)}\n")
            else:
                # For other players, just show the count but check the actual keys
                played_dev_cards = {}
                dev_cards = {}
                dev_card_types = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]

                 # Check for unplayed development cards
                dev_card_count = 0
                for card_type in dev_card_types:
                    card_key = f"{key}_{card_type}_IN_HAND"
                    count = state.player_state.get(card_key, 0)
                    if count > 0:
                        dev_cards[card_type] = count
                        dev_card_count += count

                
                # Check for played cards
                for card_type in dev_card_types:
                    card_key = f"{key}_PLAYED_{card_type}"
                    count = state.player_state.get(card_key, 0)
                    if count > 0:  # VP cards aren't visibly played
                        played_dev_cards[card_type] = count
                            
                # Show total dev cards (hidden from other players)
                if dev_card_count > 0:
                    output.write(f"    Development Cards: {dev_card_count} (hidden)\n")
                else:
                    output.write(f"    Development Cards: None\n")
                
                # Show all played cards (these are public information)
                played_cards = []
                for card_type, count in played_dev_cards.items():
                    if count > 0:
                        played_cards.append(f"{count} {card_type}")
                
                if played_cards:
                    output.write(f"    Played Cards: {', '.join(played_cards)}\n")

            # Building information (public)
            output.write(f"    Buildings:\n")
            output.write(f"      🏠 Settlements: {len(settlements)} \n")
            output.write(f"      🏙️ Cities: {len(cities)} \n")

            roads_available = state.player_state.get(f'{key}_ROADS_AVAILABLE', 0)
            roads_built = 15 - roads_available
            output.write(f"      🛣️ Roads: {roads_built} built (Longest: {longest_road})\n")

            # Special achievements
            if state.board.road_color == color:
                output.write(f"      ✅ Has Longest Road bonus (+2 VP)\n")

            largest_army_color, largest_army_size = get_largest_army(state)
            if largest_army_color == color:
                output.write(f"      ✅ Has Largest Army bonus (+2 VP)\n")

        # Bank information (public)
        output.write("\n  🏦 Bank:\n")
        dev_cards_left = len(state.development_listdeck)
        output.write(f"    Development Cards: {dev_cards_left} remaining\n")

        # Resource availability (only data relevant to decisions)
        output.write(f"    Resources available for trade with bank or ports\n")

    def _format_available_actions(self, output, playable_actions: List[Action]) -> None:
        """Format available actions as text for LLM consumption."""
        output.write("\n🎮 AVAILABLE ACTIONS:\n")

        # Group similar actions for better readability
        action_groups = {}
        for i, action in enumerate(playable_actions):
            action_type = action.action_type.name
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append((i, action))

        # Add actions by type for better organization
        for action_type, actions in action_groups.items():
            # Add a header for each action type
            header = action_type.replace("_", " ").title()
            if len(actions) > 1:
                output.write(f"\n  {header} Options:\n")
            else:
                output.write(f"\n  {header}:\n")

            # Add individual actions
            for i, action in actions:
                action_description = self._get_action_description(action)
                output.write(f"    [{i}] {action_description}\n")

        output.write("\n❓ Select action by number (0-{}):\n".format(len(playable_actions)-1))

    def _get_action_description(self, action: Action) -> str:
        """Get a clear, descriptive explanation of an action."""
        action_type = action.action_type
        value = action.value
        color = action.color

        # Standard descriptions for common actions
        descriptions = {
            ActionType.ROLL: "Roll the dice",
            ActionType.END_TURN: "End your turn",
            ActionType.BUY_DEVELOPMENT_CARD: f"Buy a development card ({self._format_cost('DEVELOPMENT_CARD')})",
            ActionType.PLAY_KNIGHT_CARD: "Play Knight card - Move the robber and steal a resource",
            ActionType.PLAY_YEAR_OF_PLENTY: f"Play Year of Plenty card - Take two resources: {value}",
            ActionType.PLAY_MONOPOLY: f"Play Monopoly card - Take all {value} from other players",
            ActionType.PLAY_ROAD_BUILDING: "Play Road Building card - Build two roads for free",
        }

        # Return from dictionary if action type is in it
        if action_type in descriptions:
            return descriptions[action_type]

        if action_type == ActionType.BUILD_ROAD:
            return f"Build a road at edge {value} (Cost: {self._format_cost('ROAD')})"
        elif action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build a settlement at node {value} (Cost: {self._format_cost('SETTLEMENT')})"
        elif action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {value} (Cost: {self._format_cost('CITY')})"
        elif action_type == ActionType.MOVE_ROBBER:
            target_color = value[1]
            target_str = f" and steal from {target_color.name}" if target_color else ""
            return f"Move robber to {value[0]}{target_str}"
        elif action_type == ActionType.MARITIME_TRADE:
            trade_resources = value
            offering = []

            # Check if trade_resources contains string values (resource names)
            if isinstance(trade_resources[0], str):
                # Count occurrences of each resource
                resource_counts = {}
                for resource in trade_resources[:4]:
                    if resource not in resource_counts:
                        resource_counts = {}
                    resource_counts[resource] = resource_counts.get(resource, 0) + 1

                # Create the offering text
                for resource, count in resource_counts.items():
                    offering.append(f"{count} {RESOURCE_EMOJI[resource]}{resource}")

                # Determine what's being received
                receiving = trade_resources[4] if len(trade_resources) > 4 else None
                receiving_emoji = RESOURCE_EMOJI.get(receiving, "")
            else:
                # Original code for when trade_resources contains integer counts
                for i, count in enumerate(trade_resources[:4]):
                    if isinstance(count, int) and count > 0:
                        resource = RESOURCES[i]
                        offering.append(f"{count} {RESOURCE_EMOJI[resource]}{resource}")

                receiving_idx = next((i for i in range(4, len(trade_resources)) if trade_resources[i] > 0), None)
                receiving = RESOURCES[receiving_idx-4] if receiving_idx is not None else None
                receiving_emoji = RESOURCE_EMOJI.get(receiving, "")

            return f"Trade port: Give {', '.join(offering)} for 1 {receiving_emoji}{receiving}"
        elif action_type == ActionType.DISCARD:
            if value is None:
                return "Discard half your cards (robber)"
            else:
                discard_str = ", ".join([f"{count} {RESOURCE_EMOJI[res]}{res}" for res, count in zip(RESOURCES, value) if count > 0])
                return f"Discard: {discard_str}"
        elif action_type == ActionType.ACCEPT_TRADE:
            offering = ", ".join([f"{count} {RESOURCE_EMOJI[res]}{res}" for res, count in zip(RESOURCES, value[:5]) if count > 0])
            receiving = ", ".join([f"{count} {RESOURCE_EMOJI[res]}{res}" for res, count in zip(RESOURCES, value[5:]) if count > 0])
            return f"Accept trade: Give {offering}, Receive {receiving}"
        elif action_type == ActionType.REJECT_TRADE:
            return "Reject the proposed trade"
        else:
            return f"{action_type.name.replace('_', ' ').title()}: {value}"

    def _format_cost(self, item_type: str) -> str:
        """Format the cost of an item in a clear, concise way."""
        cost_items = []
        for resource, count in COSTS[item_type].items():
            cost_items.append(f"{count} {RESOURCE_EMOJI[resource]}{resource}")
        return ", ".join(cost_items)

    def _extract_plan_from_response(self, content: str) -> None:
        """Extract plan from Claude's response and update tracking."""
        import re

        # Extract plan if present
        plan_match = re.search(r'<plan>(.*?)</plan>', content, re.DOTALL)
        if plan_match:
            self.current_plan = plan_match.group(1).strip()
            if self.debug_mode:
                pass
                # print(f"Updated plan: {self.current_plan}")



def substitute_variable(template: str, variable: str, value: str) -> str:
    import re
    """
    Replace only the {variable} placeholder in the template with value.
    Leaves other curly braces (like \boxed{5}) untouched.
    """
    pattern = r'\{' + re.escape(variable) + r'\}'
    return re.sub(pattern, value, template)