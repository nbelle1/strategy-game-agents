import os
import re
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
        
        # Create game state representation
        game_state = self._get_game_state_representation(game)
        
        # Create action descriptions
        action_descriptions = []
        for i, action in enumerate(playable_actions):
            action_descriptions.append(f"Action {i}: {self._describe_action(action, game)}")
        
        # Create the prompt for LLM
        prompt = f"""
        Current Game State:
        {game_state}
        
        Available Actions:
        {chr(10).join(action_descriptions)}
        
        Evaluate the above actions considering:
        1. Immediate resource gain
        2. Strategic positioning
        3. Blocking opponents
        4. Victory point potential
        
        Rank the top 3 actions from best to worst and explain your reasoning.
        Finally, output just the number of the best action (e.g., "Best Action: 2").
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

    def _get_game_state_representation(self, game):
        """
        Generate a string representation of the current game state.
        
        Args:
            game (Game): The current game state
        Returns:
            str: A human-readable representation of the game state
        """
        # Get player info
        player = game.state.players[self.color]
        
        # Get opponent info
        opponents = [p for c, p in game.state.players.items() if c != self.color]
        
        # Create string representation of game state
        state_str = f"Your Color: {self.color}\n"
        
        # Resources
        state_str += "Your Resources:\n"
        for resource in [WOOD, BRICK, SHEEP, WHEAT, ORE]:
            count = player.resource_deck.count(resource)
            state_str += f"- {resource}: {count}\n"
        
        # Victory points
        state_str += f"Your Victory Points: {player.victory_points}\n"
        
        # Buildings
        state_str += f"Your Buildings: {len(player.buildings)} settlements, {len(player.cities)} cities, {len(player.roads)} roads\n"
        
        # Development cards
        dev_cards = player.development_deck.to_dict()
        state_str += f"Your Development Cards: {dev_cards}\n"
        
        # Opponent summary
        state_str += "Opponents:\n"
        for opp in opponents:
            state_str += f"- {opp.color}: {opp.victory_points} points, {len(opp.buildings)} settlements, {len(opp.cities)} cities, {len(opp.roads)} roads\n"
        
        # Game turn
        state_str += f"Current Turn: {game.state.turn}\n"
        
        return state_str

    def _describe_action(self, action, game):
        """
        Generate a human-readable description of an action.
        
        Args:
            action (Action): The action to describe
            game (Game): The current game state
        Returns:
            str: A human-readable description of the action
        """
        action_type = action.action_type
        
        if action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build settlement at node {action.value}"
        elif action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {action.value}"
        elif action_type == ActionType.BUILD_ROAD:
            return f"Build road at edge {action.value}"
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buy development card"
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            return f"Play Knight card and move robber to {action.value[0]}"
        elif action_type == ActionType.END_TURN:
            return "End your turn"
        elif action_type == ActionType.ROLL:
            return "Roll the dice"
        elif action_type == ActionType.MOVE_ROBBER:
            target = action.value[1]
            target_str = target if target else "no player"
            return f"Move robber to {action.value[0]} and steal from {target_str}"
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return f"Play Year of Plenty card and take resources {action.value}"
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            return f"Play Road Building card and build roads at {action.value}"
        elif action_type == ActionType.PLAY_MONOPOLY:
            return f"Play Monopoly card and take all {action.value} resources"
        elif action_type == ActionType.MARITIME_TRADE:
            return f"Trade {action.value[0]} for {action.value[1]}"
        elif action_type == ActionType.DISCARD:
            return f"Discard cards {action.value}"
        elif action_type == ActionType.INITIAL_SETTLEMENT:
            return f"Build initial settlement at node {action.value}"
        elif action_type == ActionType.INITIAL_ROAD:
            return f"Build initial road at edge {action.value}"
        else:
            return str(action)