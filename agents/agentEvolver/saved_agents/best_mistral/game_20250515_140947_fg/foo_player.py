import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.state_functions import (
    get_player_freqdeck, player_num_resource_cards, get_player_buildings,
    get_longest_road_color, get_longest_road_length, get_largest_army,
    get_enemy_colors, get_actual_victory_points, get_visible_victory_points
)


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Args:
        #     game (Game): complete game state. read-only.
        #         Defined in in \"catanatron/catanatron_core/catanatron/game.py\"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions

        # ===== YOUR CODE HERE =====
        # Initialize variables to track the best action and its score
        best_action = None
        best_score = -1

        # Gather information about the game state
        state = game.state
        player_colors = game.state.colors
        player_color = self.color
        current_player_color = game.state.current_color()
        current_phase = game.state.current_prompt
        winning_color = game.winning_color()

        player_resources = get_player_freqdeck(state, player_color)
        num_resources = player_num_resource_cards(state, player_color)
        player_buildings = get_player_buildings(state, player_color, None)
        longest_road_color = get_longest_road_color(state)
        longest_road_length = get_longest_road_length(state, player_color)
        largest_army = get_largest_army(state)
        enemy_colors = get_enemy_colors(player_colors, player_color)
        actual_vps = get_actual_victory_points(state, player_color)
        visible_vps = get_visible_victory_points(state, player_color)

        # Determine the current phase of the game
        if actual_vps < 4:
            phase = 'early'
        elif actual_vps < 7:
            phase = 'mid'
        else:
            phase = 'late'

        # Iterate through all playable actions
        for action in playable_actions:
            # Create a copy of the game to simulate the action
            game_copy = game.copy()
            game_copy.execute(action)
            new_state = game_copy.state

            # Assign scores based on the type of action and the game state
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                score = 3 + len(get_player_buildings(new_state, player_color, 'SETTLEMENT'))
                if phase == 'early':
                    score += 2  # Prioritize settlements in the early game
                # Dynamic resource thresholds and trade evaluation
                if player_resources.get('wood', 0) >= 1 and player_resources.get('brick', 0) >= 1 and player_resources.get('wool', 0) >= 1 and player_resources.get('grain', 0) >= 1:
                    score += 1
                print('Choosing to build a settlement')
            elif action.action_type == ActionType.BUILD_CITY:
                score = 2 + len(get_player_buildings(new_state, player_color, 'CITY'))
                if phase == 'late':
                    score += 2  # Prioritize cities in the late game
                # Dynamic resource thresholds and trade evaluation
                if player_resources.get('ore', 0) >= 3 and player_resources.get('grain', 0) >= 2:
                    score += 1
                print('Choosing to build a city')
            elif action.action_type == ActionType.BUILD_ROAD:
                score = 1 + get_longest_road_length(new_state, player_color)
                if phase == 'mid':
                    score += 1  # Prioritize roads in the mid-game
                # Dynamic resource thresholds and trade evaluation
                if player_resources.get('wood', 0) >= 1 and player_resources.get('brick', 0) >= 1:
                    score += 1
                print('Choosing to build a road')
            else:
                score = 0

            # Aggressive blocking strategies and competitive building
            for enemy_color in enemy_colors:
                enemy_buildings = get_player_buildings(new_state, enemy_color, None)
                if action.action_type == ActionType.BUILD_SETTLEMENT and any(building.location in enemy_buildings for building in get_player_buildings(new_state, player_color, 'SETTLEMENT')):
                    score += 1  # Blocking enemy expansion
                if action.action_type == ActionType.BUILD_ROAD and any(building.location in enemy_buildings for building in get_player_buildings(new_state, player_color, 'ROAD')):
                    score += 1  # Blocking enemy expansion

            # Update the best action if the current one has a higher score
            if score > best_score:
                best_action = action
                best_score = score

        # If no preferred action is found, fall back to the first action
        if best_action is None:
            best_action = playable_actions[0]
            print('Choosing First Action on Default')

        return best_action
        # ===== END YOUR CODE =====