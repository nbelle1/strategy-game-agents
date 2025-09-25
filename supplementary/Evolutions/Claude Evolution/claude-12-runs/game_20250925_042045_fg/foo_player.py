import os
from typing import Iterable, List, Tuple

# MUST use adapters to interact with the game. Do NOT import catanatron directly.
from .adapters import (
    Game,
    Player,
    Color,
    copy_game,
    execute_deterministic,
    list_prunned_actions,
    prune_robber_actions,
    contender_fn,
    base_fn,
    DEFAULT_WEIGHTS,
    ActionType,
)


class FooPlayer(Player):
    """A lookahead player that focuses on infrastructure and robber pruning.

    Improvements over the previous version:
    - Default lookahead depth increased to 3 to capture longer-term infrastructure
      consequences (settlements, roads, cities).
    - Uses contender_fn (with DEFAULT_WEIGHTS) as the primary heuristic to bias
      evaluation toward infrastructure. Falls back to base_fn on failure.
    - Uses prune_robber_actions to reduce robber move branching when robber
      actions are available.
    - Prioritizes infrastructure actions (BUILD_SETTLEMENT, BUILD_ROAD,
      BUILD_CITY) over maritime trades when possible.

    Implementation notes / reminders:
    - Always interact with the game through the .adapters surface.
    - Keep print() debugging statements to trace decisions and node expansions.
    - Defensive try/excepts ensure we fall back to a heuristic evaluation if any
      adapter call fails instead of crashing the whole player.
    """

    def __init__(self, name: str | None = None, lookahead_depth: int = 3):
        # Initialize as BLUE (same as previous implementation). The Player
        # constructor from adapters expects (Color, name)
        super().__init__(Color.BLUE, name)

        # Prefer contender_fn to bias toward infrastructure. contender_fn in the
        # adapters expects parameters; use DEFAULT_WEIGHTS as a sensible default.
        try:
            self.value_fn = contender_fn(DEFAULT_WEIGHTS)
            print('FooPlayer.__init__: Using contender_fn with DEFAULT_WEIGHTS')
        except Exception as e:
            # If contender_fn fails for any reason, fall back to base_fn.
            print(f'FooPlayer.__init__: contender_fn failed, falling back to base_fn: {e}')
            try:
                self.value_fn = base_fn()
            except Exception as inner:
                print(f'FooPlayer.__init__: base_fn also failed, using dumb fallback. {inner}')
                # Final fallback: a lambda that returns 0.0 so code remains safe.
                self.value_fn = lambda g, c: 0.0

        # Lookahead depth controls recursion. Increase default to 3 for deeper
        # planning. Keep lower bound of 1 to avoid invalid depths.
        self.lookahead_depth = max(1, int(lookahead_depth))

        # Counters / debug info to monitor node expansions in a single decision.
        self._node_expansions = 0

    def decide(self, game: Game, playable_actions: Iterable) -> object:
        """Choose an action from playable_actions using a prioritized lookahead.

        Strategy enhancements from previous version:
        - If robber actions are present, use prune_robber_actions to keep only
          impactful robber placements.
        - Prioritize infrastructure actions (settlement/road/city) over
          maritime trades to encourage long-term VP growth.
        - Evaluate a reduced set of actions with lookahead to limit node
          expansions and computation time.
        """
        try:
            actions = list(playable_actions)
        except Exception:
            # playable_actions could be any iterable; ensure we can index it.
            actions = [a for a in playable_actions]

        # Defensive: if there are no actions, return None (game should handle it)
        if not actions:
            print('FooPlayer.decide: No playable actions available, returning None')
            return None

        # Reset debug counters
        self._node_expansions = 0

        # If there are robber actions present, prune them to reduce branching.
        try:
            has_robber = any(getattr(a, 'action_type', None) and 'ROBBER' in getattr(a.action_type, 'name', '') for a in actions)
        except Exception:
            has_robber = False

        if has_robber:
            try:
                pruned = prune_robber_actions(self.color, game, actions)
                if pruned:
                    print(f'FooPlayer.decide: Pruned robber actions from {len(actions)} to {len(pruned)}')
                    actions = pruned
            except Exception as e:
                print(f'FooPlayer.decide: prune_robber_actions failed: {e}')

        # Prioritize infrastructure actions over maritime trades and other low
        # value actions. If we have any infrastructure actions, focus on them.
        try:
            infrastructure_types = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD, ActionType.BUILD_CITY}
            infrastructure_actions = [a for a in actions if getattr(a, 'action_type', None) in infrastructure_types]
            if infrastructure_actions:
                print(f'FooPlayer.decide: Prioritizing {len(infrastructure_actions)} infrastructure actions over {len(actions)} total')
                actions = infrastructure_actions
            else:
                # If no infrastructure actions, try to deprioritize maritime trades
                # when there are many options (to avoid repeatedly choosing trades).
                if len(actions) > 6:
                    non_trade_actions = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                    if non_trade_actions:
                        print(f'FooPlayer.decide: Filtering out maritime trades from {len(actions)} to {len(non_trade_actions)} actions')
                        actions = non_trade_actions
        except Exception as e:
            print(f'FooPlayer.decide: Error during action prioritization: {e}')

        best_action = None
        best_score = float('-inf')

        print(f'FooPlayer.decide: Evaluating {len(actions)} actions with lookahead depth={self.lookahead_depth}')

        # Evaluate each candidate action by simulating its deterministic outcomes
        for idx, action in enumerate(actions):
            try:
                # copy the game and execute the action deterministically
                game_copy = copy_game(game)
                outcomes = execute_deterministic(game_copy, action)

                # outcomes is a list of (game_after_action, probability) tuples
                expected_value = 0.0
                for (outcome_game, prob) in outcomes:
                    # For each outcome, perform a recursive lookahead of depth-1
                    node_value = self._evaluate_node(outcome_game, self.lookahead_depth - 1)
                    expected_value += prob * node_value

                print(f'  Action {idx}: expected_value={expected_value} action_type={getattr(action, "action_type", None)}')

                # Since these actions are available to the current player, we
                # select the action with the highest expected value.
                if expected_value > best_score:
                    best_score = expected_value
                    best_action = action

            except Exception as e:
                # Catch exceptions per-action to avoid crashing during decide.
                print(f'FooPlayer.decide: Exception while evaluating action {action}: {e}')

        # Fallback to the first action if something went wrong and no best_action
        chosen = best_action if best_action is not None else actions[0]
        print(f'FooPlayer.decide: Chosen action={chosen} score={best_score} node_expansions={self._node_expansions}')
        return chosen

    def _evaluate_node(self, game: Game, depth: int) -> float:
        """Recursive evaluator that returns a heuristic value for the given game
        state with a remaining lookahead depth.

        Enhancements:
        - When robber actions are present for the current actor, use
          prune_robber_actions to reduce branching and focus on impactful
          robber placements.
        - When many actions exist, deprioritize maritime trades to limit
          expansion.
        """
        # Update expansion counter for debugging / profiling
        self._node_expansions += 1

        # Base case: evaluate with heuristic
        if depth <= 0:
            try:
                val = float(self.value_fn(game, self.color))
            except Exception as e:
                print(f'FooPlayer._evaluate_node: value_fn raised exception: {e}')
                val = 0.0
            return val

        # Get a pruned list of actions for this game state to reduce branching.
        try:
            actions = list_prunned_actions(game)
        except Exception as e:
            print(f'FooPlayer._evaluate_node: list_prunned_actions failed: {e}')
            # Fallback: evaluate directly
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        if not actions:
            # No actions -> evaluate terminal-like state
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        # If robber actions are present for the current actor, prune them.
        try:
            has_robber = any(getattr(a, 'action_type', None) and 'ROBBER' in getattr(a.action_type, 'name', '') for a in actions)
        except Exception:
            has_robber = False

        if has_robber:
            try:
                # Use the color of the current actor to prune appropriately.
                current_color = actions[0].color
                pruned = prune_robber_actions(current_color, game, actions)
                if pruned:
                    actions = pruned
            except Exception as e:
                print(f'FooPlayer._evaluate_node: prune_robber_actions failed: {e}')

        # If there are many actions, deprioritize maritime trades to lower
        # branching factor. Keep trades only if no other options exist.
        try:
            if len(actions) > 8:
                non_trade_actions = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                if non_trade_actions:
                    actions = non_trade_actions
        except Exception as e:
            print(f'FooPlayer._evaluate_node: Error filtering maritime trades: {e}')

        # Determine whether current player is us or the opponent by inspecting
        # the first action's color. All returned actions should be for the same
        # player (the current player in the provided game state).
        current_actor_color = actions[0].color
        is_maximizing = (current_actor_color == self.color)

        # Evaluate each action to compute either the max or min expected value.
        best_value = float('-inf') if is_maximizing else float('inf')

        for action in actions:
            try:
                game_copy = copy_game(game)
                outcomes = execute_deterministic(game_copy, action)

                expected = 0.0
                for (outcome_game, prob) in outcomes:
                    expected += prob * self._evaluate_node(outcome_game, depth - 1)

                if is_maximizing:
                    if expected > best_value:
                        best_value = expected
                else:
                    if expected < best_value:
                        best_value = expected

            except Exception as e:
                print(f'FooPlayer._evaluate_node: Exception on action {action}: {e}')

        # If something went wrong and best_value remains inf/-inf, evaluate directly
        if best_value == float('inf') or best_value == float('-inf'):
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        return best_value
