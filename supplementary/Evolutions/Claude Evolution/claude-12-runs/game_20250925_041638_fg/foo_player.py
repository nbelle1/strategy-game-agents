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
    base_fn,
)


class FooPlayer(Player):
    """A simple lookahead player that uses the adapters API to simulate
    actions and evaluate resulting states using a heuristic value function.

    Strategy implemented:
    - For each playable action, simulate the deterministic outcomes using
      execute_deterministic on a copied game state.
    - Recursively perform a shallow minimax-style lookahead with alternating
      players: maximize for this player, minimize for the opponent.
    - Use base_fn() from adapters as the heuristic evaluator at leaf nodes.

    Notes / learning points included as comments and print debugging to help
    evolve the player in subsequent iterations.
    """

    def __init__(self, name: str | None = None, lookahead_depth: int = 2):
        # Initialize as BLUE (same as previous implementation). The Player
        # constructor from adapters expects (Color, name)
        super().__init__(Color.BLUE, name)
        # Create a value function instance using the adapters' base_fn factory.
        # base_fn returns a callable f(game, color) -> float.
        self.value_fn = base_fn()

        # Lookahead depth controls the recursion depth for the minimax.
        # Depth 1 evaluates immediate resulting states; depth 2 looks one
        # opponent response deeper, etc. Keep small to limit compute.
        self.lookahead_depth = max(1, int(lookahead_depth))

        # Counters / debug info to monitor node expansions in a single decision.
        self._node_expansions = 0

    def decide(self, game: Game, playable_actions: Iterable) -> object:
        """Choose an action from playable_actions using a shallow lookahead.

        Args:
            game (Game): complete game state (read-only). Must use copy_game
                         to create simulations of this state.
            playable_actions (Iterable[Action]): available actions for the
                         current game state.
        Returns:
            An Action from playable_actions.
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
                # For deterministic execution this will typically be one element
                # with probability 1. We'll compute the expected value across
                # all outcomes.
                expected_value = 0.0
                for (outcome_game, prob) in outcomes:
                    # For each outcome, perform a recursive lookahead of depth-1
                    node_value = self._evaluate_node(outcome_game, self.lookahead_depth - 1)
                    expected_value += prob * node_value

                print(f'  Action {idx}: expected_value={expected_value}')

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

        Implementation details:
        - If depth == 0, evaluate the state with self.value_fn(game, self.color).
        - Otherwise, list pruned actions for the current game state using the
          adapters' list_prunned_actions(). For each action, simulate
          deterministic outcomes and compute the expected value recursively.
        - If the actions belong to this player (action.color == self.color), we
          take the maximum over actions. If they belong to the opponent, we
          take the minimum (adversarial assumption).

        This is a shallow minimax with deterministic expansions. Chance nodes
        (dice, dev draws) are respected by execute_deterministic / execute_spectrum
        when used; here we only call execute_deterministic for speed and
        simplicity. Future iterations could expand chance outcomes explicitly.
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
