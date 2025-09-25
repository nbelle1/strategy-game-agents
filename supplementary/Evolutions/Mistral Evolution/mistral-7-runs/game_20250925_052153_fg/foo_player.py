from .adapters import (
    Game,
    Player,
    Color,
    copy_game,
    execute_deterministic,
    execute_spectrum,
    list_prunned_actions,
    base_fn,
)

import math
from typing import List


class FooPlayer(Player):
    def __init__(self, name=None):
        # Initialize as the BLUE player by default (keeps compatibility with previous versions)
        super().__init__(Color.BLUE, name)

    def decide(self, game: Game, playable_actions):
        """
        Choose an action using a multi-ply Expectimax with probabilistic simulation.

        Strategy implemented:
        - Use an Expectimax search to a fixed depth (default 2 plies).
        - For each node, we consider the pruned action list returned by adapters.list_prunned_actions
          to reduce branching.
        - For each action we simulate all possible outcomes using execute_spectrum (which returns
          (game_after, probability) tuples). This naturally handles deterministic actions as a
          special case (single outcome with prob=1.0).
        - Chance outcomes are folded into the expected value computation for the action.
        - Nodes where the acting color equals this player's color are treated as MAX nodes;
          otherwise they are treated as MIN nodes (adversarial opponent).

        Notes & assumptions:
        - We rely only on the adapters surface (copy_game, execute_spectrum, execute_deterministic,
          list_prunned_actions, base_fn).
        - If playable_actions is empty, returns None.
        - Depth counts plies: depth=0 means evaluate the current state with the heuristic.
        - This implementation avoids additional hand-crafted heuristics and follows the
          expectimax structure proposed by the strategizer.

        Debugging:
        - Print statements emit evaluated expected values for top-level actions and any exceptions
          encountered during simulation.

        Args:
            game (Game): read-only current game state
            playable_actions (Iterable[Action]): legal actions to choose from
        Returns:
            Action | None: chosen action or None if no actions available
        """
        # Convert to a concrete list so we can iterate multiple times and index
        actions = list(playable_actions) if playable_actions is not None else []

        if len(actions) == 0:
            print('FooPlayer.decide: No playable actions available; returning None')
            return None

        # Depth for Expectimax (plies). Start with 2 as recommended.
        MAX_DEPTH = 2

        # Build the base value function once. It returns a function f(game, color)->float
        value_fn = base_fn()

        # Recursive Expectimax implementation
        def expectimax(node_game: Game, depth: int) -> float:
            """Return the expectimax value of node_game from the perspective of self.color.

            - If depth == 0 or there are no legal actions, evaluate with value_fn.
            - Otherwise, for each pruned action, compute the expected value over its spectrum
              (execute_spectrum). Then either take max or min over actions depending on the
              acting color.
            """
            try:
                # Terminal check: if the game reports a winner, evaluate directly.
                winner = None
                try:
                    # Many Game implementations expose a winning_color() method per adapters comment.
                    winner = node_game.winning_color()
                except Exception:
                    # If winning_color isn't available or errors, fall back to continuing search.
                    winner = None

                if winner is not None:
                    # Terminal state: return heuristic value (value_fn may incorporate terminal logic)
                    return value_fn(node_game, self.color)

                if depth == 0:
                    return value_fn(node_game, self.color)

                # Get a pruned list of actions to reduce branching.
                node_actions = list_prunned_actions(node_game)

                if not node_actions:
                    # No legal actions -> evaluate heuristic
                    return value_fn(node_game, self.color)

                # Determine if this node is a MAX node (our player) or MIN node (opponent).
                # We infer the acting color from the first available action; list_prunned_actions
                # returns actions with an associated color field.
                node_color = node_actions[0].color
                is_max_node = (node_color == self.color)

                if is_max_node:
                    best_value = -math.inf
                    # For each action, compute expected value across possible outcomes
                    for act in node_actions:
                        try:
                            # Use spectrum expansion to handle chance outcomes. Deterministic actions
                            # will simply return a single outcome with prob=1.0.
                            outcomes = execute_spectrum(node_game, act)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: execute_spectrum failed for action {act}: {e}')
                            continue

                        expected = 0.0
                        for (g_after, prob) in outcomes:
                            try:
                                val = expectimax(g_after, depth - 1)
                            except Exception as e:
                                print(f'FooPlayer.expectimax: recursion error on outcome {g_after}: {e}')
                                val = -math.inf
                            expected += prob * val

                        if expected > best_value:
                            best_value = expected

                    return best_value

                else:
                    # MIN node: assume adversary minimizes our value (adversarial opponent)
                    worst_value = math.inf
                    for act in node_actions:
                        try:
                            outcomes = execute_spectrum(node_game, act)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: execute_spectrum failed for action {act}: {e}')
                            continue

                        expected = 0.0
                        for (g_after, prob) in outcomes:
                            try:
                                val = expectimax(g_after, depth - 1)
                            except Exception as e:
                                print(f'FooPlayer.expectimax: recursion error on outcome {g_after}: {e}')
                                val = math.inf
                            expected += prob * val

                        if expected < worst_value:
                            worst_value = expected

                    return worst_value

            except Exception as e:
                # Any unexpected error during expectimax should yield a very low value so the action
                # won't be chosen at the top level.
                print(f'FooPlayer.expectimax: unexpected error: {e}')
                return -math.inf

        # Evaluate each top-level action using the expectimax search
        best_action = None
        best_value = -math.inf

        for idx, action in enumerate(actions):
            try:
                # Copy the game to avoid any in-place changes by adapters
                game_copy = copy_game(game)

                # Use execute_spectrum to capture all possible outcomes (handles deterministic as well)
                try:
                    outcomes = execute_spectrum(game_copy, action)
                except Exception as e:
                    # Fall back to deterministic execution if spectrum isn't supported for this action
                    print(f'FooPlayer.decide: execute_spectrum failed for top-level action {action}: {e}; trying deterministic')
                    try:
                        outcomes = execute_deterministic(game_copy, action)
                    except Exception as e2:
                        print(f'FooPlayer.decide: execute_deterministic also failed for action {action}: {e2}')
                        outcomes = []

                if not outcomes:
                    print(f'FooPlayer.decide: No outcomes for action {action}; skipping')
                    expected_value = -math.inf
                else:
                    expected_value = 0.0
                    for (g_after, prob) in outcomes:
                        try:
                            val = expectimax(g_after, MAX_DEPTH - 1)
                        except Exception as e:
                            print(f'FooPlayer.decide: expectimax error on outcome of action {action}: {e}')
                            val = -math.inf
                        expected_value += prob * val

                print(f'FooPlayer: Top-level Action #{idx} = {action} => expected value {expected_value}')

            except Exception as e:
                print(f'FooPlayer: Exception while evaluating top-level action {action}: {e}')
                expected_value = -math.inf

            # Track the best action seen so far (we maximize at the root for our player)
            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        # Fallback: if best_action is still None (all evaluations failed), pick the first action
        if best_action is None and len(actions) > 0:
            print('FooPlayer: All evaluations failed; defaulting to first action')
            best_action = actions[0]

        print(f'FooPlayer: Chosen action = {best_action} with value {best_value}')
        return best_action
