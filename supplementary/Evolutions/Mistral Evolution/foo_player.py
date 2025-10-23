from .adapters import (
    Game,
    Player,
    Color,
    copy_game,
    execute_deterministic,
    base_fn,
)

import math

class FooPlayer(Player):
    def __init__(self, name=None):
        # Initialize as the BLUE player by default (keeps compatibility with previous versions)
        super().__init__(Color.BLUE, name)

    def decide(self, game: Game, playable_actions):
        """
        Choose an action using a 1-ply expected-value lookahead.

        Strategy implemented:
        - For each legal action, simulate its deterministic outcomes using execute_deterministic
          (wrapped by the adapters API). This returns a list of (game_after, probability).
        - Evaluate each outcome with a base value function (base_fn) provided by the adapters.
        - Compute the expected value for the action as the probability-weighted sum of outcome values.
        - Choose the action with the highest expected value.

        Notes & assumptions:
        - We only perform a single-ply lookahead (no deeper recursion).
        - Uses adapters.copy_game to avoid mutating the provided game state.
        - Uses adapters.execute_deterministic to get outcome game states; deterministic actions
          should return a single outcome with probability 1.0, but we still handle multiple outcomes
          for generality.
        - If playable_actions is empty, returns None.

        Debugging:
        - Print statements emit the evaluated expected value for each action and the final choice.

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

        # Build the base value function once. It returns a function f(game, color)->float
        value_fn = base_fn()

        best_action = None
        best_value = -math.inf

        # Evaluate each action with a 1-ply expected value calculation
        for idx, action in enumerate(actions):
            try:
                # Use a copy of the game to be safe. Adapters functions typically return copies,
                # but copying upfront avoids side-effects if an adapter does in-place changes.
                game_copy = copy_game(game)

                # Execute the action deterministically; this returns a list of (game_after, prob)
                outcomes = execute_deterministic(game_copy, action)

                # Compute expected value across all possible deterministic outcomes
                expected_value = 0.0
                for (g_after, prob) in outcomes:
                    try:
                        val = value_fn(g_after, self.color)
                    except Exception as e:
                        # If the value function errors for some reason, catch and report it.
                        print(f'FooPlayer: value_fn raised {e} on action {action}; treating as -inf')
                        val = -math.inf
                    expected_value += prob * val

                print(f'FooPlayer: Action #{idx} = {action} => expected value {expected_value}')

            except Exception as e:
                # If executing the action failed (shouldn't normally happen via adapters),
                # assign a very low value so it won't be chosen.
                print(f'FooPlayer: Exception while evaluating action {action}: {e}')
                expected_value = -math.inf

            # Track the best action seen so far
            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        # Fallback: if best_action is still None (all evaluations failed), pick the first action
        if best_action is None and len(actions) > 0:
            print('FooPlayer: All evaluations failed; defaulting to first action')
            best_action = actions[0]

        print(f'FooPlayer: Chosen action = {best_action} with value {best_value}')
        return best_action
