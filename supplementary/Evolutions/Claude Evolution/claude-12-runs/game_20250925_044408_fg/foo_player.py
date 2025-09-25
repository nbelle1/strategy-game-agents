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
    """A lookahead player restored to Evolution 2 behavior.

    Key design decisions in this restoration:
    - Use contender_fn(DEFAULT_WEIGHTS) as the primary heuristic with base_fn
      as a fallback. Evolution 2 likely used contender_fn primarily.
    - Keep lookahead_depth default at 3 for deeper planning.
    - Keep prune_robber_actions to reduce pointless robber branching, but do not
      aggressively prune infrastructure.
    - Do NOT apply explicit city bonuses or action re-ordering. Let the value
      function drive decisions naturally (as in Evolution 2).
    - Use simple maritime trade filtering only when there are more than 8
      candidate actions (Evolution 2 threshold).
    - Materialize all filtered iterables into lists to avoid filter iterator
      bugs (len() and indexing). Keep defensive error handling.
    """

    def __init__(self, name: str | None = None, lookahead_depth: int = 3):
        # Initialize as BLUE (consistent with prior implementations)
        super().__init__(Color.BLUE, name)

        # Try to use contender_fn first (restore Evolution 2 primary heuristic),
        # fall back to base_fn, then to a dumb lambda if both fail.
        try:
            self.value_fn = contender_fn(DEFAULT_WEIGHTS)
            print('FooPlayer.__init__: Using contender_fn with DEFAULT_WEIGHTS')
        except Exception as e:
            print(f'FooPlayer.__init__: contender_fn failed, falling back to base_fn: {e}')
            try:
                self.value_fn = base_fn(DEFAULT_WEIGHTS)
                print('FooPlayer.__init__: Using base_fn as fallback')
            except Exception as inner:
                print(f'FooPlayer.__init__: base_fn also failed, using dumb fallback. {inner}')
                self.value_fn = lambda g, c: 0.0

        # Ensure lookahead depth is at least 1
        self.lookahead_depth = max(1, int(lookahead_depth))

        # Debug counter for node expansions within a decision
        self._node_expansions = 0

    def decide(self, game: Game, playable_actions: Iterable) -> object:
        """Choose an action from playable_actions using depth-N lookahead.

        This function intentionally keeps the decision pipeline simple and
        faithful to Evolution 2:
        - Materialize iterables into lists to avoid iterator bugs.
        - Use prune_robber_actions defensively.
        - Apply simple maritime trade filtering only when there are >8 options.
        - Let the configured value function (contender_fn, then base_fn) drive
          the choice through lookahead evaluations. Do not add ad-hoc bonuses
          or heavy re-ordering.
        """
        try:
            actions = list(playable_actions)
        except Exception:
            # Defensive fallback in case playable_actions is a problematic iterable
            actions = [a for a in playable_actions]

        if not actions:
            print('FooPlayer.decide: No playable actions available, returning None')
            return None

        # Reset expansion counter
        self._node_expansions = 0

        # Detect robber actions and prune them defensively to reduce branching.
        try:
            has_robber = any(
                getattr(a, 'action_type', None) is not None and
                'ROBBER' in getattr(a.action_type, 'name', '')
                for a in actions
            )
        except Exception:
            has_robber = False

        if has_robber:
            try:
                pruned = prune_robber_actions(self.color, game, actions)
                pruned = list(pruned) if pruned is not None else pruned
                if pruned and len(pruned) < len(actions):
                    print(f'FooPlayer.decide: Pruned robber actions from {len(actions)} to {len(pruned)}')
                    actions = pruned
            except Exception as e:
                print(f'FooPlayer.decide: prune_robber_actions failed: {e}')

        # Simple maritime trade filtering only when many options exist (>8)
        try:
            if len(actions) > 8:
                non_trade_actions = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                if non_trade_actions:
                    print(f'FooPlayer.decide: Filtering out maritime trades from {len(actions)} to {len(non_trade_actions)} actions')
                    actions = non_trade_actions
        except Exception as e:
            print(f'FooPlayer.decide: maritime trade filtering failed: {e}')

        # Evaluate all remaining actions with lookahead; do not add ad-hoc bonuses.
        best_action = None
        best_score = float('-inf')

        print(f'FooPlayer.decide: Fully evaluating {len(actions)} actions with lookahead depth={self.lookahead_depth}')

        for idx, action in enumerate(actions):
            try:
                # copy_game to avoid mutating original
                game_copy = copy_game(game)
                outcomes = execute_deterministic(game_copy, action)

                expected_value = 0.0
                for (outcome_game, prob) in outcomes:
                    node_value = self._evaluate_node(outcome_game, self.lookahead_depth - 1)
                    expected_value += prob * node_value

                print(f'  Action {idx}: expected_value={expected_value} action_type={getattr(action, "action_type", None)}')

                if expected_value > best_score:
                    best_score = expected_value
                    best_action = action

            except Exception as e:
                print(f'FooPlayer.decide: Exception while evaluating action {action}: {e}')

        # Fallback to first candidate if evaluation failed
        chosen = best_action if best_action is not None else actions[0]
        print(f'FooPlayer.decide: Chosen action={chosen} score={best_score} node_expansions={self._node_expansions}')
        return chosen

    def _evaluate_node(self, game: Game, depth: int) -> float:
        """Recursive evaluator that returns heuristic value for a game state.

        This routine uses list_prunned_actions to reduce the branching factor in
        a conservative way and also uses prune_robber_actions defensively. It
        does not apply aggressive pruning or action bonuses; the configured
        value_fn should drive preferences.
        """
        # Count node expansion
        self._node_expansions += 1

        # Base case: evaluate with heuristic
        if depth <= 0:
            try:
                return float(self.value_fn(game, self.color))
            except Exception as e:
                print(f'FooPlayer._evaluate_node: value_fn raised exception: {e}')
                return 0.0

        # Get pruned actions for this state; materialize into a list
        try:
            actions = list_prunned_actions(game)
            actions = list(actions) if actions is not None else []
        except Exception as e:
            print(f'FooPlayer._evaluate_node: list_prunned_actions failed: {e}')
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        if not actions:
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        # Prune robber actions defensively if present
        try:
            has_robber = any(
                getattr(a, 'action_type', None) is not None and
                'ROBBER' in getattr(a.action_type, 'name', '')
                for a in actions
            )
        except Exception:
            has_robber = False

        if has_robber:
            try:
                current_color = actions[0].color
                pruned = prune_robber_actions(current_color, game, actions)
                pruned = list(pruned) if pruned is not None else pruned
                if pruned and len(pruned) < len(actions):
                    print(f'FooPlayer._evaluate_node: Pruned robber actions from {len(actions)} to {len(pruned)}')
                    actions = pruned
            except Exception as e:
                print(f'FooPlayer._evaluate_node: prune_robber_actions failed: {e}')

        # Determine maximizing/minimizing player
        current_actor_color = actions[0].color
        is_maximizing = (current_actor_color == self.color)

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

        # If evaluation failed to set a value, fall back to heuristic
        if best_value == float('inf') or best_value == float('-inf'):
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        return best_value
