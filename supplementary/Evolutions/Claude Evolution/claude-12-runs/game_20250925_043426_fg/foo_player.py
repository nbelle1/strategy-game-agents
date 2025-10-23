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

    Notes about this update (rollback and fixes):
    - Reverted aggressive pruning introduced earlier that limited root/child
      expansions. We fully evaluate prioritized infrastructure actions to
      restore the successful Evolution 2 behavior.
    - Fixed bugs caused by using Python's filter() without materializing into
      a list. All filtering uses list comprehensions so len() and indexing work.
    - Robust defensive error handling kept so any adapter failure falls back
      to heuristic evaluation instead of crashing the player.
    """

    def __init__(self, name: str | None = None, lookahead_depth: int = 3):
        # Initialize as BLUE (same as previous implementation). The Player
        # constructor from adapters expects (Color, name)
        super().__init__(Color.BLUE, name)

        # Prefer contender_fn to bias toward infrastructure.
        try:
            self.value_fn = contender_fn(DEFAULT_WEIGHTS)
            print('FooPlayer.__init__: Using contender_fn with DEFAULT_WEIGHTS')
        except Exception as e:
            print(f'FooPlayer.__init__: contender_fn failed, falling back to base_fn: {e}')
            try:
                self.value_fn = base_fn()
                print('FooPlayer.__init__: Using base_fn as fallback')
            except Exception as inner:
                print(f'FooPlayer.__init__: base_fn also failed, using dumb fallback. {inner}')
                self.value_fn = lambda g, c: 0.0

        # Lookahead depth controls recursion. Increase default to 3 for deeper
        # planning. Keep lower bound of 1 to avoid invalid depths.
        self.lookahead_depth = max(1, int(lookahead_depth))

        # Counters / debug info to monitor node expansions in a single decision.
        self._node_expansions = 0

    def decide(self, game: Game, playable_actions: Iterable) -> object:
        """Choose an action from playable_actions using a prioritized lookahead.

        Strategy enhancements and bug fixes:
        - Materialize any iterables into lists (avoid filter iterator bugs).
        - Use prune_robber_actions when appropriate.
        - Prioritize infrastructure actions (BUILD_SETTLEMENT, BUILD_ROAD, BUILD_CITY)
          over maritime trades when possible.

        Note: aggressive root/child pruning was intentionally removed to restore
        Evolution 2 behavior that achieved high win rates.
        """
        try:
            actions = list(playable_actions)
        except Exception:
            # playable_actions could be any iterable; ensure we can iterate it.
            actions = [a for a in playable_actions]

        if not actions:
            print('FooPlayer.decide: No playable actions available, returning None')
            return None

        # Reset debug counters
        self._node_expansions = 0

        # Detect and prune robber actions (safe check using name contains 'ROBBER')
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
                # Ensure pruned is a list; adapters should return a list but be defensive
                pruned = list(pruned) if pruned is not None else pruned
                if pruned and len(pruned) < len(actions):
                    print(f'FooPlayer.decide: Pruned robber actions from {len(actions)} to {len(pruned)}')
                    actions = pruned
            except Exception as e:
                print(f'FooPlayer.decide: prune_robber_actions failed: {e}')

        # Prioritize infrastructure actions. Strongly prefer BUILD_CITY but do not
        # exclude other infrastructure — evaluate cities first but keep settlements
        # and roads available as fallbacks so the agent remains flexible.
        try:
            infrastructure_types = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD, ActionType.BUILD_CITY}

            # Find any infrastructure actions among current candidates.
            infrastructure_actions = [a for a in actions if getattr(a, 'action_type', None) in infrastructure_types]

            # Extract explicit city-upgrade actions so we can prioritize them (but not
            # make them exclusive). We will evaluate city actions first by ordering.
            city_actions = [a for a in infrastructure_actions if getattr(a, 'action_type', None) == ActionType.BUILD_CITY]

            if city_actions:
                # Re-order actions so city upgrades are evaluated first, followed by
                # other infrastructure actions. This strongly biases selection toward
                # city upgrades while still allowing settlements/roads to be chosen
                # if they evaluate higher during full lookahead.
                ordered_infra = city_actions + [a for a in infrastructure_actions if a not in city_actions]
                print(f'FooPlayer.decide: Prioritizing {len(city_actions)} city upgrade(s) among {len(infrastructure_actions)} infrastructure actions')
                actions = ordered_infra
            elif infrastructure_actions:
                print(f'FooPlayer.decide: Prioritizing {len(infrastructure_actions)} infrastructure actions over {len(actions)} total')
                actions = infrastructure_actions
            else:
                # If no infrastructure actions, try to deprioritize maritime trades
                # when there are many options (reverted to Evolution 2 threshold >8).
                if len(actions) > 8:
                    non_trade_actions = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                    if non_trade_actions:
                        print(f'FooPlayer.decide: Filtering out maritime trades from {len(actions)} to {len(non_trade_actions)} actions')
                        actions = non_trade_actions
        except Exception as e:
            print(f'FooPlayer.decide: Error during action prioritization: {e}')

        # Full evaluation of all remaining actions with lookahead (no aggressive pruning)
        best_action = None
        best_score = float('-inf')

        print(f'FooPlayer.decide: Fully evaluating {len(actions)} actions with lookahead depth={self.lookahead_depth}')

        # Evaluate all candidate actions with full lookahead
        for idx, action in enumerate(actions):
            try:
                game_copy = copy_game(game)
                outcomes = execute_deterministic(game_copy, action)

                expected_value = 0.0
                for (outcome_game, prob) in outcomes:
                    node_value = self._evaluate_node(outcome_game, self.lookahead_depth - 1)
                    expected_value += prob * node_value

                # Small explicit bonus for city upgrades to further bias selection toward
                # upgrading settlements to cities (helps restore Evolution 2 behavior).
                try:
                    if getattr(action, 'action_type', None) == ActionType.BUILD_CITY:
                        # Add a modest bonus (tunable). We use an additive bonus so the
                        # heuristic scale from adapters continues to drive major decisions.
                        city_bonus = 50.0
                        expected_value += city_bonus
                        print(f'  Action {idx}: applied city bonus (+{city_bonus})')
                except Exception:
                    pass

                print(f'  Action {idx}: expected_value={expected_value} action_type={getattr(action, "action_type", None)}')

                if expected_value > best_score:
                    best_score = expected_value
                    best_action = action

            except Exception as e:
                print(f'FooPlayer.decide: Exception while evaluating action {action}: {e}')

        # Fallback to the first original action if something went wrong
        chosen = best_action if best_action is not None else actions[0]
        print(f'FooPlayer.decide: Chosen action={chosen} score={best_score} node_expansions={self._node_expansions}')
        return chosen

    def _evaluate_node(self, game: Game, depth: int) -> float:
        """Recursive evaluator that returns a heuristic value for the given game
        state with a remaining lookahead depth.

        This function intentionally avoids aggressive child-pruning. It will
        still use list_prunned_actions and prune_robber_actions to reduce
        obviously irrelevant moves, but will otherwise recurse into all
        remaining legal/pruned actions so the search can find strong
        infrastructure lines.
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
            # Make sure we have a materialized list
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

        # If robber actions are present for the current actor, prune them.
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
                if pruned:
                    # Only accept pruning if it meaningfully reduces branching
                    if len(pruned) < len(actions):
                        print(f'FooPlayer._evaluate_node: Pruned robber actions from {len(actions)} to {len(pruned)}')
                        actions = pruned
            except Exception as e:
                print(f'FooPlayer._evaluate_node: prune_robber_actions failed: {e}')

        # Determine whether current player is us or the opponent by inspecting
        # the first action's color. All returned actions should be for the same
        # player (the current player in the provided game state).
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

        # If something went wrong and best_value remains +/-inf, evaluate directly
        if best_value == float('inf') or best_value == float('-inf'):
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        return best_value
