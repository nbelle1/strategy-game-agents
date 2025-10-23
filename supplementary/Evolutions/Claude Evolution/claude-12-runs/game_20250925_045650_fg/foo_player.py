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
    """A hybrid lookahead player with hard constraints against maritime trades.

    Design summary:
    - Primary heuristic: contender_fn(DEFAULT_WEIGHTS) with base_fn fallback.
    - Depth-N lookahead (default N=3) preserved for strategic evaluation.
    - Hard constraint: filter OUT maritime trades unless no infrastructure actions
      (cities/settlements/roads) exist. This prevents persistent maritime trade
      overvaluation.
    - Tiered infrastructure prioritization (cities > settlements > roads > dev
      cards), but use lookahead to choose the best action within each tier.
    - Defensive use of prune_robber_actions and materialization of lists to
      avoid iterator/filter bugs.
    - Plenty of debugging print statements to trace decisions and node counts.
    """

    # Tunable parameters for defensive limits and small tactical nudges
    PER_ACTION_EXPANSION_LIMIT = 300  # if a single candidate explodes nodes, abort and fallback
    ROBBER_PRUNE_MIN_ACTIONS = 8  # only prune robber actions when branching is large
    ROAD_BONUS = 5.0  # small bonus to favor road building when otherwise equal

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
        """Choose an action from playable_actions using a hybrid strategy.

        Strategy:
        - Materialize iterables into lists to avoid iterator bugs.
        - Prune robber actions defensively to reduce pointless branching.
        - Hard-filter maritime trades out unless no infrastructure actions exist.
        - Use tiered infrastructure prioritization (city > settlement > road > dev)
          but use the depth-N lookahead to pick the best action within each tier.
        - If no infra actions exist, evaluate dev cards, then non-trade actions,
          and finally allow maritime trades as an absolute fallback.
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
                # Only apply aggressive pruning when the branching factor is large
                if len(actions) > self.ROBBER_PRUNE_MIN_ACTIONS:
                    pruned = prune_robber_actions(self.color, game, actions)
                    pruned = list(pruned) if pruned is not None else pruned
                    if pruned and len(pruned) < len(actions):
                        print(f'FooPlayer.decide: Pruned robber actions from {len(actions)} to {len(pruned)}')
                        actions = pruned
                    else:
                        # Do not prune if it doesn't meaningfully reduce branching
                        print('FooPlayer.decide: prune_robber_actions returned no meaningful reduction, skipping')
                else:
                    print('FooPlayer.decide: Small action set, skipping robber pruning')
            except Exception as e:
                print(f'FooPlayer.decide: prune_robber_actions failed: {e}')

        # Materialize actions as a list (already done) and prepare tiered lists.
        try:
            infrastructure_types = {ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD}
            infra_actions = [a for a in actions if getattr(a, 'action_type', None) in infrastructure_types]
        except Exception as e:
            print(f'FooPlayer.decide: Failed to compute infrastructure actions: {e}')
            infra_actions = []

        # Hard-filter maritime trades only if there exist infrastructure actions.
        # This prevents the persistent maritime-trade bias.
        try:
            if infra_actions:
                non_trade_infra = [a for a in infra_actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                if non_trade_infra:
                    infra_actions = non_trade_infra
                # Also reduce the global actions to non-trades when infra exists so
                # later fallbacks don't accidentally consider trades before infra.
                non_trade_actions_global = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
                if non_trade_actions_global:
                    actions = non_trade_actions_global
                    print(f'FooPlayer.decide: Infra exists, filtering out maritime trades from global actions, now {len(actions)} actions')
        except Exception as e:
            print(f'FooPlayer.decide: maritime trade hard-filtering failed: {e}')

        # Helper: evaluate candidates with lookahead but protect against runaway expansions
        def evaluate_candidates(candidates: List) -> Tuple[object, float]:
            """Evaluate a list of candidate actions using lookahead and return (best_action, best_score)."""
            best_a = None
            best_s = float('-inf')
            print(f'FooPlayer.decide: Fully evaluating {len(candidates)} candidates with lookahead depth={self.lookahead_depth}')
            for idx, a in enumerate(candidates):
                try:
                    # Soft per-candidate expansion cap: if a single candidate causes too many
                    # node expansions, abort its full lookahead and fallback to heuristic.
                    start_nodes = self._node_expansions
                    val = self._expected_value_for_action(game, a, self.lookahead_depth)
                    used_nodes = self._node_expansions - start_nodes
                    if used_nodes > self.PER_ACTION_EXPANSION_LIMIT:
                        # Abortative fallback: use heuristic evaluation instead of runaway search
                        try:
                            fallback_val = float(self.value_fn(game, self.color))
                        except Exception:
                            fallback_val = 0.0
                        print(f'FooPlayer.decide: Candidate {idx} ({getattr(a, "action_type", None)}) used {used_nodes} nodes, exceeding limit {self.PER_ACTION_EXPANSION_LIMIT}. Using fallback heuristic {fallback_val}')
                        val = fallback_val

                    # Small tactical nudge: prefer roads modestly to improve expansion
                    if getattr(a, 'action_type', None) == ActionType.BUILD_ROAD:
                        val += self.ROAD_BONUS

                    print(f'  Candidate {idx}: expected_value={val} action_type={getattr(a, "action_type", None)}')
                    if val > best_s:
                        best_s = val
                        best_a = a
                except Exception as e:
                    print(f'FooPlayer.decide: Exception evaluating candidate {a}: {e}')
            return best_a, best_s

        # If infra actions exist, evaluate per-tier
        try:
            if infra_actions:
                # BUILD_CITY
                city_cands = [a for a in infra_actions if getattr(a, 'action_type', None) == ActionType.BUILD_CITY]
                if city_cands:
                    chosen, score = evaluate_candidates(city_cands)
                    print(f'FooPlayer.decide: Chosen city action={chosen} score={score} node_expansions={self._node_expansions}')
                    if chosen:
                        return chosen

                # BUILD_SETTLEMENT
                sett_cands = [a for a in infra_actions if getattr(a, 'action_type', None) == ActionType.BUILD_SETTLEMENT]
                if sett_cands:
                    chosen, score = evaluate_candidates(sett_cands)
                    print(f'FooPlayer.decide: Chosen settlement action={chosen} score={score} node_expansions={self._node_expansions}')
                    if chosen:
                        return chosen

                # BUILD_ROAD
                road_cands = [a for a in infra_actions if getattr(a, 'action_type', None) == ActionType.BUILD_ROAD]
                if road_cands:
                    chosen, score = evaluate_candidates(road_cands)
                    print(f'FooPlayer.decide: Chosen road action={chosen} score={score} node_expansions={self._node_expansions}')
                    if chosen:
                        return chosen
        except Exception as e:
            print(f'FooPlayer.decide: Exception during tiered infra evaluation: {e}')

        # If no infra chosen, consider development cards (BUY_DEV_CARD)
        try:
            dev_cands = [a for a in actions if getattr(a, 'action_type', None) == ActionType.BUY_DEV_CARD]
            if dev_cands:
                # Robust per-candidate evaluation for dev cards to avoid exceptions
                best_dev = None
                best_dev_score = float('-inf')
                for idx, a in enumerate(dev_cands):
                    try:
                        start_nodes = self._node_expansions
                        val = self._expected_value_for_action(game, a, self.lookahead_depth)
                        used_nodes = self._node_expansions - start_nodes
                        if used_nodes > self.PER_ACTION_EXPANSION_LIMIT:
                            try:
                                fallback_val = float(self.value_fn(game, self.color))
                            except Exception:
                                fallback_val = 0.0
                            print(f'FooPlayer.decide: Dev candidate {idx} used {used_nodes} nodes, exceeding limit. Using fallback {fallback_val}')
                            val = fallback_val
                        if val > best_dev_score:
                            best_dev_score = val
                            best_dev = a
                        print(f'  Dev Candidate {idx}: expected_value={val} action_type={getattr(a, "action_type", None)}')
                    except Exception as e:
                        # If evaluating this dev candidate failed, skip it but do not abort whole dev evaluation
                        print(f'FooPlayer.decide: Exception evaluating dev candidate {a}: {e}')
                if best_dev:
                    print(f'FooPlayer.decide: Chosen dev card action={best_dev} score={best_dev_score} node_expansions={self._node_expansions}')
                    return best_dev
        except Exception as e:
            print(f'FooPlayer.decide: Exception evaluating dev cards: {e}')

        # Next consider non-trade actions (robber, end-turn, etc.) if any
        try:
            non_trade_cands = [a for a in actions if getattr(a, 'action_type', None) != ActionType.MARITIME_TRADE]
            if non_trade_cands:
                chosen, score = evaluate_candidates(non_trade_cands)
                print(f'FooPlayer.decide: Chosen non-trade action={chosen} score={score} node_expansions={self._node_expansions}')
                if chosen:
                    return chosen
        except Exception as e:
            print(f'FooPlayer.decide: Exception evaluating non-trade actions: {e}')

        # Absolute fallback: evaluate all remaining actions including maritime trades
        try:
            chosen, score = evaluate_candidates(actions)
            print(f'FooPlayer.decide: Fallback chosen action={chosen} score={score} node_expansions={self._node_expansions}')
            if chosen:
                return chosen
        except Exception as e:
            print(f'FooPlayer.decide: Exception in final fallback evaluation: {e}')

        # As a final safety net return the first action
        print('FooPlayer.decide: All evaluations failed or none returned a choice, returning first available action')
        return actions[0]

    def _expected_value_for_action(self, game: Game, action, depth: int) -> float:
        """Compute expected value of an action by executing deterministically and
        evaluating resulting states with recursive lookahead (_evaluate_node).

        depth parameter is the full lookahead depth to pass to _evaluate_node for
        resulting states (we treat the action execution as consuming one ply).
        """
        try:
            game_copy = copy_game(game)
            outcomes = execute_deterministic(game_copy, action)

            expected_value = 0.0
            for (outcome_game, prob) in outcomes:
                try:
                    node_value = self._evaluate_node(outcome_game, max(0, depth - 1))
                except Exception as e:
                    print(f'FooPlayer._expected_value_for_action: _evaluate_node failed: {e}')
                    try:
                        node_value = float(self.value_fn(outcome_game, self.color))
                    except Exception:
                        node_value = 0.0
                expected_value += prob * node_value
            return expected_value
        except Exception as e:
            print(f'FooPlayer._expected_value_for_action: Exception executing action {action}: {e}')
            # Fallback to heuristic on current game state (conservative)
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

    def _evaluate_node(self, game: Game, depth: int) -> float:
        """Recursive evaluator that returns heuristic value for a game state.

        This routine uses list_prunned_actions to reduce the branching factor in
        a conservative way and also uses prune_robber_actions defensively. It
        will maximize for nodes where the current actor is this player's color
        and minimize otherwise.
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
                # Only prune when branching is large to avoid removing useful options
                if len(actions) > self.ROBBER_PRUNE_MIN_ACTIONS:
                    current_color = actions[0].color
                    pruned = prune_robber_actions(current_color, game, actions)
                    pruned = list(pruned) if pruned is not None else pruned
                    if pruned and len(pruned) < len(actions):
                        print(f'FooPlayer._evaluate_node: Pruned robber actions from {len(actions)} to {len(pruned)}')
                        actions = pruned
                    else:
                        # Skip pruning if not meaningful
                        pass
                else:
                    # Small action set: do not prune
                    pass
            except Exception as e:
                print(f'FooPlayer._evaluate_node: prune_robber_actions failed: {e}')

        # Determine maximizing/minimizing player
        # Defensive: if actions is empty now, fall back to heuristic
        if not actions:
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

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
