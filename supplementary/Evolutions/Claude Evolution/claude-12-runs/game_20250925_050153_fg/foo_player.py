import os
from typing import Iterable, List, Tuple, Dict, Any
import traceback

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
    - Per-decision transposition cache to reduce duplicate evaluations.
    - Plenty of debugging print statements to trace decisions and node counts.
    """

    # Tunable parameters for defensive limits and small tactical nudges
    PER_ACTION_EXPANSION_LIMIT = 800  # allow more nodes per candidate so roads can be evaluated
    ROBBER_PRUNE_MIN_ACTIONS = 12  # only prune robber actions when branching is quite large
    ROAD_BONUS = 20.0  # stronger bonus to favor road building when otherwise equal

    # Actions considered low-impact (we'll evaluate them more shallowly)
    LOW_IMPACT_ACTION_NAMES = {
        'PLAY_YEAR_OF_PLENTY',
        'PLAY_MONOPOLY',
        'PLAY_SOLDIER',
        'PLAY_ROAD_BUILD',
    }

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

        # Per-decision transposition cache (initialized in decide)
        self._eval_cache: Dict[Tuple[str, int], float] = {}

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

        # Reset expansion counter and per-decision cache
        self._node_expansions = 0
        self._eval_cache = {}

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
                    # Accept pruning only if it doesn't collapse options to too few
                    if pruned and len(pruned) >= max(2, len(actions) // 4):
                        print(f'FooPlayer.decide: Pruned robber actions from {len(actions)} to {len(pruned)}')
                        actions = pruned
                    else:
                        print('FooPlayer.decide: prune_robber_actions returned overly aggressive pruning or no meaningful reduction, skipping')
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
                    # Decide whether to use reduced depth for low-impact actions
                    action_type = getattr(a, 'action_type', None)
                    action_name = getattr(action_type, 'name', '') if action_type is not None else ''
                    eval_depth = self.lookahead_depth
                    if action_name in self.LOW_IMPACT_ACTION_NAMES:
                        eval_depth = 1

                    # Soft per-candidate expansion cap: if a single candidate causes too many
                    # node expansions, abort its full lookahead and fallback to heuristic.
                    start_nodes = self._node_expansions
                    val = self._expected_value_for_action(game, a, eval_depth)
                    used_nodes = self._node_expansions - start_nodes
                    if used_nodes > self.PER_ACTION_EXPANSION_LIMIT:
                        # Abortative fallback: use heuristic evaluation instead of runaway search
                        try:
                            fallback_val = float(self.value_fn(game, self.color))
                        except Exception:
                            fallback_val = 0.0
                        print(f'FooPlayer.decide: Candidate {idx} ({action_name}) used {used_nodes} nodes, exceeding limit {self.PER_ACTION_EXPANSION_LIMIT}. Using fallback heuristic {fallback_val}')
                        val = fallback_val

                    # Stronger tactical nudge: prefer roads to improve expansion
                    if action_type == ActionType.BUILD_ROAD:
                        val += self.ROAD_BONUS

                    print(f'  Candidate {idx}: expected_value={val} action_type={action_type}')
                    if val > best_s:
                        best_s = val
                        best_a = a
                except Exception as e:
                    print(f'FooPlayer.decide: Exception evaluating candidate {a}: {e}')
                    print(traceback.format_exc())
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
            print(traceback.format_exc())

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
                        # Dev cards can be noisy; allow slightly reduced depth
                        val = self._expected_value_for_action(game, a, max(1, self.lookahead_depth - 1))
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
                        print(traceback.format_exc())
                if best_dev:
                    print(f'FooPlayer.decide: Chosen dev card action={best_dev} score={best_dev_score} node_expansions={self._node_expansions}')
                    return best_dev
        except Exception as e:
            print(f'FooPlayer.decide: Exception evaluating dev cards: {e}')
            print(traceback.format_exc())

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
            print(traceback.format_exc())

        # Absolute fallback: evaluate all remaining actions including maritime trades
        try:
            chosen, score = evaluate_candidates(actions)
            print(f'FooPlayer.decide: Fallback chosen action={chosen} score={score} node_expansions={self._node_expansions}')
            if chosen:
                return chosen
        except Exception as e:
            print(f'FooPlayer.decide: Exception in final fallback evaluation: {e}')
            print(traceback.format_exc())

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
        except Exception as e:
            print(f'FooPlayer._expected_value_for_action: copy_game failed for action {action}: {e}')
            print(traceback.format_exc())
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        try:
            outcomes = execute_deterministic(game_copy, action)
        except Exception as e:
            print(f'FooPlayer._expected_value_for_action: execute_deterministic failed for action {action}: {e}')
            print(traceback.format_exc())
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        expected_value = 0.0
        # outcomes is a list of (game, prob) tuples; iterate defensively
        if not outcomes:
            try:
                return float(self.value_fn(game, self.color))
            except Exception:
                return 0.0

        for (outcome_game, prob) in outcomes:
            try:
                node_value = self._evaluate_node(outcome_game, max(0, depth - 1))
            except Exception as e:
                print(f'FooPlayer._expected_value_for_action: _evaluate_node failed for outcome: {e}')
                print(traceback.format_exc())
                try:
                    node_value = float(self.value_fn(outcome_game, self.color))
                except Exception:
                    node_value = 0.0
            try:
                expected_value += (prob or 0.0) * node_value
            except Exception:
                # Defensive: if prob is malformed, treat as zero contribution and continue
                print('FooPlayer._expected_value_for_action: malformed probability, skipping contribution')
        return expected_value

    def _evaluate_node(self, game: Game, depth: int) -> float:
        """Recursive evaluator that returns heuristic value for a game state.

        This routine uses list_prunned_actions to reduce the branching factor in
        a conservative way and also uses prune_robber_actions defensively. It
        will maximize for nodes where the current actor is this player's color
        and minimize otherwise.
        """
        # Attempt to build a cache key from the game state representation
        try:
            state_repr = repr(game.state)
        except Exception:
            try:
                state_repr = str(id(game))
            except Exception:
                state_repr = ''

        cache_key = (state_repr, depth)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        # Count node expansion
        self._node_expansions += 1

        # Base case: evaluate with heuristic
        if depth <= 0:
            try:
                val = float(self.value_fn(game, self.color))
                self._eval_cache[cache_key] = val
                return val
            except Exception as e:
                print(f'FooPlayer._evaluate_node: value_fn raised exception: {e}')
                print(traceback.format_exc())
                self._eval_cache[cache_key] = 0.0
                return 0.0

        # Get pruned actions for this state; materialize into a list
        try:
            actions = list_prunned_actions(game)
            actions = list(actions) if actions is not None else []
        except Exception as e:
            print(f'FooPlayer._evaluate_node: list_prunned_actions failed: {e}')
            print(traceback.format_exc())
            try:
                val = float(self.value_fn(game, self.color))
                self._eval_cache[cache_key] = val
                return val
            except Exception:
                self._eval_cache[cache_key] = 0.0
                return 0.0

        if not actions:
            try:
                val = float(self.value_fn(game, self.color))
                self._eval_cache[cache_key] = val
                return val
            except Exception:
                self._eval_cache[cache_key] = 0.0
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
                if len(actions) > self.ROBBER_PRUNE_MIN_ACTIONS:
                    current_color = actions[0].color
                    pruned = prune_robber_actions(current_color, game, actions)
                    pruned = list(pruned) if pruned is not None else pruned
                    if pruned and len(pruned) >= max(2, len(actions) // 4):
                        print(f'FooPlayer._evaluate_node: Pruned robber actions from {len(actions)} to {len(pruned)}')
                        actions = pruned
                    else:
                        # Skip overly aggressive pruning
                        pass
                else:
                    pass
            except Exception as e:
                print(f'FooPlayer._evaluate_node: prune_robber_actions failed: {e}')
                print(traceback.format_exc())

        # Determine maximizing/minimizing player
        if not actions:
            try:
                val = float(self.value_fn(game, self.color))
                self._eval_cache[cache_key] = val
                return val
            except Exception:
                self._eval_cache[cache_key] = 0.0
                return 0.0

        current_actor_color = actions[0].color
        is_maximizing = (current_actor_color == self.color)

        best_value = float('-inf') if is_maximizing else float('inf')

        for action in actions:
            try:
                game_copy = copy_game(game)
            except Exception as e:
                print(f'FooPlayer._evaluate_node: copy_game failed for action {action}: {e}')
                print(traceback.format_exc())
                continue

            try:
                outcomes = execute_deterministic(game_copy, action)
            except Exception as e:
                print(f'FooPlayer._evaluate_node: execute_deterministic failed for action {action}: {e}')
                print(traceback.format_exc())
                continue

            expected = 0.0
            if not outcomes:
                # If an action produces no outcomes, skip it defensively
                continue

            for (outcome_game, prob) in outcomes:
                try:
                    val = self._evaluate_node(outcome_game, depth - 1)
                except Exception as e:
                    print(f'FooPlayer._evaluate_node: recursive _evaluate_node failed for an outcome: {e}')
                    print(traceback.format_exc())
                    try:
                        val = float(self.value_fn(outcome_game, self.color))
                    except Exception:
                        val = 0.0
                try:
                    expected += (prob or 0.0) * val
                except Exception:
                    print('FooPlayer._evaluate_node: malformed probability in outcomes, skipping contribution')

            if is_maximizing:
                if expected > best_value:
                    best_value = expected
            else:
                if expected < best_value:
                    best_value = expected

        # If evaluation failed to set a value, fall back to heuristic
        if best_value == float('inf') or best_value == float('-inf'):
            try:
                val = float(self.value_fn(game, self.color))
                self._eval_cache[cache_key] = val
                return val
            except Exception:
                self._eval_cache[cache_key] = 0.0
                return 0.0

        # Cache and return
        self._eval_cache[cache_key] = best_value
        return best_value
