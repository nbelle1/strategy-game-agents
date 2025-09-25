from .adapters import (
    Game,
    Player,
    Color,
    Action,
    ActionType,
    copy_game,
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    list_prunned_actions,
    prune_robber_actions,
    base_fn,
    value_production,
    get_value_fn,
)

import math
import time
from typing import List, Tuple


class FooPlayer(Player):
    def __init__(self, name=None, max_depth: int = 3, node_budget: int = 8000):
        """
        FooPlayer uses an iterative-deepening Expectimax search with simple move ordering,
        robber-action pruning, and a transposition cache to control search cost.

        Args:
            name: Optional player name
            max_depth: Maximum plies to search (iterative deepening will grow to this)
            node_budget: Hard limit on number of recursive nodes to evaluate per decide() call
        """
        # Initialize as the BLUE player by default (keeps compatibility with previous versions)
        super().__init__(Color.BLUE, name)
        # Configurable search parameters
        self.max_depth = max_depth
        self.node_budget = node_budget

    def decide(self, game: Game, playable_actions):
        """
        Choose an action using iterative-deepening Expectimax with probabilistic simulation.

        Key features implemented:
        - Iterative deepening from depth=1 up to self.max_depth (inclusive).
        - Transposition cache (simple dict keyed by game.state repr and depth) to avoid
          re-evaluating identical subtrees.
        - Move ordering to explore promising actions first (helps pruning & early good bounds).
        - Robber action pruning via adapters.prune_robber_actions when robber moves appear.
        - Uses execute_spectrum for probabilistic outcomes; falls back to execute_deterministic.
        - Evaluates leaf nodes with base_fn() (can be swapped with get_value_fn if desired).

        Notes on safety and adapters usage:
        - Only calls functions exposed by adapters.py (no direct catanatron imports).
        - Attempts to be robust to adapter failures by catching exceptions and using
          reasonable fallbacks.

        Args:
            game: Current Game state (read-only from caller's perspective)
            playable_actions: Iterable of legal actions to choose from
        Returns:
            An Action chosen from playable_actions or None if no actions available
        """
        actions = list(playable_actions) if playable_actions is not None else []

        if len(actions) == 0:
            print('FooPlayer.decide: No playable actions available; returning None')
            return None

        # Parameters for search
        MAX_DEPTH = max(1, min(4, self.max_depth))  # cap to [1..4] to avoid runaway costs
        NODE_BUDGET = max(100, self.node_budget)

        # Value function factory (primary heuristic)
        value_fn = base_fn()

        # Transposition cache: maps (state_repr, depth) -> value
        cache = {}

        # Node evaluation counter and timing
        node_count = 0
        start_time = time.time()

        # Small helper to generate a cache key for a game state
        def _state_key(g: Game) -> str:
            # Game implementations typically expose a serializable .state; using repr as a fallback
            try:
                return repr(g.state)
            except Exception:
                try:
                    return repr(g)
                except Exception:
                    return str(id(g))

        # Quick move-ordering heuristic to try promising moves first. This helps the search find
        # good solutions early which improves the utility of iterative deepening.
        def _action_priority(act: Action) -> float:
            # Higher values are explored first.
            try:
                at = act.action_type
                # Prioritize playing dev cards, building settlements/cities, then roads.
                if at == ActionType.PLAY_DEV_CARD:
                    return 100.0
                if at == ActionType.BUILD_SETTLEMENT:
                    return 90.0
                if at == ActionType.BUILD_CITY:
                    return 95.0
                if at == ActionType.BUILD_ROAD:
                    return 50.0
                if at == ActionType.BUY_DEV_CARD:
                    return 45.0
                if at == ActionType.MOVE_ROBBER:
                    return 30.0
                if at == ActionType.TRADE:
                    return 20.0
                if at == ActionType.ROLL:
                    return 10.0
            except Exception:
                pass
            return 0.0

        # Custom heuristic wrapper that augments base_fn with small additional signals.
        # We avoid heavy rule-based logic; additional terms are small nudges to prefer
        # productive states. This remains conservative and primarily relies on base_fn.
        def custom_heuristic(g: Game) -> float:
            try:
                base_val = value_fn(g, self.color)
            except Exception as e:
                print(f'FooPlayer.custom_heuristic: base_fn failed: {e}')
                base_val = -1e9

            # Small bonus for production (best-effort). We attempt to call value_production if possible.
            prod_bonus = 0.0
            try:
                # value_production expects a sample and player_name; many adapters expose this utility
                # but we don't know the exact sample shape; call with g.state if available.
                sample = getattr(g, 'state', g)
                prod = value_production(sample, getattr(self, 'name', 'P0'), include_variety=True)
                # Scale down production so it doesn't overwhelm base_fn
                prod_bonus = 0.01 * float(prod)
            except Exception:
                # If unavailable, silently ignore.
                prod_bonus = 0.0

            return base_val + prod_bonus

        # Expectimax implementation with a node budget and caching.
        def expectimax(node_game: Game, depth: int) -> float:
            nonlocal node_count

            # Enforce node budget
            node_count += 1
            if node_count > NODE_BUDGET:
                # Budget exhausted; return a heuristic estimate of current node to stop deep recursion
                print('FooPlayer.expectimax: node budget exhausted; returning heuristic')
                return custom_heuristic(node_game)

            # Check cache
            key = (_state_key(node_game), depth)
            if key in cache:
                return cache[key]

            # Terminal / winner check
            try:
                winner = None
                try:
                    winner = node_game.winning_color()
                except Exception:
                    winner = None
                if winner is not None:
                    val = custom_heuristic(node_game)
                    cache[key] = val
                    return val
            except Exception as e:
                print(f'FooPlayer.expectimax: winner check failed: {e}')

            # Depth limit -> evaluate
            if depth == 0:
                val = custom_heuristic(node_game)
                cache[key] = val
                return val

            # Get pruned action list
            try:
                node_actions = list_prunned_actions(node_game)
            except Exception as e:
                print(f'FooPlayer.expectimax: list_prunned_actions failed: {e}')
                node_actions = []

            if not node_actions:
                val = custom_heuristic(node_game)
                cache[key] = val
                return val

            # If robber actions exist, prune them to focus on impactful robber moves
            try:
                if any((getattr(a, 'action_type', None) == ActionType.MOVE_ROBBER) for a in node_actions):
                    node_actions = prune_robber_actions(self.color, node_game, node_actions)
            except Exception as e:
                # If pruning fails, continue with unpruned actions
                print(f'FooPlayer.expectimax: prune_robber_actions failed: {e}')

            # Move ordering: sort by priority descending
            try:
                node_actions.sort(key=_action_priority, reverse=True)
            except Exception:
                pass

            # Determine node type: MAX if acting color is our color, else MIN
            try:
                node_color = node_actions[0].color
                is_max = (node_color == self.color)
            except Exception:
                # Fallback: assume opponent node to be conservative
                is_max = False

            if is_max:
                best_value = -math.inf
                for act in node_actions:
                    # Expand outcomes
                    try:
                        outcomes = execute_spectrum(node_game, act)
                    except Exception:
                        try:
                            outcomes = execute_deterministic(node_game, act)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: action execution failed for {act}: {e}')
                            continue

                    # If no outcomes, skip
                    if not outcomes:
                        continue

                    expected = 0.0
                    for (g_after, prob) in outcomes:
                        try:
                            val = expectimax(g_after, depth - 1)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: recursion error for outcome: {e}')
                            val = -1e9
                        expected += prob * val

                    if expected > best_value:
                        best_value = expected

                    # Small optimization: if best_value already extremely high, we could break early
                    # but we avoid aggressive pruning to keep semantics correct.

                cache[key] = best_value
                return best_value
            else:
                # MIN node: adversarial opponent minimizing our value
                worst_value = math.inf
                for act in node_actions:
                    try:
                        outcomes = execute_spectrum(node_game, act)
                    except Exception:
                        try:
                            outcomes = execute_deterministic(node_game, act)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: action execution failed for {act}: {e}')
                            continue

                    if not outcomes:
                        continue

                    expected = 0.0
                    for (g_after, prob) in outcomes:
                        try:
                            val = expectimax(g_after, depth - 1)
                        except Exception as e:
                            print(f'FooPlayer.expectimax: recursion error for outcome: {e}')
                            val = 1e9
                        expected += prob * val

                    if expected < worst_value:
                        worst_value = expected

                cache[key] = worst_value
                return worst_value

        # Iterative deepening over increasing depths; keep best action found at each depth.
        best_action = None
        best_value = -math.inf

        # Preserve results across depths using the same cache to accelerate deeper searches
        for depth in range(1, MAX_DEPTH + 1):
            # Reset node counter for each depth iteration to enforce per-depth budget
            node_count = 0
            print(f'FooPlayer.decide: Iterative deepening at depth {depth}')

            # Evaluate each top-level action
            for idx, action in enumerate(actions):
                expected_value = -math.inf
                try:
                    # Work on a copy to avoid side-effects from adapters
                    game_copy = copy_game(game)

                    # Expand top-level action outcomes
                    try:
                        outcomes = execute_spectrum(game_copy, action)
                    except Exception as e:
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
                                val = expectimax(g_after, depth - 1)
                            except Exception as e:
                                print(f'FooPlayer.decide: expectimax error on outcome of action {action}: {e}')
                                val = -1e9
                            expected_value += prob * val

                    print(f'FooPlayer: Depth {depth} Top-level Action #{idx} = {action} => expected value {expected_value}')

                except Exception as e:
                    print(f'FooPlayer: Exception while evaluating top-level action {action}: {e}')
                    expected_value = -math.inf

                # Update best action when we find a better expected value
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action

                # If node budget exhausted, break early
                if node_count > NODE_BUDGET:
                    print('FooPlayer.decide: node budget exceeded during evaluations; breaking depth loop')
                    break

            # If node budget exhausted at this depth, stop iterative deepening
            if node_count > NODE_BUDGET:
                break

        # Fallback: if evaluations all failed, pick the first action
        if best_action is None and len(actions) > 0:
            print('FooPlayer: All evaluations failed; defaulting to first action')
            best_action = actions[0]

        print(f'FooPlayer: Chosen action = {best_action} with value {best_value} (search depth up to {depth})')
        return best_action
