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
    def __init__(self, name=None, max_depth: int = 5, node_budget: int = 15000):
        """
        FooPlayer uses iterative-deepening Expectimax with specialized heuristics
        and search optimizations (move ordering, robber pruning, caching).

        Args:
            name: Optional player name
            max_depth: Maximum plies to search (iterative deepening will grow to this)
            node_budget: Hard limit on number of recursive nodes to evaluate per decide() call
        """
        # Initialize with BLUE by default to preserve compatibility
        super().__init__(Color.BLUE, name)
        self.max_depth = max_depth
        self.node_budget = node_budget

    def _get_game_phase(self, game: Game) -> str:
        """
        Determine the game phase (early, mid, late) based on an available turn counter.
        This tries several common locations for a turn counter to be robust.
        """
        try:
            # Prefer a direct attribute if present
            turn_count = getattr(game, 'turn_count', None)
            if turn_count is None:
                # Try nested state structures
                st = getattr(game, 'state', None)
                if isinstance(st, dict):
                    turn_count = st.get('turn_count') or st.get('turn') or st.get('tick')
                else:
                    turn_count = getattr(st, 'turn_count', None) if st is not None else None

            if turn_count is None:
                # Fall back to 0 if not available
                turn_count = 0

            turn_count = int(turn_count)
        except Exception:
            turn_count = 0

        if turn_count < 20:
            return 'early'
        elif turn_count < 40:
            return 'mid'
        else:
            return 'late'

    def decide(self, game: Game, playable_actions):
        """
        Choose an action using iterative-deepening Expectimax with probabilistic simulation.

        This implementation follows the Strategizer's recommendations:
        - Dynamic heuristic weighting by game phase (early/mid/late)
        - Aggressive move ordering prioritizing high-value actions
        - Iterative deepening up to self.max_depth (capped at 5)
        - Global node budget enforced
        - Use of list_prunned_actions and prune_robber_actions to reduce branching
        """
        actions = list(playable_actions) if playable_actions is not None else []

        if len(actions) == 0:
            print('FooPlayer.decide: No playable actions available; returning None')
            return None

        # Cap maximum search depth to [1..5]
        MAX_DEPTH = max(1, min(5, self.max_depth))
        NODE_BUDGET = max(100, self.node_budget)

        # Primary base value function
        base_value_fn = base_fn()

        # Transposition cache: (state_repr, depth) -> value
        cache = {}

        # Node counter and timing
        node_count = 0
        start_time = time.time()

        # Helper to produce a reproducible cache key for a game state
        def _state_key(g: Game) -> str:
            try:
                return repr(g.state)
            except Exception:
                try:
                    return repr(g)
                except Exception:
                    return str(id(g))

        # Move ordering heuristic (higher = more promising)
        def _action_priority(act: Action) -> float:
            try:
                at = act.action_type
                # Aggressive priorities recommended by the Strategizer
                if at == ActionType.BUILD_CITY:
                    return 150.0
                if at == ActionType.PLAY_DEV_CARD:
                    return 140.0
                if at == ActionType.PLAY_KNIGHT_CARD:
                    # Prioritize knight plays to move robber / gain army
                    return 140.0
                if at == ActionType.BUILD_SETTLEMENT:
                    return 130.0
                if at == ActionType.BUILD_ROAD:
                    return 100.0
                if at == ActionType.BUY_DEV_CARD:
                    return 80.0
                if at == ActionType.MOVE_ROBBER:
                    return 70.0
                if at == ActionType.TRADE:
                    return 60.0
                if at == ActionType.ROLL:
                    return 50.0
                if at == ActionType.END_TURN:
                    # Strongly deprioritize ending the turn
                    return -200.0
            except Exception:
                pass
            return 0.0

        # Specialized heuristics as suggested by Strategizer.
        # Each returns a raw signal; combined_heuristic will apply the configured weights.

        def expansion_heuristic(g: Game, color: Color) -> float:
            """Estimate long-term expansion potential using value_production.
            Returns raw production signal (not weighted).
            """
            try:
                sample = getattr(g, 'state', g)
                player_name = getattr(self, 'name', 'P0')
                prod = value_production(sample, player_name, include_variety=True)
                return float(prod)
            except Exception as e:
                # Be conservative on failures
                # print(f'FooPlayer.expansion_heuristic failed: {e}')
                return 0.0

        def robber_heuristic(g: Game, color: Color) -> float:
            """Estimate impact of robber placement by measuring opponent production.
            Returns the maximum opponent production (raw), combined_heuristic will weight it negatively.
            """
            try:
                sample = getattr(g, 'state', g)
                max_opponent_prod = 0.0
                # Iterate over known colors and measure production; skip our color
                for opp in list(Color):
                    if opp == color:
                        continue
                    try:
                        opp_name = getattr(self, 'name', f'P{opp.value}')
                        p = value_production(sample, opp_name, include_variety=False)
                        max_opponent_prod = max(max_opponent_prod, float(p))
                    except Exception:
                        continue
                return float(max_opponent_prod)
            except Exception:
                return 0.0

        def dev_card_heuristic(g: Game, color: Color) -> float:
            """Prefer states where playing certain dev cards (MONOPOLY, ROAD_BUILDING)
            is likely to be impactful. This returns a raw signal (counts weighted) and
            will be scaled by the combined heuristic based on game phase.
            """
            try:
                sample = getattr(g, 'state', None)
                player_name = getattr(self, 'name', 'P0')
                if sample is None:
                    return 0.0

                # Defensive access to dev card counts
                devs = None
                if isinstance(sample, dict):
                    devs = sample.get('dev_cards') or sample.get('devs')
                else:
                    devs = getattr(g, 'dev_cards', None) or getattr(sample, 'dev_cards', None)

                if not devs:
                    return 0.0

                # Extract counts robustly
                count_mon = 0
                count_rb = 0
                try:
                    # devs might be structured as {player_name: {'MONOPOLY': n, ...}, ...}
                    count_mon = int(devs.get(player_name, {}).get('MONOPOLY', 0))
                    count_rb = int(devs.get(player_name, {}).get('ROAD_BUILDING', 0))
                except Exception:
                    try:
                        # Or keyed by color
                        count_mon = int(devs.get(color, {}).get('MONOPOLY', 0))
                        count_rb = int(devs.get(color, {}).get('ROAD_BUILDING', 0))
                    except Exception:
                        count_mon = 0
                        count_rb = 0

                # Return a raw combined signal: weight monopoly higher than road building
                return float(count_mon) * 1.0 + float(count_rb) * 0.8
            except Exception:
                return 0.0

        # Combined heuristic: base value + phase-weighted specialized heuristics
        def combined_heuristic(g: Game, color: Color) -> float:
            # Base evaluation
            try:
                base_val = base_value_fn(g, color)
            except Exception as e:
                print(f'FooPlayer.combined_heuristic: base_fn failed: {e}')
                base_val = -1e9

            # Determine game phase for dynamic weights
            phase = self._get_game_phase(g)

            # Phase-specific weights
            if phase == 'early':
                exp_w = 0.4
                rob_w = -0.2
                dev_w = 0.2
            else:  # mid or late
                exp_w = 0.2
                rob_w = -0.4
                dev_w = 0.6

            try:
                exp_v = exp_w * expansion_heuristic(g, color)
                rob_v = rob_w * robber_heuristic(g, color)
                dev_v = dev_w * dev_card_heuristic(g, color)
            except Exception as e:
                print(f'FooPlayer.combined_heuristic: specialized heuristic failed: {e}')
                exp_v = rob_v = dev_v = 0.0

            # Debug print of heuristic components for traceability
            # (Keep prints moderate to avoid flooding)
            # print(f'FooPlayer.heuristic: phase={phase}, base={base_val:.2f}, exp={exp_v:.2f}, rob={rob_v:.2f}, dev={dev_v:.2f}')

            return base_val + exp_v + rob_v + dev_v

        # Expectimax with caching and node budget. Uses combined_heuristic at leaves.
        def expectimax(node_game: Game, depth: int) -> float:
            nonlocal node_count

            # Enforce node budget (global across iterative deepening)
            node_count += 1
            if node_count > NODE_BUDGET:
                # Return a heuristic estimate if we're out of budget
                return combined_heuristic(node_game, self.color)

            key = (_state_key(node_game), depth)
            if key in cache:
                return cache[key]

            # Terminal check (winning_color) if available
            try:
                winner = None
                try:
                    winner = node_game.winning_color()
                except Exception:
                    winner = None
                if winner is not None:
                    val = combined_heuristic(node_game, self.color)
                    cache[key] = val
                    return val
            except Exception as e:
                print(f'FooPlayer.expectimax: winner check failed: {e}')

            # Depth cutoff
            if depth == 0:
                val = combined_heuristic(node_game, self.color)
                cache[key] = val
                return val

            # Get pruned action list
            try:
                node_actions = list_prunned_actions(node_game)
            except Exception as e:
                print(f'FooPlayer.expectimax: list_prunned_actions failed: {e}')
                node_actions = []

            if not node_actions:
                val = combined_heuristic(node_game, self.color)
                cache[key] = val
                return val

            # If robber moves exist, prune them
            try:
                if any((getattr(a, 'action_type', None) == ActionType.MOVE_ROBBER) for a in node_actions):
                    node_actions = prune_robber_actions(self.color, node_game, node_actions)
            except Exception as e:
                print(f'FooPlayer.expectimax: prune_robber_actions failed: {e}')

            # Move ordering
            try:
                node_actions.sort(key=_action_priority, reverse=True)
            except Exception:
                pass

            # Determine node type: MAX if acting color == our color
            try:
                node_color = node_actions[0].color
                is_max = (node_color == self.color)
            except Exception:
                is_max = False

            if is_max:
                best_value = -math.inf
                for act in node_actions:
                    # Expand probabilistic outcomes
                    try:
                        outcomes = execute_spectrum(node_game, act)
                    except Exception:
                        try:
                            outcomes = execute_deterministic(node_game, act)
                        except Exception as e:
                            # Skip actions that cannot be executed
                            # print(f'FooPlayer.expectimax: action execution failed (max) for {act}: {e}')
                            continue

                    if not outcomes:
                        continue

                    expected = 0.0
                    for (g_after, prob) in outcomes:
                        try:
                            val = expectimax(g_after, depth - 1)
                        except Exception as e:
                            # On recursion error, give a conservative low value
                            # print(f'FooPlayer.expectimax: recursion error (max) for outcome: {e}')
                            val = -1e9
                        expected += prob * val

                    if expected > best_value:
                        best_value = expected

                    # Early stopping if node budget exhausted
                    if node_count > NODE_BUDGET:
                        break

                cache[key] = best_value
                return best_value
            else:
                # MIN node: model opponent as adversarial minimizing our heuristic
                worst_value = math.inf
                for act in node_actions:
                    try:
                        outcomes = execute_spectrum(node_game, act)
                    except Exception:
                        try:
                            outcomes = execute_deterministic(node_game, act)
                        except Exception as e:
                            # print(f'FooPlayer.expectimax: action execution failed (min) for {act}: {e}')
                            continue

                    if not outcomes:
                        continue

                    expected = 0.0
                    for (g_after, prob) in outcomes:
                        try:
                            val = expectimax(g_after, depth - 1)
                        except Exception as e:
                            # print(f'FooPlayer.expectimax: recursion error (min) for outcome: {e}')
                            val = 1e9
                        expected += prob * val

                    if expected < worst_value:
                        worst_value = expected

                    if node_count > NODE_BUDGET:
                        break

                cache[key] = worst_value
                return worst_value

        # Iterative deepening. Use a global node budget across all depths.
        best_action = None
        best_value = -math.inf
        depth_reached = 0

        # Determine current game phase for debugging and heuristic selection
        current_phase = self._get_game_phase(game)
        print(f'FooPlayer.decide: starting iterative deepening (phase={current_phase}, max_depth={MAX_DEPTH}, node_budget={NODE_BUDGET})')

        for depth in range(1, MAX_DEPTH + 1):
            print(f'FooPlayer.decide: Iterative deepening at depth {depth}')
            depth_reached = depth

            # Evaluate top-level actions in move-ordered sequence to get good bounds early
            ordered_actions = sorted(actions, key=_action_priority, reverse=True)

            for idx, action in enumerate(ordered_actions):
                if node_count > NODE_BUDGET:
                    print('FooPlayer.decide: Global node budget reached; stopping search')
                    break

                expected_value = -math.inf
                try:
                    game_copy = copy_game(game)
                    try:
                        outcomes = execute_spectrum(game_copy, action)
                    except Exception:
                        # Fallback to deterministic
                        try:
                            outcomes = execute_deterministic(game_copy, action)
                        except Exception as e:
                            print(f'FooPlayer.decide: execute_deterministic also failed for action {action}: {e}')
                            outcomes = []

                    if not outcomes:
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

                # Update best action found so far (across depths)
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action

                # Respect global budget
                if node_count > NODE_BUDGET:
                    break

            # Stop deepening if budget exhausted
            if node_count > NODE_BUDGET:
                break

        if best_action is None and len(actions) > 0:
            print('FooPlayer: All evaluations failed or were skipped; defaulting to first action')
            best_action = actions[0]

        elapsed = time.time() - start_time
        print(f'FooPlayer: Chosen action = {best_action} with value {best_value} (depth reached {depth_reached}, nodes {node_count}, time {elapsed:.3f}s)')
        return best_action
