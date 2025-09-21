import time
from .adapters import (
    Player,
    Color,
    Action,
    ActionType,
    copy_game,
    execute,
    make_value_fn,
    DEFAULT_WEIGHTS,
    chance_children_for_action,
    chance_children,
    opponent_freqdeck,
    player_build_nodes,
    dev_cards_in_hand,
)


class FooPlayer(Player):
    """
    FooPlayer: an iterative improvement over the trivial "first-action" bot.

    Implemented strategy (1-ply lookahead + chance-awareness):
    - Filter out END_TURN when there are available build/play actions.
    - Cap number of actions evaluated (default 20) for performance.
    - For each candidate action:
      * copy the game, execute the action on the copy,
      * if the action produces stochastic outcomes use chance_children_for_action
        to compute an expected score (fallback to Monte Carlo / deterministic score),
      * otherwise score deterministically with make_value_fn("base_fn").
    - Choose the action with the highest expected score; tie-break by a prioritized
      action ordering that prefers building and development actions.

    Notes:
    - This class uses the adapters surface only. The code is defensive about
      varying adapter signatures and enum name variants across forks.
    - Print statements are included to help trace decision-making during
      experiments. Keep them for debugging; remove or lower verbosity when
      logs become too large.
    """

    # Heuristic bonus params (tunable)
    CITY_BONUS = 5.0  # Increased for city-bias experiment
    SETTLEMENT_BONUS = 1.0
    ROAD_EXTENSION_BONUS = 0.8
    ROAD_CONNECTS_SETTLEMENT_BONUS = 1.5
    KNIGHT_PLAY_BONUS = 2.5
    MAX_BONUS_SCALE = 10.0

    def __init__(self, name: str | None = None):
        super().__init__(Color.BLUE, name)
        # one-time introspection flag
        self._did_inspect = False

    # --- Helper heuristics ---
    def estimate_settlement_node_value(self, game, node_id, color):
        """
        Cheap proxy for settlement node value: sum of adjacent tile pip probabilities.
        Defensive: adapts to several board APIs; returns float >= 0.
        """
        total = 0.0
        try:
            # Many forks expose game.state.board.node_tiles(node_id) -> list of tile ids/objects
            tiles = game.state.board.node_tiles(node_id)
            for tile in tiles:
                pip = None
                # try tile.pip or related attribute
                try:
                    pip = getattr(tile, 'pip', None)
                except Exception:
                    pip = None
                if pip is None:
                    # try adapter-provided helper method on board
                    try:
                        pip = game.state.board.tile_pip_probability(tile)
                    except Exception:
                        pip = 1.0
                try:
                    total += float(pip)
                except Exception:
                    total += 0.0
        except Exception:
            total = 0.0
        return total

    def road_extends_toward_open_spot(self, game, edge_id, color):
        """
        Return True if edge touches any node without owner (open spot).
        Uses defensive API calls and logs for debugging.
        """
        try:
            board = game.state.board
            # Try canonical API
            nodes = None
            try:
                nodes = board.edge_nodes(edge_id)
            except Exception:
                # fallback: try edges mapping
                try:
                    nodes = board.edges[edge_id].nodes
                except Exception:
                    nodes = None

            # Debug print of nodes and owners for troubleshooting
            try:
                owners = []
                if nodes:
                    for n in nodes:
                        try:
                            owner = board.node_owner(n)
                        except Exception:
                            try:
                                owner = board.nodes[n].owner
                            except Exception:
                                owner = None
                        owners.append(owner)
                    print(f"[foo_player-debug] Edge {edge_id} nodes: {nodes}, Owners: {owners}")
            except Exception:
                pass

            if not nodes:
                return False

            for n in nodes:
                try:
                    owner = board.node_owner(n)
                except Exception:
                    try:
                        owner = board.nodes[n].owner
                    except Exception:
                        owner = None
                if owner is None:
                    return True
        except Exception:
            pass
        return False

    def will_gain_largest_army(self, game, color):
        """
        Return True if playing one more knight would give this player the largest army.
        Adds logging for debugging.
        """
        try:
            pstate = game.state.players[color]
            my_knights = getattr(pstate, 'knights_played', 0)
            opps = []
            for c, p in game.state.players.items():
                if c == color:
                    continue
                opps.append(getattr(p, 'knights_played', 0))
            opp_knights = max(opps) if opps else 0
            # Debug log
            print(f"[foo_player] Knights: me={my_knights}, opp_max={opp_knights}")
            return (my_knights + 1) > opp_knights
        except Exception as e:
            print(f"[foo_player] will_gain_largest_army error: {e}")
            return False

    def decide(self, game, playable_actions):
        """
        Decide which action to take from playable_actions.

        Args:
            game: full Game object (read-only for this method).
            playable_actions: iterable of Action objects available this turn.
        Returns:
            One Action chosen from playable_actions.
        """
        # Configurable knobs
        max_actions = 20  # cap number of candidate actions to evaluate
        samples_mc = 10  # Monte Carlo fallback samples per chance action
        max_simulations = 200  # cap total simulations across all actions
        time_budget_per_action = 0.5  # seconds; abort expensive action sims past this

        # Helper: robustly resolve ActionType members across forks
        def resolve_action_type(*names):
            for n in names:
                if hasattr(ActionType, n):
                    return getattr(ActionType, n)
            return None

        # Common action types (fallback-safe)
        AT_BUILD_CITY = resolve_action_type('BUILD_CITY')
        AT_BUILD_SETTLEMENT = resolve_action_type('BUILD_SETTLEMENT', 'BUILD_SETTLES')
        AT_BUILD_ROAD = resolve_action_type('BUILD_ROAD', 'BUILD_ROADS')
        AT_BUY_DEV = resolve_action_type('BUY_DEVELOPMENT_CARD', 'BUY_DEV_CARD', 'BUY_DEVELOPMENT')
        AT_PLAY_DEV = resolve_action_type('PLAY_DEVELOPMENT_CARD', 'PLAY_DEV_CARD', 'PLAY_DEVELOPMENT')
        AT_TRADE = resolve_action_type('TRADE')
        AT_MOVE_ROBBER = resolve_action_type('MOVE_ROBBER', 'MOVE_ROBBER_TO_TILE')
        AT_ROLL = resolve_action_type('ROLL')
        AT_END_TURN = resolve_action_type('END_TURN')

        # Priority mapping for tie-breaking. Higher -> more preferred.
        _priority_map = {
            AT_BUILD_CITY: 100,
            AT_BUILD_SETTLEMENT: 90,
            AT_BUILD_ROAD: 80,
            AT_BUY_DEV: 70,
            AT_PLAY_DEV: 60,
            AT_TRADE: 50,
            AT_MOVE_ROBBER: 40,
            AT_END_TURN: 0,
        }
        PRIORITY = {k: v for k, v in _priority_map.items() if k is not None}

        def action_priority(a: Action | None) -> int:
            if a is None:
                return -1
            return PRIORITY.get(getattr(a, 'action_type', None), 10)

        # Convert to list for multiple passes
        actions = list(playable_actions)
        if not actions:
            print('[foo_player] No playable actions available; returning None')
            return None

        # Introspection prints (one-time) to inspect ActionType and sample actions
        if not getattr(self, '_did_inspect', False):
            try:
                print("[foo_player] ActionType members:")
                try:
                    print([m.name for m in ActionType])
                except Exception:
                    print([n for n in dir(ActionType) if not n.startswith('_')])
                print("[foo_player] Sample playable_actions:")
                print([repr(a) for a in actions[:10]])
            except Exception as e:
                print(f"[foo_player] Introspect error: {e}")
            self._did_inspect = True

        # Build-like detection: avoid END_TURN when we can build/play dev
        build_like_types = {t for t in (AT_BUILD_CITY, AT_BUILD_SETTLEMENT, AT_BUILD_ROAD, AT_BUY_DEV, AT_PLAY_DEV) if t is not None}

        def _is_build_like_action(a):
            try:
                at = getattr(a, 'action_type', None)
                return at in (
                    AT_BUILD_CITY,
                    AT_BUILD_SETTLEMENT,
                    AT_BUILD_ROAD,
                    AT_BUY_DEV,
                    AT_PLAY_DEV,
                )
            except Exception:
                return False

        has_build_like = any(_is_build_like_action(a) for a in actions)

        # Robust END_TURN gating
        if AT_END_TURN is not None:
            filtered = [a for a in actions if not (getattr(a, 'action_type', None) == AT_END_TURN and has_build_like)]
        else:
            filtered = list(actions)

        if not filtered:
            filtered = actions

        # Cap number of candidate actions to evaluate
        if len(filtered) > max_actions:
            filtered = sorted(filtered, key=lambda x: action_priority(x), reverse=True)[:max_actions]

        # Build the value function (defensive to differing signatures)
        try:
            value_fn = make_value_fn("base_fn", DEFAULT_WEIGHTS)
        except TypeError:
            value_fn = make_value_fn("base_fn")

        # Helper: compute expected score from a list of (state, prob) or list of states
        def _expected_score_from_outcomes(outcomes, color):
            """
            outcomes may be:
              - List[(Game, float)] where float is probability
              - List[Game]
            Returns expected scalar score (float).
            """
            if not outcomes:
                return None
            # Try to detect (state, prob) pairs
            try:
                first = outcomes[0]
            except Exception:
                return None
            # (state, prob) pairs
            if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], (float, int)):
                total = 0.0
                for st, p in outcomes:
                    try:
                        total += p * value_fn(st, self.color)
                    except TypeError:
                        total += p * value_fn(st)
                return total
            # Otherwise assume list of states -> average
            total = 0.0
            count = 0
            for st in outcomes:
                try:
                    total += value_fn(st, self.color)
                except TypeError:
                    total += value_fn(st)
                count += 1
            return total / max(1, count)

        # Helper: try to get chance outcomes via adapters, with fallbacks
        def _get_chance_outcomes(gstate, action):
            """Return outcomes or None if not available.
            Preferred: chance_children_for_action(game, action) -> [(game_copy, prob), ...]
            Fallback: chance_children(game, [action]) -> {action: [(game_copy, prob), ...]}
            """
            try:
                if callable(chance_children_for_action):
                    return chance_children_for_action(gstate, action)
            except Exception:
                # Continue to other fallback
                pass
            try:
                if callable(chance_children):
                    batch = chance_children(gstate, [action])
                    return batch.get(action, None)
            except Exception:
                pass
            return None

        best_action = None
        best_score = -float('inf')
        evaluated = 0
        simulations = 0

        # Iterate candidates
        for action in filtered:
            if simulations >= max_simulations:
                print(f"[foo_player] reached max_simulations={max_simulations}; stopping evaluation")
                break

            start_time = time.time()
            try:
                gcopy = copy_game(game)
                # execute on the copy; use validate=False when available for speed
                try:
                    execute(gcopy, action, validate=False)
                except TypeError:
                    execute(gcopy, action)
            except Exception as e:
                print(f"[foo_player] simulation setup failed for action={action} error={e}")
                continue

            # Try to obtain chance outcomes for this executed action/state
            outcomes = None
            try:
                outcomes = _get_chance_outcomes(gcopy, action)
            except Exception:
                outcomes = None

            score = None
            # If outcomes returned, compute expected score exactly
            if outcomes:
                try:
                    score = _expected_score_from_outcomes(outcomes, self.color)
                except Exception as e:
                    print(f"[foo_player] expected score computation failed: {e}")
                    score = None

            # If we couldn't compute an expectation, and outcomes is None but action type is known to be chancey,
            # attempt a small Monte Carlo by re-executing the action on fresh copies.
            if score is None:
                is_chancey = False
                try:
                    if getattr(action, 'action_type', None) in {AT_ROLL, AT_BUY_DEV, AT_MOVE_ROBBER}:
                        is_chancey = True
                except Exception:
                    is_chancey = False

                if is_chancey:
                    # Monte Carlo fallback
                    total = 0.0
                    samples = samples_mc
                    for i in range(samples):
                        try:
                            sims_gcopy = copy_game(game)
                            try:
                                execute(sims_gcopy, action, validate=False)
                            except TypeError:
                                execute(sims_gcopy, action)
                            # If executing the action triggered a chance resolution inside execute,
                            # the state now reflects one sampled outcome; score it.
                            try:
                                total += value_fn(sims_gcopy, self.color)
                            except TypeError:
                                total += value_fn(sims_gcopy)
                        except Exception:
                            # If an execution failed mid-MC, ignore and reduce sample count
                            samples -= 1
                            continue
                        simulations += 1
                        if simulations >= max_simulations:
                            break
                    if samples > 0:
                        score = total / samples
                    else:
                        score = -float('inf')
                else:
                    # Deterministic scoring of the post-action copied state
                    try:
                        score = value_fn(gcopy, self.color)
                    except TypeError:
                        score = value_fn(gcopy)

            # --- NEW: Apply heuristic bonuses for city/settlement/road/knight ---
            base_score = score
            try:
                atype = getattr(action, 'action_type', None)

                # BUILD_CITY: immediate VP increase + nudge
                if atype == AT_BUILD_CITY:
                    try:
                        pstate = game.state.players.get(self.color, None)
                        if pstate is None:
                            raise ValueError("Player state not found")

                        # Count settlements
                        settlements = 0
                        try:
                            settlements = len([n for n in game.state.board.nodes()
                                              if game.state.board.node_owner(n) == self.color
                                              and game.state.board.node_is_settlement(n)])
                        except Exception:
                            settlements = 0

                        # Resource counts
                        wheat = 0
                        ore = 0
                        try:
                            res = getattr(pstate, 'resources', {})
                            wheat = int(res.get('wheat', res.get('WHEAT', res.get('grain', 0))))
                            ore = int(res.get('ore', res.get('ORE', 0)))
                        except Exception:
                            wheat = ore = 0

                        # Debug print
                        print(f"[foo_player-debug] Resources: {getattr(pstate,'resources',{})}, Settlements: {settlements}")

                        # Apply bonus
                        if settlements >= 2 and wheat >= 3 and ore >= 2:
                            score = (score or 0.0) + self.CITY_BONUS
                        else:
                            score = (score or 0.0) + (self.CITY_BONUS / 2.0)
                    except Exception as e:
                        print(f"[foo_player] BUILD_CITY heuristic error: {e}")
                        score = (score or 0.0) + (self.CITY_BONUS / 2.0)

                # BUILD_SETTLEMENT: favor high-pip or diversification
                elif atype == AT_BUILD_SETTLEMENT:
                    node = getattr(action, 'value', None)
                    if node is not None:
                        try:
                            sv = self.estimate_settlement_node_value(game, node, self.color)
                            score = (score or 0.0) + self.SETTLEMENT_BONUS * min(2.0, sv / 5.0)
                        except Exception:
                            pass

                # BUILD_ROAD: prefer extensions and roads that connect to settlement spots
                elif atype == AT_BUILD_ROAD:
                    edge = getattr(action, 'value', None)
                    if edge is not None:
                        try:
                            if self.road_extends_toward_open_spot(game, edge, self.color):
                                score = (score or 0.0) + self.ROAD_CONNECTS_SETTLEMENT_BONUS
                            else:
                                score = (score or 0.0) + self.ROAD_EXTENSION_BONUS
                        except Exception:
                            pass

                # PLAY_DEV_CARD (KNIGHT): prefer playing if it secures largest army
                elif atype == AT_PLAY_DEV and getattr(action, 'value', None) in ('KNIGHT', 'Knight', 'knight'):
                    try:
                        if self.will_gain_largest_army(game, self.color):
                            score = (score or 0.0) + self.KNIGHT_PLAY_BONUS
                    except Exception:
                        pass

            except Exception as e:
                print(f"[foo_player] heuristic error for action {action}: {e}")

            # Cap total bonus so heuristics don't overwhelm the learned value
            try:
                if (score or 0.0) - (base_score or 0.0) > self.MAX_BONUS_SCALE:
                    score = (base_score or 0.0) + self.MAX_BONUS_SCALE
            except Exception:
                pass

            # Optional logging when heuristic changed the score
            try:
                if score != base_score:
                    b = base_score if base_score is not None else 0.0
                    print(f"[foo_player] Heuristic applied: {action} base={b:.2f} bonus={(score - b):.2f} => {score:.2f}")
            except Exception:
                pass

            # Robber heuristic: tiny bonus for robber moves that target rich opponents
            try:
                if getattr(action, 'action_type', None) == AT_MOVE_ROBBER:
                    target = getattr(action, 'value', None)
                    if target is not None:
                        try:
                            freq = opponent_freqdeck(game, target)
                            disruption = 0
                            try:
                                # freq may be a dict-like
                                if hasattr(freq, 'values'):
                                    disruption = sum(freq.values())
                                else:
                                    disruption = sum(freq)
                            except Exception:
                                disruption = 0
                            score = (score or 0.0) + 0.05 * disruption
                        except Exception:
                            pass
            except Exception:
                pass

            simulations += 1
            evaluated += 1

            # Tie-breaker: prefer higher score then higher priority
            pri = action_priority(action)
            if (score is not None and score > best_score) or (score == best_score and pri > action_priority(best_action)):
                best_score = score
                best_action = action

            # Time guard
            if (time.time() - start_time) > time_budget_per_action:
                print(f"[foo_player] time budget exceeded for action {action}; breaking early for this action")
                # don't abort entire loop; continue to next action
                continue

        if best_action is None:
            print('[foo_player] No simulated action succeeded; falling back to first playable action')
            return actions[0]

        print(f"[foo_player] Chosen action: {best_action}, expected_score: {best_score:.3f}, evaluated: {evaluated}/{len(filtered)}, simulations: {simulations}")
        return best_action
