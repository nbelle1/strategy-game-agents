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

    Implemented strategy (1-ply lookahead + chance-awareness + heuristics):
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
    CITY_BONUS = 5.0  # legacy param (kept for compatibility)
    # Aggressive defaults introduced by Strategizer
    CITY_BONUS_AGGRESSIVE = 6.0
    CITY_BONUS_FALLBACK = 3.0
    CITY_FORCE_IF_AFFORDABLE = False

    SETTLEMENT_BONUS = 1.0
    ROAD_EXTENSION_BONUS = 1.0
    ROAD_CONNECTS_SETTLEMENT_BONUS = 1.5
    KNIGHT_PLAY_BONUS = 2.5
    MAX_BONUS_SCALE = 10.0

    # Road scoring tuning
    ROAD_REACH_BONUS = 2.0
    ROAD_PIP_SCALE = 0.6
    ROAD_MAX_DISTANCE = 4

    # Additional road/settlement params from Strategizer
    ROAD_CHAIN_BONUS = 1.5
    ROAD_PIP_NORMALIZER = 8.0
    LONGEST_ROAD_LOCAL_BONUS = 0.8

    SETTLEMENT_HIGH_PIP_BONUS = 3.0
    SETTLEMENT_PIP_THRESHOLD = 4.0
    SETTLEMENT_PIP_NORMALIZER = 8.0
    SETTLEMENT_ENABLES_CITY_BONUS = 1.5

    # END_TURN strict gating
    END_TURN_STRICT = True
    END_TURN_PENALTY = 5.0

    # New strategist tuning params
    ROAD_CHAIN_WEIGHT = 2.0
    PIP_WEIGHT = 1.0
    DIST_PENALTY = 0.7
    OPP_BLOCK_BONUS = 0.5
    ROAD_PIP_NORMALIZER = 6.0

    SETTLEMENT_CONNECTIVITY_WEIGHT = 1.5
    SETTLEMENT_BUILD_THRESHOLD = 3.5
    SETTLEMENT_PIP_NORMALIZER = 6.0
    SETTLEMENT_HIGH_PIP_BONUS = 2.0

    DEV_CARD_BONUS = 1.0

    # New settlement-focused tunables from Strategizer
    SETTLEMENT_URGENCY_WEIGHT = 2.5
    SETTLEMENT_URGENCY_TURN_WINDOW = 12
    SETTLEMENT_NEAR_AFFORD_TURNS = 2
    NEAR_AFFORD_BOOST = 1.5
    DEV_CARD_BIAS_REDUCTION_FACTOR = 0.25

    def __init__(self, name: str | None = None):
        super().__init__(Color.BLUE, name)
        # one-time introspection flag
        self._did_inspect = False
        self._did_inspect_resources_edges = False
        # counters for debug
        self._city_bonus_count = 0
        self._game_road_count = 0
        self._game_settle_count = 0
        # diagnostics
        self._edge_resolve_failures = 0
        self._resource_shape_failures = 0
        self._did_log_shapes = False
        self._logged_list_resources = False
        self._logged_unknown_res = False

    # --- Adapter fixes: unified resource normalizer ---
    def _normalize_resources(self, pstate):
        """
        Return canonical dict {'wheat','ore','wood','brick','sheep'}.
        Handles all observed shapes: dict, list-of-pairs, list-of-ints, objects.
        Logs first occurrence of list-shaped mapping.
        """
        norm = {'wheat': 0, 'ore': 0, 'wood': 0, 'brick': 0, 'sheep': 0}
        try:
            raw = getattr(pstate, 'resources', None)
            if raw is None:
                return norm

            # Case 1: dict-like
            if hasattr(raw, 'items'):
                for k, v in raw.items():
                    k2 = str(k).upper()
                    try:
                        val = int(v)
                    except Exception:
                        continue
                    if k2 in ('WHEAT', 'GRAIN'):
                        norm['wheat'] = val
                    elif k2 == 'ORE':
                        norm['ore'] = val
                    elif k2 == 'WOOD':
                        norm['wood'] = val
                    elif k2 in ('BRICK', 'CLAY'):
                        norm['brick'] = val
                    elif k2 in ('SHEEP', 'WOOL'):
                        norm['sheep'] = val
                return norm

            # Case 2: list/tuple of (key, val) pairs
            if isinstance(raw, (list, tuple)) and len(raw) and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
                for k, v in raw:
                    k2 = str(k).upper()
                    try:
                        val = int(v)
                    except Exception:
                        continue
                    if k2 in ('WHEAT', 'GRAIN'):
                        norm['wheat'] = val
                    elif k2 == 'ORE':
                        norm['ore'] = val
                    elif k2 == 'WOOD':
                        norm['wood'] = val
                    elif k2 in ('BRICK', 'CLAY'):
                        norm['brick'] = val
                    elif k2 in ('SHEEP', 'WOOL'):
                        norm['sheep'] = val
                return norm

            # Case 3: list/tuple of 5 ints (empirical: [wood, brick, sheep, wheat, ore])
            if isinstance(raw, (list, tuple)) and len(raw) == 5 and all(isinstance(x, int) for x in raw):
                norm['wood'] = int(raw[0])
                norm['brick'] = int(raw[1])
                norm['sheep'] = int(raw[2])
                norm['wheat'] = int(raw[3])
                norm['ore'] = int(raw[4])
                if not getattr(self, '_logged_list_resources', False):
                    print(f"[foo_player] Detected list-resource shape -> mapped to {norm}")
                    self._logged_list_resources = True
                return norm

            # Case 4: object with attributes
            for k in ('wheat', 'ore', 'wood', 'brick', 'sheep'):
                try:
                    norm[k] = int(getattr(raw, k, 0))
                except Exception:
                    pass
            # If nothing matched, log once
            if not getattr(self, '_logged_unknown_res', False):
                try:
                    print(f"[foo_player] Unknown pstate.resources shape: type={type(raw)} repr={repr(raw)[:200]}")
                except Exception:
                    print(f"[foo_player] Unknown pstate.resources shape: type={type(raw)}")
                self._logged_unknown_res = True
            return norm
        except Exception as e:
            print(f"[foo_player] _normalize_resources error: {e}")
            return norm

    # small safe wrapper for settlement value estimation
    def _safe_estimate_settlement_node_value(self, game, node_id):
        try:
            return self.estimate_settlement_node_value(game, node_id, self.color)
        except Exception:
            return 0.0

    # --- Replace affordability with strict normalizer usage ---
    def _can_afford_build(self, game, color, action_type):
        """Deterministic affordability check using _normalize_resources."""
        try:
            pstate = game.state.players.get(color)
            if not pstate:
                return False
            res = self._normalize_resources(pstate)
            if action_type == getattr(ActionType, 'BUILD_CITY', None):
                return res['wheat'] >= 3 and res['ore'] >= 2
            if action_type == getattr(ActionType, 'BUILD_SETTLEMENT', None):
                return res['wood'] >= 1 and res['brick'] >= 1 and res['wheat'] >= 1 and res['sheep'] >= 1
            if action_type == getattr(ActionType, 'BUILD_ROAD', None):
                return res['wood'] >= 1 and res['brick'] >= 1
            if action_type == getattr(ActionType, 'BUY_DEVELOPMENT_CARD', None):
                return res['wheat'] >= 1 and res['sheep'] >= 1 and res['ore'] >= 1
        except Exception as e:
            print(f"[foo_player] _can_afford_build error: {e}")
        return False

    def estimate_settlement_node_value(self, game, node_id, color):
        """Estimate pip-sum for a node; defensive to API differences."""
        total = 0.0
        try:
            board = game.state.board
            tiles = None
            try:
                tiles = board.node_tiles(node_id)
            except Exception:
                try:
                    # board.tiles may be dict-like keyed by id
                    tiles_attr = getattr(board, 'tiles', None)
                    if isinstance(tiles_attr, dict):
                        tiles = [tiles_attr[t] for t in tiles_attr if node_id in getattr(tiles_attr[t], 'nodes', [])]
                    else:
                        # tiles might be a list of objects
                        tiles = [t for t in getattr(board, 'tiles', []) if node_id in getattr(t, 'nodes', [])]
                except Exception:
                    print(f"[foo_player] Failed to resolve node_tiles for node {node_id}")
                    return 0.0
            for tile in tiles:
                pip = None
                try:
                    pip = getattr(tile, 'pip_probability', None)
                except Exception:
                    pip = None
                if pip is None:
                    try:
                        pip = getattr(tile, 'pip', None)
                    except Exception:
                        pip = 1.0
                try:
                    total += float(pip)
                except Exception:
                    total += 0.0
        except Exception as e:
            print(f"[foo_player] estimate_settlement_node_value error: {e}")
        return total

    def _edge_to_nearest_open_info(self, game, start_edge):
        """Return (distance, pip_sum) for nearest open node reachable from start_edge."""
        try:
            board = game.state.board
            start_nodes = None

            # Try all possible ways to resolve edge -> nodes
            try:
                start_nodes = board.edge_nodes(start_edge)
            except Exception:
                try:
                    start_nodes = board.edges[start_edge].nodes
                except Exception:
                    try:
                        if isinstance(start_edge, (tuple, list)) and len(start_edge) == 2:
                            start_nodes = list(start_edge)
                        elif hasattr(start_edge, 'nodes'):
                            start_nodes = list(getattr(start_edge, 'nodes', []))
                        elif hasattr(board, 'edges') and start_edge in board.edges:
                            edge_obj = board.edges[start_edge]
                            start_nodes = getattr(edge_obj, 'nodes', None) or getattr(edge_obj, 'nodes', [])
                    except Exception:
                        pass

            if not start_nodes:
                print(f"[foo_player] No start_nodes found for edge {start_edge}")
                return (None, 0.0)

            from collections import deque
            visited_edges = set()
            visited_nodes = set()
            q = deque([(n, 0) for n in start_nodes])
            visited_nodes.update(start_nodes)

            best_dist = None
            best_pip = 0.0

            while q:
                node, dist = q.popleft()
                if dist > getattr(self, 'ROAD_MAX_DISTANCE', 6):
                    continue
                owner = None
                try:
                    owner = board.node_owner(node)
                except Exception:
                    try:
                        owner = board.nodes[node].owner
                    except Exception:
                        owner = getattr(board.nodes.get(node, {}), 'owner', None)
                if owner is None:
                    pip_sum = 0.0
                    try:
                        tiles = board.node_tiles(node)
                        for t in tiles:
                            pip = getattr(t, 'pip_probability', getattr(t, 'pip', 1.0))
                            pip_sum += float(pip)
                    except Exception:
                        pip_sum = 0.0
                    if best_dist is None or dist < best_dist or (dist == best_dist and pip_sum > best_pip):
                        best_dist = dist
                        best_pip = pip_sum
                try:
                    incident_edges = None
                    try:
                        incident_edges = board.node_edges(node)
                    except Exception:
                        try:
                            # board.edges may be dict-like keyed by id
                            incident_edges = [e for e in getattr(board, 'edges', {}) if node in getattr(board.edges[e], 'nodes', [])]
                        except Exception:
                            incident_edges = []
                    for e in incident_edges:
                        if e in visited_edges:
                            continue
                        visited_edges.add(e)
                        try:
                            neigh_nodes = None
                            try:
                                neigh_nodes = board.edge_nodes(e)
                            except Exception:
                                try:
                                    neigh_nodes = board.edges[e].nodes
                                except Exception:
                                    neigh_nodes = None
                            for nn in neigh_nodes:
                                if nn not in visited_nodes:
                                    visited_nodes.add(nn)
                                    q.append((nn, dist + 1))
                        except Exception:
                            continue
                except Exception:
                    pass
            return (best_dist, best_pip)
        except Exception as e:
            print(f"[foo_player] _edge_to_nearest_open_info error: {e}")
            return (None, 0.0)

    def road_extends_toward_open_spot(self, game, edge_id, color):
        """
        Return True if edge touches any node without owner (open spot).
        Uses defensive API calls and logs for debugging.
        """
        try:
            board = game.state.board
            nodes = None
            try:
                nodes = board.edge_nodes(edge_id)
            except Exception:
                try:
                    nodes = board.edges[edge_id].nodes
                except Exception:
                    print(f"[foo_player] Failed to resolve edge_nodes for edge {edge_id}")
                    return False
            owners = []
            for n in nodes:
                owner = None
                try:
                    owner = board.node_owner(n)
                except Exception:
                    try:
                        owner = board.nodes[n].owner
                    except Exception:
                        try:
                            owner = getattr(board.nodes.get(n, {}), 'owner', None)
                        except Exception:
                            owner = None
                owners.append(owner)
                if owner is None:
                    # debug print owners and nodes
                    try:
                        print(f"[foo_player-debug] Edge {edge_id} nodes: {nodes}, Owners: {owners}")
                    except Exception:
                        pass
                    return True
        except Exception as e:
            print(f"[foo_player] road_extends_toward_open_spot error: {e}")
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

    # --- New strategist helpers ---
    def _road_chain_score(self, game, edge_id):
        """Score road edges by chain extension, reach, and opponent blocking."""
        try:
            board = game.state.board
            score = 0.0
            nodes = []
            # 1. Chain extension: Does this edge connect to my existing roads?
            try:
                nodes = board.edge_nodes(edge_id)
            except Exception:
                try:
                    eobj = getattr(getattr(board, 'edges', {}), edge_id, None)
                    nodes = getattr(eobj, 'nodes', None) or []
                except Exception:
                    nodes = []
            for n in nodes or []:
                try:
                    incident_edges = board.node_edges(n)
                except Exception:
                    incident_edges = [e for e in getattr(board, 'edges', {}) if n in getattr(getattr(board.edges, e, {}), 'nodes', [])]
                for e in incident_edges:
                    try:
                        owner = board.edge_owner(e)
                    except Exception:
                        owner = getattr(getattr(board.edges, e, {}), 'owner', None)
                    if owner == self.color:
                        score += getattr(self, 'ROAD_CHAIN_WEIGHT', 2.0)
                        break

            # 2. Distance to high-pip open node
            try:
                dist, pip_sum = self._edge_to_nearest_open_info(game, edge_id)
                if dist is not None:
                    score += getattr(self, 'PIP_WEIGHT', 1.0) * (float(pip_sum) / getattr(self, 'ROAD_PIP_NORMALIZER', 6.0))
                    score -= getattr(self, 'DIST_PENALTY', 0.7) * float(dist)
            except Exception:
                pass

            # 3. Opponent blocking: Small bonus if adjacent node is owned by opponent
            try:
                for n in nodes or []:
                    try:
                        owner = board.node_owner(n)
                    except Exception:
                        owner = getattr(getattr(board.nodes, n, {}), 'owner', None)
                    if owner and owner != self.color:
                        score += getattr(self, 'OPP_BLOCK_BONUS', 0.5)
            except Exception:
                pass

            return float(score)
        except Exception as e:
            print(f"[foo_player] _road_chain_score error: {e}")
            return 0.0

    def _settlement_node_score(self, game, node_id):
        """Score settlement nodes by pip-sum and connectivity."""
        try:
            pip_sum = float(self.estimate_settlement_node_value(game, node_id, self.color) or 0.0)
            connectivity = 0.0
            try:
                board = game.state.board
                try:
                    edges = board.node_edges(node_id)
                except Exception:
                    edges = [e for e in getattr(board, 'edges', {}) if node_id in getattr(getattr(board.edges, e, {}), 'nodes', [])]
                for e in edges:
                    try:
                        owner = board.edge_owner(e)
                    except Exception:
                        owner = getattr(getattr(board.edges, e, {}), 'owner', None)
                    if owner == self.color:
                        connectivity += 1.0
            except Exception:
                pass
            score = pip_sum + getattr(self, 'SETTLEMENT_CONNECTIVITY_WEIGHT', 1.5) * connectivity
            return float(score)
        except Exception as e:
            print(f"[foo_player] _settlement_node_score error: {e}")
            return 0.0

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
                # Additional diagnostics requested by Strategizer
                try:
                    print("[foo_player] Board API:", dir(game.state.board))
                except Exception:
                    print("[foo_player] Board API: <unavailable>")
                try:
                    pstate = game.state.players.get(self.color)
                    print("[foo_player] Player state API:", dir(pstate))
                except Exception:
                    print("[foo_player] Player state API: <unavailable>")
            except Exception as e:
                print(f"[foo_player] Introspect error: {e}")
            self._did_inspect = True

        # Additional one-time resource/edge diagnostics
        if not getattr(self, '_did_inspect_resources_edges', False):
            try:
                pstate = game.state.players.get(self.color)
                print(f"[foo_player] pstate.resources type (me): {type(getattr(pstate,'resources', None))}")
                print(f"[foo_player] pstate.resources repr (me): {repr(getattr(pstate,'resources', None))}")

                # sample BUILD_ROAD playable actions
                try:
                    print("[foo_player] Sample BUILD_ROAD actions:", [(getattr(a,'value',None), type(getattr(a,'value',None))) for a in actions if getattr(a,'action_type',None)==AT_BUILD_ROAD][:10])
                except Exception as e:
                    print(f"[foo_player] Sample BUILD_ROAD actions error: {e}")

                # board.edges introspection
                try:
                    edges_attr = getattr(game.state.board, 'edges', None)
                    print(f"[foo_player] board.edges type: {type(edges_attr)}")
                    try:
                        print("[foo_player] board.edges sample keys:", list(edges_attr.keys())[:10])
                    except Exception as e:
                        print(f"[foo_player] board.edges sample keys error: {e}")
                except Exception as e:
                    print(f"[foo_player] board introspect error: {e}")

            except Exception as e:
                print(f"[foo_player] resources/edges introspect error: {e}")
            self._did_inspect_resources_edges = True

        # NEW: one-time shape diagnostics per Strategizer
        if not getattr(self, '_did_log_shapes', False):
            try:
                pstate = game.state.players.get(self.color)
                print(f"[foo_player] DEBUG resources type={type(getattr(pstate,'resources',None))} repr={repr(getattr(pstate,'resources',None))}")
                for a in actions:
                    if getattr(a, 'action_type', None) == AT_BUILD_ROAD:
                        edge = getattr(a, 'value', None)
                        print(f"[foo_player] DEBUG BUILD_ROAD edge type={type(edge)} repr={repr(edge)}")
                        break
            except Exception as e:
                print(f"[foo_player] shape diag error: {e}")
            self._did_log_shapes = True

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

        # Secondary: compute near-affordable settlements and tag build actions
        near_affordable_settlement = False
        try:
            pstate = game.state.players.get(self.color)
            res = self._normalize_resources(pstate)
            missing = max(0, 1 - res['wood']) + max(0, 1 - res['brick']) + max(0, 1 - res['wheat']) + max(0, 1 - res['sheep'])
            if missing <= 2:
                for a in actions:
                    if getattr(a, 'action_type', None) == AT_BUILD_SETTLEMENT:
                        node = getattr(a, 'value', None)
                        if node is not None:
                            try:
                                ns = self._settlement_node_score(game, node)
                                if ns >= (self.SETTLEMENT_BUILD_THRESHOLD - 1.0):
                                    near_affordable_settlement = True
                                    break
                            except Exception:
                                continue
        except Exception as e:
            print(f"[foo_player] near-afford settlement check error: {e}")

        if near_affordable_settlement:
            for a in actions:
                if getattr(a, 'action_type', None) == AT_BUILD_SETTLEMENT:
                    setattr(a, '_near_afford_boost', True)

        # Robust END_TURN gating: use resource-based affordability instead of execute() try
        affordable_build_exists = False
        try:
            if self.END_TURN_STRICT:
                # defensive loop-based check so individual _can_afford_build errors don't abort gating
                for a in actions:
                    try:
                        atype = getattr(a, 'action_type', None)
                        if atype in (AT_BUILD_SETTLEMENT, AT_BUILD_ROAD, AT_BUILD_CITY):
                            if self._can_afford_build(game, self.color, atype):
                                affordable_build_exists = True
                                break
                    except Exception as e:
                        print(f"[foo_player] afford-check error for action {a}: {e}")
                        continue
            else:
                affordable_build_exists = has_build_like
        except Exception:
            affordable_build_exists = has_build_like

        if AT_END_TURN is not None:
            filtered = [a for a in actions if not (getattr(a, 'action_type', None) == AT_END_TURN and affordable_build_exists)]
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
                            # Defensive: board.nodes may be callable or iterable
                            try:
                                nodes_iter = game.state.board.nodes()
                            except Exception:
                                nodes_iter = getattr(game.state.board, 'nodes', [])
                            # This lambda intentionally left as old placeholder; try robust count below
                            settlements = 0
                            for n in nodes_iter:
                                try:
                                    owner = None
                                    try:
                                        owner = game.state.board.node_owner(n)
                                    except Exception:
                                        try:
                                            owner = getattr(game.state.board.nodes[n], 'owner', None)
                                        except Exception:
                                            owner = None
                                    if owner == self.color:
                                        # Detect if it's a settlement vs city where possible
                                        is_settlement = True
                                        try:
                                            is_settlement = game.state.board.node_is_settlement(n)
                                        except Exception:
                                            # fallback: assume it's a settlement if owned
                                            is_settlement = True
                                        if is_settlement:
                                            settlements += 1
                                except Exception:
                                    continue
                        except Exception:
                            # Fallback: try player state count
                            settlements = getattr(pstate, 'settlement_count', 0)

                        # Resource counts using normalized helper
                        # Use the robust mapping helper to handle list/dict shapes
                        try:
                            res = self._normalize_resources(pstate)
                        except Exception:
                            res = {'wheat': 0, 'ore': 0, 'wood': 0, 'brick': 0, 'sheep': 0}
                        wheat = res['wheat']
                        ore = res['ore']

                        # Debug print
                        print(f"[foo_player-debug] Resources: {getattr(pstate,'resources',{})}, Settlements: {settlements}")

                        # Apply bonus: aggressive if fully affordable, fallback if partial
                        full_afford = (wheat >= 3 and ore >= 2)
                        partial_afford = (wheat >= 1 and ore >= 1)

                        if self.CITY_FORCE_IF_AFFORDABLE and full_afford:
                            print("[foo_player] FORCE BUILD_CITY (debug)")
                            # Immediate forced return - play this action
                            print(f"[foo_player] Forcing action: {action}")
                            return action

                        if full_afford:
                            score = (score or 0.0) + self.CITY_BONUS_AGGRESSIVE
                            self._city_bonus_count = getattr(self, '_city_bonus_count', 0) + 1
                        elif partial_afford and settlements >= 1:
                            score = (score or 0.0) + self.CITY_BONUS_FALLBACK
                        else:
                            # small nudge to consider city in long-term
                            score = (score or 0.0) + (self.CITY_BONUS_FALLBACK / 2.0)
                    except Exception as e:
                        print(f"[foo_player] BUILD_CITY heuristic error: {e}")
                        score = (score or 0.0) + (self.CITY_BONUS_FALLBACK / 2.0)

                # BUILD_SETTLEMENT: favor high-pip or diversification
                elif atype == AT_BUILD_SETTLEMENT:
                    node = getattr(action, 'value', None)
                    if node is not None:
                        try:
                            # Use the new settlement node scoring
                            node_score = self._settlement_node_score(game, node)
                            if node_score >= getattr(self, 'SETTLEMENT_BUILD_THRESHOLD', 3.5):
                                add = getattr(self, 'SETTLEMENT_HIGH_PIP_BONUS', 2.0) * (node_score / getattr(self, 'SETTLEMENT_PIP_NORMALIZER', 6.0))
                                score = (score or 0.0) + add
                                print(f"[foo_player] Settlement node score for {node}: {node_score:.2f} added {add:.2f}")
                            else:
                                # small default settlement nudge
                                score = (score or 0.0) + self.SETTLEMENT_BONUS
                        except Exception as e:
                            print(f"[foo_player] settlement scoring error for {node}: {e}")

                    # Primary Strategizer patch: settlement urgency additive
                    try:
                        if node is not None:
                            urgency = self._settlement_urgency(game, node)
                            add = getattr(self, 'SETTLEMENT_URGENCY_WEIGHT', 2.5) * urgency
                            score = (score or 0.0) + add
                            print(f"[foo_player] Settlement urgency for {node}: {urgency:.3f} added={add:.2f}")
                    except Exception as e:
                        print(f"[foo_player] settlement urgency error for {node}: {e}")

                    # Secondary: near-afford boost if tagged earlier
                    try:
                        if getattr(action, '_near_afford_boost', False):
                            boost = getattr(self, 'NEAR_AFFORD_BOOST', 1.5)
                            score = (score or 0.0) + boost
                            print(f"[foo_player] Near-afford boost applied for settlement {node}: +{boost:.2f}")
                    except Exception as e:
                        print(f"[foo_player] near-afford boost error: {e}")

                # BUILD_ROAD: prefer extensions and roads that connect to settlement spots
                elif atype == AT_BUILD_ROAD:
                    edge = getattr(action, 'value', None)
                    if edge is not None:
                        try:
                            # Use road chain scoring to favor chain-extension and reach
                            chain = 0.0
                            try:
                                chain = self._road_chain_score(game, edge)
                                score = (score or 0.0) + chain
                                print(f"[foo_player] Road chain score for {edge}: {chain:.2f}")
                            except Exception:
                                # fallback to reach-based logic
                                dist, pip_sum = self._edge_to_nearest_open_info(game, edge)
                                if dist is not None:
                                    norm = max(0.0, (self.ROAD_MAX_DISTANCE - dist) / float(self.ROAD_MAX_DISTANCE))
                                    pip_factor = min(1.0, pip_sum / 8.0)
                                    bonus = self.ROAD_REACH_BONUS * norm + self.ROAD_PIP_SCALE * pip_factor
                                else:
                                    bonus = self.ROAD_EXTENSION_BONUS
                                score = (score or 0.0) + bonus
                                print(f"[foo_player-debug] BUILD_ROAD fallback edge={edge} dist={dist} pip_sum={pip_sum} bonus={bonus:.2f}")
                        except Exception as e:
                            print(f"[foo_player] BUILD_ROAD heuristic error: {e}")

                # PLAY_DEV_CARD (KNIGHT): prefer playing if it secures largest army
                elif atype == AT_PLAY_DEV and getattr(action, 'value', None) in ('KNIGHT', 'Knight', 'knight'):
                    try:
                        if self.will_gain_largest_army(game, self.color):
                            score = (score or 0.0) + self.KNIGHT_PLAY_BONUS
                    except Exception:
                        pass

                # BUY_DEVCARD: small bias when resources idle and no builds affordable
                elif atype == AT_BUY_DEV:
                    try:
                        pstate = game.state.players.get(self.color)
                        res = self._normalize_resources(pstate)
                        enough = res.get('wheat', 0) >= 1 and res.get('sheep', 0) >= 1 and res.get('ore', 0) >= 1
                        # adjust dev-card bias per Strategizer: reduce when builds possible
                        can_build_any = any(self._can_afford_build(game, self.color, t) for t in (AT_BUILD_SETTLEMENT, AT_BUILD_ROAD, AT_BUILD_CITY))
                        if enough:
                            if can_build_any:
                                dev_bonus = getattr(self, 'DEV_CARD_BONUS', 1.0) * getattr(self, 'DEV_CARD_BIAS_REDUCTION_FACTOR', 0.25)
                                score = (score or 0.0) + dev_bonus
                                print(f"[foo_player] Dev card bonus reduced to {dev_bonus:.2f} due to settlement possibility")
                            else:
                                score = (score or 0.0) + getattr(self, 'DEV_CARD_BONUS', 1.0)
                                print(f"[foo_player] Dev card bonus applied: {getattr(self,'DEV_CARD_BONUS',1.0):.2f}")
                    except Exception as e:
                        print(f"[foo_player] dev-card bias error: {e}")

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

        # Increment per-game counters for chosen build actions
        try:
            at_chosen = getattr(best_action, 'action_type', None)
            if at_chosen == AT_BUILD_ROAD:
                self._game_road_count = getattr(self, '_game_road_count', 0) + 1
            if at_chosen == AT_BUILD_SETTLEMENT:
                self._game_settle_count = getattr(self, '_game_settle_count', 0) + 1
        except Exception:
            pass

        # Debug: report city-bonus usage during the run
        try:
            if hasattr(self, '_city_bonus_count'):
                print(f"[foo_player] City bonuses applied this game: {self._city_bonus_count}")
        except Exception:
            pass

        # Print per-game build counters
        try:
            print(f"[foo_player] Game build counters: roads_built={getattr(self,'_game_road_count',0)}, settlements_built={getattr(self,'_game_settle_count',0)}")
        except Exception:
            pass

        print(f"[foo_player] Chosen action: {best_action}, expected_score: {best_score:.3f}, evaluated: {evaluated}/{len(filtered)}, simulations: {simulations}")
        return best_action
