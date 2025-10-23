import math
import random
import statistics
import sys
import traceback
from typing import Any, Optional, Iterable, List, Dict, Tuple

# MUST import only from the adapters surface provided
from .adapters import Game, Player, Color

# Optional helper functions exported by adapters (may not exist in this environment)
try:
    from .adapters import copy_game, get_playable_actions, do_action
    HAVE_ADAPTER_HELPERS = True
except Exception:
    HAVE_ADAPTER_HELPERS = False

# Optionally import a value function builder if adapters provide one
try:
    from .adapters import make_value_fn
    HAVE_VALUE_FN = True
except Exception:
    HAVE_VALUE_FN = False

# Hyperparameters
SIMULATIONS = 100          # MCTS iterations per decision
K_ROLLOUTS = 30            # rollouts per action fallback / leaf eval
MAX_ROLLOUT_DEPTH = 10     # depth limit for random rollouts
MAX_ACTIONS_TO_EVALUATE = 12
UCT_C = 1.414
DEBUG = True

# One-time diagnostic guard
_DUMPED_PLAYER_SCHEMA = False


# ----------------- Import hardening helper (lazy adapters) -----------------
def _resolve_adapters():
    """Attempt to import the adapters module lazily.

    This avoids binding to adapters at module-import time and allows the
    harness to fail more gracefully. Call inside methods (not at top-level)
    so foo_player can be imported even if adapters import is problematic.
    """
    try:
        import adapters
        return adapters
    except Exception as e:
        if DEBUG:
            print('WARNING: adapters not importable:', e, file=sys.stderr)
        return None


class FooPlayer(Player):
    """Player that uses adapter-first MCTS with rollouts fallback.

    Implementation notes:
    - Uses adapters when available (via _resolve_adapters()) for copy/get/do ops.
    - MCTS is implemented as an inner class to reuse FooPlayer helpers.
    - Falls back to 1-ply Monte Carlo rollouts if adapters/value-fn or MCTS fail.
    """

    def __init__(self, name: Optional[str] = None):
        try:
            super().__init__(Color.BLUE, name)
        except Exception:
            try:
                super().__init__()
            except Exception:
                # Best-effort: continue without base initialization
                pass
        random.seed(None)

    # ----------------- Adapter wrappers / defensive helpers -----------------
    def _copy_game(self, game: Game) -> Optional[Game]:
        """Copy a game state using adapters if available, otherwise try common APIs."""
        adapters = _resolve_adapters()
        if adapters is not None and hasattr(adapters, 'copy_game'):
            try:
                return adapters.copy_game(game)
            except Exception:
                if DEBUG:
                    print('FooPlayer._copy_game: adapters.copy_game failed; falling back', file=sys.stderr)
        # Try common game APIs
        try:
            if hasattr(game, 'copy') and callable(getattr(game, 'copy')):
                return game.copy()
        except Exception:
            pass
        try:
            clone = getattr(game, 'clone', None)
            if callable(clone):
                return clone()
        except Exception:
            pass
        try:
            import copy as _cpy
            return _cpy.deepcopy(game)
        except Exception:
            if DEBUG:
                print('FooPlayer._copy_game: deepcopy failed', file=sys.stderr)
            return None

    def _get_playable_actions(self, game: Game) -> List[Any]:
        """Get playable actions using adapter helper if possible, else probe game.

        Returns a list (possibly empty).
        """
        adapters = _resolve_adapters()
        if adapters is not None and hasattr(adapters, 'get_playable_actions'):
            try:
                acts = adapters.get_playable_actions(game)
                return list(acts) if acts is not None else []
            except Exception:
                if DEBUG:
                    print('FooPlayer._get_playable_actions: adapters.get_playable_actions failed; falling back', file=sys.stderr)
        # Probe common names on game and game.state
        try_names = [
            'get_playable_actions',
            'playable_actions',
            'legal_actions',
            'get_legal_actions',
        ]
        for name in try_names:
            try:
                attr = getattr(game, name, None)
                if attr is None:
                    continue
                res = attr() if callable(attr) else attr
                if res is None:
                    continue
                try:
                    return list(res)
                except Exception:
                    return [res]
            except Exception:
                continue
        # Try state helpers
        try:
            st = getattr(game, 'state', None)
            if st is not None:
                for name in try_names:
                    try:
                        attr = getattr(st, name, None)
                        if attr is None:
                            continue
                        res = attr() if callable(attr) else attr
                        if res is None:
                            continue
                        try:
                            return list(res)
                        except Exception:
                            return [res]
                    except Exception:
                        continue
        except Exception:
            pass
        return []

    def _do_action(self, game: Game, action: Any) -> bool:
        """Apply an action using adapter do_action if available, otherwise try common APIs."""
        adapters = _resolve_adapters()
        if adapters is not None and hasattr(adapters, 'do_action'):
            try:
                adapters.do_action(game, action)
                return True
            except Exception:
                if DEBUG:
                    print('FooPlayer._do_action: adapters.do_action failed; falling back', file=sys.stderr)
        try:
            if hasattr(game, 'execute') and callable(getattr(game, 'execute')):
                game.execute(action)
                return True
        except Exception:
            pass
        try:
            if hasattr(game, 'apply') and callable(getattr(game, 'apply')):
                game.apply(action)
                return True
        except Exception:
            pass
        try:
            if hasattr(game, 'do_action') and callable(getattr(game, 'do_action')):
                game.do_action(action)
                return True
        except Exception:
            pass
        return False

    # ----------------- Robust state evaluator & diagnostic dump -----------------
    def _evaluate_state(self, game: Game) -> float:
        """Extract player metrics and compute a composite float score.

        The method searches for the current player's object in the game state
        using a sequence of defensive attempts, extracts numeric metrics in an
        ordered way, and computes a composite score.
        """
        global _DUMPED_PLAYER_SCHEMA

        # Default metrics
        vp = settlements = cities = roads = dev_vp = army = 0

        # Attempt to find players container in a robust way
        players = None
        try:
            st = getattr(game, 'state', None)
            if st is not None:
                players = getattr(st, 'players', None)
        except Exception:
            players = None
        if players is None:
            players = getattr(game, 'players', None)
        if players is None:
            players = getattr(game, 'player_state', None)

        # Helper to coerce numeric values safely
        def _coerce_count(x: Any) -> Optional[int]:
            try:
                if x is None:
                    return None
                if isinstance(x, (list, tuple, set)):
                    return len(x)
                if callable(x):
                    x = x()
                return int(x)
            except Exception:
                return None

        # Find our player object using ordered attempts
        my_player_obj = None
        try:
            # If mapping, try several key types
            if isinstance(players, dict):
                candidate_keys = []
                try:
                    candidate_keys.append(getattr(self, 'color', None))
                except Exception:
                    pass
                try:
                    candidate_keys.append(str(getattr(self, 'color', None)))
                except Exception:
                    pass
                try:
                    candidate_keys.append(getattr(getattr(self, 'color', None), 'name', None))
                except Exception:
                    pass
                try:
                    candidate_keys.append(int(getattr(self, 'color', None)))
                except Exception:
                    pass
                for key in candidate_keys:
                    try:
                        if key in players:
                            my_player_obj = players[key]
                            break
                    except Exception:
                        continue
                if my_player_obj is None:
                    for p in players.values():
                        try:
                            if hasattr(p, 'color') and getattr(p, 'color', None) == getattr(self, 'color', None):
                                my_player_obj = p
                                break
                            if isinstance(p, dict) and p.get('color', None) == getattr(self, 'color', None):
                                my_player_obj = p
                                break
                            if hasattr(p, 'name') and getattr(p, 'name', None) == getattr(self, 'name', None):
                                my_player_obj = p
                                break
                        except Exception:
                            continue
            elif isinstance(players, (list, tuple)):
                for idx, p in enumerate(players):
                    try:
                        if hasattr(p, 'color') and getattr(p, 'color', None) == getattr(self, 'color', None):
                            my_player_obj = p
                            break
                        if hasattr(p, 'name') and getattr(p, 'name', None) == getattr(self, 'name', None):
                            my_player_obj = p
                            break
                        if isinstance(p, dict) and (p.get('color') == getattr(self, 'color', None) or p.get('player_id') == getattr(self, 'player_id', None)):
                            my_player_obj = p
                            break
                    except Exception:
                        continue
                if my_player_obj is None and hasattr(self, 'index'):
                    try:
                        idx = getattr(self, 'index')
                        my_player_obj = players[idx]
                    except Exception:
                        my_player_obj = None
            else:
                my_player_obj = players
        except Exception:
            my_player_obj = None

        # Last resort: pick first available player in container
        try:
            if my_player_obj is None and players is not None:
                if isinstance(players, dict):
                    vals = list(players.values())
                    if vals:
                        my_player_obj = vals[0]
                elif isinstance(players, (list, tuple)) and players:
                    my_player_obj = players[0]
        except Exception:
            my_player_obj = None

        # Ordered extraction for each metric
        try:
            p = my_player_obj

            # Victory points
            for key in ('victory_points', 'victoryPoints', 'vp', 'points'):
                try:
                    if isinstance(p, dict) and key in p:
                        v = p.get(key)
                    else:
                        v = getattr(p, key, None)
                    if callable(v):
                        v = v()
                    iv = _coerce_count(v)
                    if iv is not None:
                        vp = iv
                        break
                except Exception:
                    continue
            # game helper
            if vp == 0:
                try:
                    if hasattr(game, 'get_victory_points'):
                        try:
                            val = game.get_victory_points(p)
                            iv = _coerce_count(val)
                            if iv is not None:
                                vp = iv
                        except Exception:
                            try:
                                val = game.get_victory_points(getattr(self, 'color', None))
                                iv = _coerce_count(val)
                                if iv is not None:
                                    vp = iv
                            except Exception:
                                pass
                except Exception:
                    pass

            # Settlements
            for key in ('settlements', 'settlement_positions', 'settlement_count', 'settles'):
                try:
                    if isinstance(p, dict) and key in p:
                        val = p.get(key)
                    else:
                        val = getattr(p, key, None)
                    if callable(val):
                        val = val()
                    iv = _coerce_count(val)
                    if iv is not None:
                        settlements = iv
                        break
                except Exception:
                    continue

            # Cities
            for key in ('cities', 'city_count'):
                try:
                    if isinstance(p, dict) and key in p:
                        val = p.get(key)
                    else:
                        val = getattr(p, key, None)
                    if callable(val):
                        val = val()
                    iv = _coerce_count(val)
                    if iv is not None:
                        cities = iv
                        break
                except Exception:
                    continue

            # Roads
            for key in ('roads', 'road_count'):
                try:
                    if isinstance(p, dict) and key in p:
                        val = p.get(key)
                    else:
                        val = getattr(p, key, None)
                    if callable(val):
                        val = val()
                    iv = _coerce_count(val)
                    if iv is not None:
                        roads = iv
                        break
                except Exception:
                    continue

            # Dev VP
            for key in ('dev_vp', 'dev_points'):
                try:
                    if isinstance(p, dict) and key in p:
                        val = p.get(key)
                    else:
                        val = getattr(p, key, None)
                    if callable(val):
                        val = val()
                    iv = _coerce_count(val)
                    if iv is not None:
                        dev_vp = iv
                        break
                except Exception:
                    continue
            # dev cards list inference
            if dev_vp == 0:
                try:
                    dev_cards = None
                    for key in ('dev_cards', 'development_cards'):
                        try:
                            if isinstance(p, dict) and key in p:
                                dev_cards = p.get(key)
                                break
                            dev_cards = getattr(p, key, None)
                            if dev_cards is not None:
                                break
                        except Exception:
                            continue
                    if dev_cards:
                        try:
                            count = 0
                            for d in dev_cards:
                                try:
                                    if getattr(d, 'is_victory', False) or getattr(d, 'type', None) == 'vp' or (isinstance(d, dict) and d.get('type') == 'vp'):
                                        count += 1
                                except Exception:
                                    continue
                            dev_vp = int(count)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Army
            for key in ('army_size', 'largest_army'):
                try:
                    if isinstance(p, dict) and key in p:
                        val = p.get(key)
                    else:
                        val = getattr(p, key, None)
                    if callable(val):
                        val = val()
                    iv = _coerce_count(val)
                    if iv is not None:
                        army = iv
                        break
                except Exception:
                    continue

        except Exception:
            # If something unexpected happened, keep defaults
            if DEBUG:
                print('FooPlayer._evaluate_state: unexpected exception during probe', file=sys.stderr)
                traceback.print_exc()

        # One-time diagnostic dump if primary metrics all zero
        try:
            if DEBUG and not _DUMPED_PLAYER_SCHEMA and vp == 0 and settlements == 0 and cities == 0 and roads == 0:
                print('\n=== DIAGNOSTIC DUMP (FooPlayer._evaluate_state) ===', file=sys.stderr)
                try:
                    print(f'Game type: {type(game)}', file=sys.stderr)
                    print(f"Game.state type: {type(getattr(game, 'state', None))}", file=sys.stderr)
                    print(f'Players container type: {type(players)}', file=sys.stderr)
                    try:
                        plen = len(players) if players is not None else 'N/A'
                    except Exception:
                        plen = 'N/A'
                    print(f'Players length: {plen}', file=sys.stderr)
                    if isinstance(players, dict):
                        print('Player keys sample:', list(players.keys())[:10], file=sys.stderr)
                        for k, v in list(players.items())[:4]:
                            print(f'-- key: {k} type: {type(v)}', file=sys.stderr)
                            try:
                                print('   repr:', repr(v)[:200], file=sys.stderr)
                            except Exception:
                                print('   repr: <unreprable>', file=sys.stderr)
                            try:
                                attrs = [a for a in dir(v) if not a.startswith('_')]
                                print('   attrs sample:', attrs[:40], file=sys.stderr)
                            except Exception:
                                print('   attrs: <failed>', file=sys.stderr)
                    elif isinstance(players, (list, tuple)):
                        for idx, v in enumerate(list(players)[:4]):
                            print(f'-- idx: {idx} type: {type(v)}', file=sys.stderr)
                            try:
                                print('   repr:', repr(v)[:200], file=sys.stderr)
                            except Exception:
                                print('   repr: <unreprable>', file=sys.stderr)
                            try:
                                attrs = [a for a in dir(v) if not a.startswith('_')]
                                print('   attrs sample:', attrs[:40], file=sys.stderr)
                            except Exception:
                                print('   attrs: <failed>', file=sys.stderr)
                except Exception:
                    print('Diagnostic dump failed', file=sys.stderr)
                    traceback.print_exc()
                _DUMPED_PLAYER_SCHEMA = True
        except Exception:
            pass

        # Compute composite score
        try:
            score = float(vp * 1000 + cities * 100 + settlements * 10 + roads * 3 + dev_vp * 50 + army * 50)
        except Exception:
            score = float(vp)

        if DEBUG:
            try:
                print(f'FooPlayer._evaluate_state: vp={vp}, cities={cities}, settlements={settlements}, roads={roads}, dev_vp={dev_vp}, army={army} -> score={score}', file=sys.stderr)
            except Exception:
                pass

        return score

    # ----------------- Rollout / evaluation that uses _evaluate_state -----------------
    def _evaluate_action(self, game_after_action: Game) -> float:
        """Evaluate a successor state. Prefer make_value_fn, otherwise use rollouts that
        evaluate terminal/leaf states using _evaluate_state.
        """
        adapters = _resolve_adapters()
        # Try fast value function first
        if adapters is not None and hasattr(adapters, 'make_value_fn'):
            try:
                vfn = adapters.make_value_fn(game_after_action)
                try:
                    val = vfn(game_after_action, getattr(self, 'color', None))
                except Exception:
                    val = vfn(game_after_action)
                try:
                    return float(val)
                except Exception:
                    pass
            except Exception:
                if DEBUG:
                    print('FooPlayer._evaluate_action: adapters.make_value_fn failed; falling back to rollouts', file=sys.stderr)

        # Monte-Carlo rollouts: evaluate each terminal/leaf with _evaluate_state
        scores = []
        for k in range(K_ROLLOUTS):
            try:
                rg = self._copy_game(game_after_action)
                if rg is None:
                    if DEBUG:
                        print('FooPlayer._evaluate_action: copy failed for rollout', file=sys.stderr)
                    continue
                depth = 0
                while depth < MAX_ROLLOUT_DEPTH:
                    # terminal?
                    try:
                        is_term = False
                        if hasattr(rg, 'is_terminal') and callable(getattr(rg, 'is_terminal')):
                            try:
                                if rg.is_terminal():
                                    is_term = True
                            except Exception:
                                pass
                        # Some games expose game_over or winning_color
                        if not is_term:
                            if hasattr(rg, 'game_over'):
                                try:
                                    if getattr(rg, 'game_over'):
                                        is_term = True
                                except Exception:
                                    pass
                        if is_term:
                            break
                    except Exception:
                        pass
                    # sample action
                    acts = []
                    if adapters is not None and hasattr(adapters, 'get_playable_actions'):
                        try:
                            acts = adapters.get_playable_actions(rg)
                        except Exception:
                            acts = self._get_playable_actions(rg)
                    else:
                        acts = self._get_playable_actions(rg)
                    if not acts:
                        break
                    try:
                        act = random.choice(list(acts))
                    except Exception:
                        act = acts[0]
                    applied = False
                    if adapters is not None and hasattr(adapters, 'do_action'):
                        try:
                            adapters.do_action(rg, act)
                            applied = True
                        except Exception:
                            applied = self._do_action(rg, act)
                    else:
                        applied = self._do_action(rg, act)
                    if not applied:
                        break
                    depth += 1
                # Leaf/terminal evaluation
                sc = self._evaluate_state(rg)
                scores.append(sc)
            except Exception:
                if DEBUG:
                    print('FooPlayer._evaluate_action: exception during rollout', file=sys.stderr)
                    traceback.print_exc()
                continue
        if not scores:
            return 0.0
        try:
            return float(statistics.mean(scores))
        except Exception:
            return float(sum(scores) / len(scores))

    # ------------------------ MCTS inner classes & helpers ------------------------
    class _MCTSNode:
        def __init__(self, game: Game, parent: Optional['FooPlayer._MCTSNode'] = None, action: Any = None, untried_actions: Optional[List[Any]] = None):
            self.game = game
            self.parent = parent
            self.action = action
            self.children: List['FooPlayer._MCTSNode'] = []
            self.visits = 0
            self.total_value = 0.0
            # actions not yet expanded from this node
            self.untried_actions = list(untried_actions) if untried_actions is not None else []

        def uct_score(self, c: float = UCT_C) -> float:
            if self.visits == 0:
                return float('inf')
            try:
                return (self.total_value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)
            except Exception:
                return float('inf')

        def best_child_by_uct(self, c: float = UCT_C) -> Optional['FooPlayer._MCTSNode']:
            if not self.children:
                return None
            return max(self.children, key=lambda ch: ch.uct_score(c))

        def add_child(self, child_node: 'FooPlayer._MCTSNode') -> None:
            self.children.append(child_node)

    def _is_terminal(self, game: Game) -> bool:
        try:
            if hasattr(game, 'is_terminal') and callable(getattr(game, 'is_terminal')):
                return bool(game.is_terminal())
        except Exception:
            pass
        try:
            if hasattr(game, 'game_over'):
                return bool(getattr(game, 'game_over'))
        except Exception:
            pass
        # try winning_color or similar
        try:
            if hasattr(game, 'winning_color'):
                wc = getattr(game, 'winning_color')
                if callable(wc):
                    try:
                        res = wc()
                        return res is not None
                    except Exception:
                        pass
                else:
                    return wc is not None
        except Exception:
            pass
        return False

    def _state_hash(self, game: Game) -> Tuple:
        """Create a small, best-effort hashable key for caching.

        This is intentionally simple to avoid depending on game internals.
        """
        try:
            st = getattr(game, 'state', None)
            tick = getattr(game, 'tick', None)
            turn = getattr(game, 'turn', None)
            # try to capture player VPs if available
            players = None
            try:
                if st is not None:
                    players = getattr(st, 'players', None)
            except Exception:
                players = getattr(game, 'players', None)
            vp_list = []
            try:
                if isinstance(players, dict):
                    for v in players.values():
                        try:
                            vp_list.append(getattr(v, 'victory_points', getattr(v, 'vp', None)))
                        except Exception:
                            vp_list.append(None)
                elif isinstance(players, (list, tuple)):
                    for v in players:
                        try:
                            vp_list.append(getattr(v, 'victory_points', getattr(v, 'vp', None)))
                        except Exception:
                            vp_list.append(None)
            except Exception:
                pass
            return (type(game).__name__, tick, turn, tuple(vp_list))
        except Exception:
            return (type(game).__name__, None, None, None)

    class _MCTS:
        def __init__(self, root_game: Game, root_actions: List[Any], player_color: Any, iterations: int = SIMULATIONS, max_depth: int = MAX_ROLLOUT_DEPTH):
            self.root_game = root_game
            self.root_actions = list(root_actions)
            self.player_color = player_color
            self.iterations = iterations
            self.max_depth = max_depth
            self.cache: Dict[Tuple, float] = {}
            # prepare root node with untried actions
            self.root = FooPlayer._MCTSNode(root_game, parent=None, action=None, untried_actions=self.root_actions)

        def run(self):
            for i in range(self.iterations):
                try:
                    node = self._select(self.root)
                    reward = self._simulate(node)
                    self._backpropagate(node, reward)
                except Exception:
                    if DEBUG:
                        print('FooPlayer._MCTS.run: exception during iteration', file=sys.stderr)
                        traceback.print_exc()
                    continue

        def _select(self, node: 'FooPlayer._MCTSNode') -> 'FooPlayer._MCTSNode':
            # descend until we find a node with untried actions or terminal
            current = node
            while True:
                if current.untried_actions:
                    # expand one action
                    return self._expand(current)
                if not current.children:
                    return current
                next_node = current.best_child_by_uct(UCT_C)
                if next_node is None:
                    return current
                current = next_node

        def _expand(self, node: 'FooPlayer._MCTSNode') -> 'FooPlayer._MCTSNode':
            # progressive widening: limit number of expansions considered
            try:
                n_actions = len(node.untried_actions)
                max_expand = min(MAX_ACTIONS_TO_EVALUATE, max(1, int(2 * math.sqrt(max(1, n_actions)))))
                # pick an action to expand (pop from untried)
                action = node.untried_actions.pop(0)
            except Exception:
                # nothing to expand
                return node
            # create child game state
            # copy parent game then apply action
            parent_game = node.game
            new_game = self._copy_for_mcts(parent_game)
            if new_game is None:
                return node
            applied = False
            adapters = _resolve_adapters()
            if adapters is not None and hasattr(adapters, 'do_action'):
                try:
                    adapters.do_action(new_game, action)
                    applied = True
                except Exception:
                    applied = self._do_action(new_game, action)
            else:
                applied = self._do_action(new_game, action)
            if not applied:
                return node
            child = FooPlayer._MCTSNode(new_game, parent=node, action=action, untried_actions=self._get_actions_for_node(new_game))
            node.add_child(child)
            return child

        def _get_actions_for_node(self, game: Game) -> List[Any]:
            adapters = _resolve_adapters()
            if adapters is not None and hasattr(adapters, 'get_playable_actions'):
                try:
                    a = adapters.get_playable_actions(game)
                    return list(a) if a is not None else []
                except Exception:
                    return self._get_playable_actions(game)
            return self._get_playable_actions(game)

        def _copy_for_mcts(self, game: Game) -> Optional[Game]:
            adapters = _resolve_adapters()
            if adapters is not None and hasattr(adapters, 'copy_game'):
                try:
                    return adapters.copy_game(game)
                except Exception:
                    return self._copy_game(game)
            return self._copy_game(game)

        def _simulate(self, node: 'FooPlayer._MCTSNode') -> float:
            # Run a random (biased) playout from node.game up to max_depth
            try:
                g = self._copy_for_mcts(node.game)
                if g is None:
                    return 0.0
                depth = 0
                adapters = _resolve_adapters()
                while depth < self.max_depth and not self._is_terminal(g):
                    acts = None
                    if adapters is not None and hasattr(adapters, 'get_playable_actions'):
                        try:
                            acts = adapters.get_playable_actions(g)
                        except Exception:
                            acts = self._get_playable_actions(g)
                    else:
                        acts = self._get_playable_actions(g)
                    if not acts:
                        break
                    # biased random: sample some actions and pick best by immediate eval
                    try:
                        choices = random.sample(list(acts), min(3, len(list(acts))))
                    except Exception:
                        choices = [random.choice(list(acts))]
                    best_act = None
                    best_score = -float('inf')
                    for a in choices:
                        g2 = self._copy_for_mcts(g)
                        if g2 is None:
                            continue
                        applied = False
                        if adapters is not None and hasattr(adapters, 'do_action'):
                            try:
                                adapters.do_action(g2, a)
                                applied = True
                            except Exception:
                                applied = self._do_action(g2, a)
                        else:
                            applied = self._do_action(g2, a)
                        if not applied:
                            continue
                        sc = self._evaluate_state(g2)
                        if sc > best_score:
                            best_score = sc
                            best_act = a
                    if best_act is None:
                        # fall back to pure random
                        try:
                            act = random.choice(list(acts))
                        except Exception:
                            break
                        applied2 = False
                        if adapters is not None and hasattr(adapters, 'do_action'):
                            try:
                                adapters.do_action(g, act)
                                applied2 = True
                            except Exception:
                                applied2 = self._do_action(g, act)
                        else:
                            applied2 = self._do_action(g, act)
                        if not applied2:
                            break
                    else:
                        applied2 = False
                        if adapters is not None and hasattr(adapters, 'do_action'):
                            try:
                                adapters.do_action(g, best_act)
                                applied2 = True
                            except Exception:
                                applied2 = self._do_action(g, best_act)
                        else:
                            applied2 = self._do_action(g, best_act)
                        if not applied2:
                            break
                    depth += 1
                # evaluate final state
                return self._evaluate_state(g)
            except Exception:
                if DEBUG:
                    print('FooPlayer._MCTS._simulate: exception', file=sys.stderr)
                    traceback.print_exc()
                return 0.0

        def _evaluate_state(self, game: Game) -> float:
            # Delegate to outer instance's evaluator for consistency
            return FooPlayer._evaluate_state(self_outer, game) if False else self_outer._evaluate_state(game)  # placeholder

        def _backpropagate(self, node: 'FooPlayer._MCTSNode', reward: float) -> None:
            current = node
            while current is not None:
                current.visits += 1
                try:
                    current.total_value += float(reward)
                except Exception:
                    current.total_value += 0.0
                current = current.parent

    # ------------------------ decide entry point (uses MCTS) ------------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Any:
        # Safe adapters handle
        adapters = _resolve_adapters()

        # If no actions, return None
        try:
            if not playable_actions:
                if DEBUG:
                    print('FooPlayer.decide: no playable actions -> None', file=sys.stderr)
                return None
        except Exception:
            pass

        try:
            actions = list(playable_actions)
        except Exception:
            try:
                return playable_actions[0]
            except Exception:
                return None

        if len(actions) == 1:
            if DEBUG:
                print('FooPlayer.decide: single action -> returning it', file=sys.stderr)
            return actions[0]

        # Sample candidate actions if too many
        if len(actions) > MAX_ACTIONS_TO_EVALUATE:
            try:
                candidates = random.sample(actions, MAX_ACTIONS_TO_EVALUATE)
            except Exception:
                candidates = actions[:MAX_ACTIONS_TO_EVALUATE]
        else:
            candidates = actions

        if DEBUG:
            print(f'Decide: running MCTS on {len(candidates)} candidates (SIMULATIONS={SIMULATIONS})', file=sys.stderr)

        # Run MCTS root over the candidate set
        try:
            # copy root game for safety
            root_game = self._copy_game(game)
            if root_game is None:
                if DEBUG:
                    print('FooPlayer.decide: failed to copy root game for MCTS; falling back to 1-ply', file=sys.stderr)
                return self._fallback_1ply(game, actions)

            # initialize MCTS instance
            # Bind outer instance helpers into MCTS by setting self_outer variable used in nested class
            global self_outer
            self_outer = self
            mcts = FooPlayer._MCTS(root_game, candidates, getattr(self, 'color', None), iterations=SIMULATIONS, max_depth=MAX_ROLLOUT_DEPTH)
            mcts.run()

            # choose best child by visits
            if not mcts.root.children:
                if DEBUG:
                    print('MCTS produced no children; falling back to 1-ply', file=sys.stderr)
                return self._fallback_1ply(game, actions)
            best_child = max(mcts.root.children, key=lambda n: n.visits)
            chosen = best_child.action
            if DEBUG:
                mean_val = (best_child.total_value / best_child.visits) if best_child.visits > 0 else 0.0
                print(f'Selected action (MCTS): {repr(chosen)} visits={best_child.visits} mean={mean_val:.3f}', file=sys.stderr)
            return chosen
        except Exception:
            if DEBUG:
                print('MCTS failed, falling back to 1-ply evaluator', file=sys.stderr)
                traceback.print_exc()
            return self._fallback_1ply(game, actions)

    # ------------------------ Fallback 1-ply evaluator ------------------------
    def _fallback_1ply(self, game: Game, actions: List[Any]) -> Any:
        """Existing 1-ply rollout-based evaluator used as a fallback."""
        if DEBUG:
            print('Using fallback 1-ply evaluator', file=sys.stderr)
        # Sample subset if needed
        if len(actions) > MAX_ACTIONS_TO_EVALUATE:
            try:
                candidates = random.sample(actions, MAX_ACTIONS_TO_EVALUATE)
            except Exception:
                candidates = actions[:MAX_ACTIONS_TO_EVALUATE]
        else:
            candidates = actions
        results = []
        for i, a in enumerate(candidates):
            try:
                ng = self._copy_game(game)
                if ng is None:
                    results.append((a, float('-inf'), 0.0))
                    continue
                applied = self._do_action(ng, a)
                if not applied:
                    results.append((a, float('-inf'), 0.0))
                    continue
                v = self._evaluate_action(ng)
                results.append((a, float(v), 0.0))
                if DEBUG:
                    print(f'Fallback action {i}: score={v}', file=sys.stderr)
            except Exception:
                if DEBUG:
                    print('Fallback evaluation error', file=sys.stderr)
                    traceback.print_exc()
                results.append((a, float('-inf'), 0.0))
        try:
            best_mean = max((m for (_, m, _) in results))
        except Exception:
            best_mean = float('-inf')
        best_actions = [a for (a, m, s) in results if m == best_mean]
        if not best_actions or best_mean == float('-inf'):
            try:
                return actions[0]
            except Exception:
                return None
        chosen = random.choice(best_actions)
        if DEBUG:
            print(f'Fallback selected: {repr(chosen)} mean={best_mean}', file=sys.stderr)
        return chosen
