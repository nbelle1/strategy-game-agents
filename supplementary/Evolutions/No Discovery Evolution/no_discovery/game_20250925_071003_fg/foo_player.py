import math
import random
import statistics
import sys
import traceback
from typing import Any, Optional, Iterable

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

# Hyperparameters (as requested in META)
K_ROLLOUTS = 30
MAX_ROLLOUT_DEPTH = 10
MAX_ACTIONS_TO_EVALUATE = 12
DEBUG = True

# One-time diagnostic guard
_DUMPED_PLAYER_SCHEMA = False


class FooPlayer(Player):
    """Player that evaluates successor states with Monte-Carlo rollouts.

    Behavior (decide):
    - If playable_actions empty -> None
    - If many actions, sample up to MAX_ACTIONS_TO_EVALUATE
    - For each candidate action, copy game, apply action, then evaluate the
      successor state using make_value_fn (if available) or K_ROLLOUTS random
      rollouts (depth-limited). Extract VP from rollout endpoints using a
      robust probing function.
    - Choose action with highest average rollout score. Robust fallbacks and
      defensive error handling ensure the harness does not crash.
    """

    def __init__(self, name: Optional[str] = None):
        # Defensive constructor call: Player base signatures may vary
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
        if HAVE_ADAPTER_HELPERS:
            try:
                return copy_game(game)
            except Exception:
                if DEBUG:
                    print('FooPlayer._copy_game: adapter copy_game failed; falling back', file=sys.stderr)
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

    def _get_playable_actions(self, game: Game) -> list:
        """Get playable actions using adapter helper if possible, else probe game.

        Returns a list (possibly empty).
        """
        if HAVE_ADAPTER_HELPERS:
            try:
                acts = get_playable_actions(game)
                if acts is None:
                    return []
                return list(acts)
            except Exception:
                if DEBUG:
                    print('FooPlayer._get_playable_actions: adapter get_playable_actions failed; falling back', file=sys.stderr)
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
        if HAVE_ADAPTER_HELPERS:
            try:
                do_action(game, action)
                return True
            except Exception:
                if DEBUG:
                    print('FooPlayer._do_action: adapter do_action failed; falling back', file=sys.stderr)
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
        # Try fast value function first
        if HAVE_VALUE_FN:
            try:
                vfn = make_value_fn(game_after_action)
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
                    print('FooPlayer._evaluate_action: make_value_fn failed; falling back to rollouts', file=sys.stderr)

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
                        if is_term:
                            break
                    except Exception:
                        pass
                    # sample action
                    acts = []
                    if HAVE_ADAPTER_HELPERS:
                        try:
                            acts = get_playable_actions(rg)
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
                    if HAVE_ADAPTER_HELPERS:
                        try:
                            do_action(rg, act)
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

    # ------------------------ decide entry point ------------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Any:
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
            if DEBUG:
                print(f'Evaluating {len(candidates)} actions (sampled from {len(actions)}) with {K_ROLLOUTS} rollouts, depth {MAX_ROLLOUT_DEPTH}', file=sys.stderr)
        else:
            candidates = actions
            if DEBUG:
                print(f'Evaluating all {len(candidates)} actions with {K_ROLLOUTS} rollouts, depth {MAX_ROLLOUT_DEPTH}', file=sys.stderr)

        # Score each candidate using _evaluate_action
        results = []  # list of (action, mean_score, std)
        for i, a in enumerate(candidates):
            try:
                # Copy root game
                ng = None
                if HAVE_ADAPTER_HELPERS:
                    try:
                        ng = copy_game(game)
                    except Exception:
                        ng = self._copy_game(game)
                else:
                    ng = self._copy_game(game)

                if ng is None:
                    if DEBUG:
                        print(f'Action {i}: failed to copy root game; assigning very low score', file=sys.stderr)
                    results.append((a, float('-inf'), 0.0))
                    continue

                # Apply action on copy
                applied = False
                if HAVE_ADAPTER_HELPERS:
                    try:
                        do_action(ng, a)
                        applied = True
                    except Exception:
                        applied = self._do_action(ng, a)
                else:
                    applied = self._do_action(ng, a)

                if not applied:
                    if DEBUG:
                        print(f'Action {i}: failed to apply action on copy; marking very low score', file=sys.stderr)
                    results.append((a, float('-inf'), 0.0))
                    continue

                # Evaluate successor state
                try:
                    if HAVE_VALUE_FN:
                        try:
                            vfn = make_value_fn(ng)
                            try:
                                v = vfn(ng, getattr(self, 'color', None))
                            except Exception:
                                v = vfn(ng)
                            v = float(v)
                            results.append((a, v, 0.0))
                            if DEBUG:
                                print(f'Action {i}: value_fn returned {v}', file=sys.stderr)
                            continue
                        except Exception:
                            if DEBUG:
                                print(f'Action {i}: make_value_fn failed; falling back to rollouts', file=sys.stderr)
                    # Run rollouts
                    vals = []
                    for r in range(K_ROLLOUTS):
                        try:
                            rg = self._copy_game(ng)
                            if rg is None:
                                continue
                            depth = 0
                            while depth < MAX_ROLLOUT_DEPTH:
                                acts = []
                                if HAVE_ADAPTER_HELPERS:
                                    try:
                                        acts = get_playable_actions(rg)
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
                                applied2 = False
                                if HAVE_ADAPTER_HELPERS:
                                    try:
                                        do_action(rg, act)
                                        applied2 = True
                                    except Exception:
                                        applied2 = self._do_action(rg, act)
                                else:
                                    applied2 = self._do_action(rg, act)
                                if not applied2:
                                    break
                                depth += 1
                            # At the end of rollout evaluate using _evaluate_state
                            vals.append(self._evaluate_state(rg))
                        except Exception:
                            if DEBUG:
                                print('Exception during rollout for action', i, file=sys.stderr)
                                traceback.print_exc()
                            continue
                    if not vals:
                        mean_v = 0.0
                        std_v = 0.0
                    else:
                        mean_v = float(statistics.mean(vals))
                        try:
                            std_v = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
                        except Exception:
                            std_v = 0.0
                    results.append((a, mean_v, std_v))
                    if DEBUG:
                        print(f'Action {i}: mean={mean_v:.3f} std={std_v:.3f} over {len(vals)} rollouts', file=sys.stderr)
                except Exception:
                    if DEBUG:
                        print(f'Action {i}: evaluation error', file=sys.stderr)
                        traceback.print_exc()
                    results.append((a, float('-inf'), 0.0))
            except Exception:
                if DEBUG:
                    print(f'Unexpected error evaluating action {i}', file=sys.stderr)
                    traceback.print_exc()
                results.append((a, float('-inf'), 0.0))

        # Choose best action by mean score (break ties randomly)
        try:
            best_mean = max((m for (_, m, _) in results))
        except Exception:
            best_mean = float('-inf')

        best_actions = [a for (a, m, s) in results if m == best_mean]
        if not best_actions or best_mean == float('-inf'):
            if DEBUG:
                print('All action evaluations failed or returned -inf; falling back to first action', file=sys.stderr)
            try:
                return actions[0]
            except Exception:
                return None

        chosen = random.choice(best_actions)
        if DEBUG:
            print(f'Selected action: {repr(chosen)} with mean score {best_mean}', file=sys.stderr)
        return chosen
