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
    """Player that uses a 1-ply lookahead with Monte-Carlo rollouts.

    Implementation notes:
    - Prefers adapter helpers copy_game/get_playable_actions/do_action when available.
    - If adapter helpers are not available, falls back to defensive probing of
      game.copy()/game.clone(), game.execute/apply/do_action, and game.state helpers.
    - Robust extraction of victory points from rollout terminal states using
      a probing plan. Emits a one-time diagnostic dump if extraction finds nothing.
    """

    def __init__(self, name: Optional[str] = None):
        # Try various Player constructors defensively
        try:
            super().__init__(Color.BLUE, name)
        except Exception:
            try:
                super().__init__()
            except Exception:
                # Last resort: continue without calling base
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
                    print('FooPlayer._copy_game: copy_game failed; falling back', file=sys.stderr)
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
                print('FooPlayer._copy_game: deep copy failed', file=sys.stderr)
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
        # Probe common names on game
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

    # ----------------- Robust extraction for rollouts -----------------
    def _extract_vp_from_game(self, game: Game, my_color: Any) -> int:
        """Try to extract victory points for my_color using ordered probes.

        Returns integer VP or 0 on failure.
        Also prints a one-time diagnostic dump if nothing usable is found.
        """
        global _DUMPED_PLAYER_SCHEMA

        vp = 0

        # Attempt to find player container
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

        # Candidate keys for mapping lookup
        def _candidate_keys():
            keys = []
            keys.append(getattr(my_color, 'value', None) if hasattr(my_color, 'value') else None)
            try:
                keys.append(str(my_color))
            except Exception:
                pass
            try:
                keys.append(getattr(my_color, 'name', None))
            except Exception:
                pass
            try:
                keys.append(int(my_color))
            except Exception:
                pass
            return [k for k in keys if k is not None]

        player_obj = None
        try:
            if isinstance(players, dict):
                for key in _candidate_keys():
                    try:
                        if key in players:
                            player_obj = players[key]
                            break
                    except Exception:
                        continue
                if player_obj is None:
                    for p in players.values():
                        try:
                            if hasattr(p, 'color') and getattr(p, 'color', None) == my_color:
                                player_obj = p
                                break
                            if isinstance(p, dict) and p.get('color', None) == my_color:
                                player_obj = p
                                break
                        except Exception:
                            continue
            elif isinstance(players, (list, tuple)):
                for p in players:
                    try:
                        if hasattr(p, 'color') and getattr(p, 'color', None) == my_color:
                            player_obj = p
                            break
                        if isinstance(p, dict) and p.get('color', None) == my_color:
                            player_obj = p
                            break
                    except Exception:
                        continue
            else:
                player_obj = players
        except Exception:
            player_obj = None

        # Fallback to scanning game.state.players or first entry
        if player_obj is None:
            try:
                if isinstance(players, dict):
                    vals = list(players.values())
                    if vals:
                        player_obj = vals[0]
                elif isinstance(players, (list, tuple)) and players:
                    player_obj = players[0]
            except Exception:
                player_obj = None

        # Helper to coerce to int
        def _to_int(x: Any) -> Optional[int]:
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

        try:
            p = player_obj
            # Victory points candidates
            for attr in ('victory_points', 'victoryPoints', 'vp', 'points'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        vp = iv
                        break
                except Exception:
                    continue

            # Try game helper
            if vp == 0:
                try:
                    if hasattr(game, 'get_victory_points'):
                        try:
                            val = game.get_victory_points(p)
                            iv = _to_int(val)
                            if iv is not None:
                                vp = iv
                        except Exception:
                            try:
                                val = game.get_victory_points(my_color)
                                iv = _to_int(val)
                                if iv is not None:
                                    vp = iv
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            vp = 0

        # Diagnostic dump if we couldn't find anything
        try:
            if DEBUG and not _DUMPED_PLAYER_SCHEMA and vp == 0:
                print('\n=== DIAGNOSTIC DUMP (FooPlayer - rollout extraction) ===', file=sys.stderr)
                try:
                    print(f'Game type: {type(game)}', file=sys.stderr)
                    print(f'Game.state type: {type(getattr(game, \"state\", None))}', file=sys.stderr)
                    print(f'Players container type: {type(players)}', file=sys.stderr)
                    try:
                        plen = len(players) if players is not None else 'N/A'
                    except Exception:
                        plen = 'N/A'
                    print(f'Players length: {plen}', file=sys.stderr)
                    if isinstance(players, dict):
                        print('Player keys:', list(players.keys())[:10], file=sys.stderr)
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

        return int(vp or 0)

    # ----------------- Monte-Carlo evaluation for a successor -----------------
    def _evaluate_action(self, game_after_action: Game) -> float:
        """Evaluate a game state after applying one candidate action.

        If make_value_fn is available and works, prefer it. Otherwise run
        K_ROLLOUTS random rollouts and return the average VP.
        """
        # Try fast value function first
        if HAVE_VALUE_FN:
            try:
                vfn = make_value_fn(game_after_action)
                try:
                    val = vfn(game_after_action, getattr(self, 'color', None))
                except Exception:
                    val = vfn(game_after_action)
                # Interpret val as estimated victory points or score
                try:
                    return float(val)
                except Exception:
                    pass
            except Exception:
                if DEBUG:
                    print('FooPlayer._evaluate_action: make_value_fn failed; falling back to rollouts', file=sys.stderr)

        # Monte-Carlo rollouts
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
                        if hasattr(rg, 'is_terminal') and callable(getattr(rg, 'is_terminal')) and rg.is_terminal():
                            break
                    except Exception:
                        pass
                    actions = []
                    if HAVE_ADAPTER_HELPERS:
                        try:
                            actions = get_playable_actions(rg)
                        except Exception:
                            actions = self._get_playable_actions(rg)
                    else:
                        actions = self._get_playable_actions(rg)
                    if not actions:
                        break
                    try:
                        act = random.choice(list(actions))
                    except Exception:
                        act = actions[0]
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
                # After rollout, extract VP for our color
                vp = self._extract_vp_from_game(rg, getattr(self, 'color', None))
                scores.append(vp)
            except Exception:
                if DEBUG:
                    print('FooPlayer._evaluate_action: exception during rollout', file=sys.stderr)
                    traceback.print_exc()
                continue
        if not scores:
            return 0.0
        # Return average VP as float
        try:
            return float(statistics.mean(scores))
        except Exception:
            return float(sum(scores) / len(scores))

    # ------------------------ decide entry point ------------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Any:
        # Handle empty
        try:
            if not playable_actions:
                if DEBUG:
                    print('FooPlayer.decide: no playable actions -> None')
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
                print('FooPlayer.decide: single action -> returning it')
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

        # Score each candidate
        results = []  # list of (action, mean, std)
        for i, a in enumerate(candidates):
            try:
                # Apply action on a copy of the root game
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

                # Try adapter do_action first
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
                                # apply
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
                            vp = self._extract_vp_from_game(rg, getattr(self, 'color', None))
                            vals.append(vp)
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
