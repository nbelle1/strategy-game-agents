import random
import statistics
import sys
import traceback
from typing import Any, Optional

# MUST import only from the adapters surface provided
from .adapters import (
    Game, Player, Color
)

# Optionally import a value function builder if adapters provide one
try:
    from .adapters import make_value_fn
    HAVE_VALUE_FN = True
except Exception:
    HAVE_VALUE_FN = False

# Hyperparameters for this player. Tweak across evolutions.
K_ROLLOUTS = 0  # rollouts disabled in this adapter-limited implementation
MAX_ROLLOUT_DEPTH = 10  # not used currently; kept for future use
MAX_ACTIONS_TO_EVALUATE = 12
DEBUG = True

# One-time diagnostic guard to avoid log flooding
_DUMPED_PLAYER_SCHEMA = False


class FooPlayer(Player):
    """A stronger FooPlayer that performs a 1-ply lookahead and evaluates
    the immediate successor state using a robust, defensive static evaluator.

    The evaluator tries many common access patterns to find a player object
    and extract victory points and common counts (settlements, cities,
    roads, dev VPs, army). If extraction fails it emits a one-time
    diagnostic dump to stderr to help adapt the probing logic.
    """

    def __init__(self, name: Optional[str] = None):
        # Use BLUE as the default color for this agent implementation
        super().__init__(Color.BLUE, name)
        # Local RNG can be seeded if desired; leave default for varied play
        random.seed(None)

    def decide(self, game: Game, playable_actions):
        """Choose an action from playable_actions.

        Strategy implemented:
        - If there are many playable actions, randomly sample up to
          MAX_ACTIONS_TO_EVALUATE actions to limit computation.
        - For each candidate action, copy the game, execute the action on the
          copy, and evaluate the resulting state with _evaluate_state().
        - Choose the action with the highest evaluation. Break ties randomly.

        Defensive behavior: any exception while copying/applying/evaluating
        will not crash the harness. Such actions are penalized.
        """
        # Defensive: if no actions available, return None
        if not playable_actions:
            if DEBUG:
                print('FooPlayer.decide: no playable_actions -> returning None')
            return None

        # Convert playable_actions to a list so we can sample and index
        try:
            actions = list(playable_actions)
        except Exception:
            if DEBUG:
                print('FooPlayer.decide: playable_actions not list-like; defaulting to first')
            try:
                return playable_actions[0]
            except Exception:
                return None

        # Sample candidate actions if there are too many
        if len(actions) > MAX_ACTIONS_TO_EVALUATE:
            candidates = random.sample(actions, MAX_ACTIONS_TO_EVALUATE)
            if DEBUG:
                print(f'FooPlayer.decide: sampled {len(candidates)} of {len(actions)} actions to evaluate')
        else:
            candidates = actions
            if DEBUG:
                print(f'FooPlayer.decide: evaluating all {len(candidates)} actions')

        # Evaluate each candidate action by applying it to a copy of the game
        scores = []  # list of (action, score)
        for i, action in enumerate(candidates):
            try:
                # Copy the game to avoid mutating the original
                new_game = game.copy()

                # Apply the candidate action on the copied game.
                # The standard Game API exposes execute(action) to apply an action.
                # We try both .execute and .apply for defensive compatibility.
                executed = False
                try:
                    new_game.execute(action)
                    executed = True
                except Exception:
                    try:
                        new_game.apply(action)
                        executed = True
                    except Exception:
                        executed = False

                if not executed:
                    # If we couldn't apply the action on the copy, mark it as
                    # very poor and continue.
                    if DEBUG:
                        print(f'FooPlayer.decide: failed to execute candidate action {i}; marking score -inf')
                    scores.append((action, float('-inf')))
                    continue

                # If a fast value function is available from adapters, try it
                if HAVE_VALUE_FN:
                    try:
                        # Defensive: make_value_fn may accept a game or return a
                        # function that expects (game, player_color). Try both.
                        vfn = make_value_fn(new_game)
                        try:
                            # Try calling vfn with (game, color)
                            val = vfn(new_game, self.color)
                        except Exception:
                            # Try calling vfn with only game
                            val = vfn(new_game)
                        score = float(val)
                        scores.append((action, score))
                        if DEBUG:
                            print(f'FooPlayer.decide: action #{i} -> value_fn score {score}')
                        continue
                    except Exception as e:
                        if DEBUG:
                            print(f'FooPlayer.decide: make_value_fn failed for action #{i}: {e}; falling back to static eval')

                # Evaluate the successor state with our static evaluator
                score = self._evaluate_state(new_game)
                scores.append((action, score))
                if DEBUG:
                    print(f'FooPlayer.decide: action #{i} -> score {score}')

            except Exception as e:
                # Catch-all: do not let the player crash the harness. Penalize
                # the action and continue evaluating others.
                if DEBUG:
                    print(f'FooPlayer.decide: exception while evaluating action #{i}: {e}! Marking -inf')
                    traceback.print_exc()
                scores.append((action, float('-inf')))

        # Choose the best action. If all are -inf or evaluation failed, fall back
        # to the original first-action policy.
        if not scores:
            if DEBUG:
                print('FooPlayer.decide: no scores produced -> defaulting to first action')
            return actions[0]

        # Compute the maximum score
        max_score = max(score for (_, score) in scores)
        # Filter all actions that have the max score (handle ties)
        best_candidates = [a for (a, s) in scores if s == max_score]

        if not best_candidates or max_score == float('-inf'):
            # All evaluations failed; fallback
            if DEBUG:
                print('FooPlayer.decide: all evaluations failed -> defaulting to first action')
            return actions[0]

        chosen = random.choice(best_candidates)
        if DEBUG:
            try:
                # Try to pretty-print a small summary for debugging
                print(f'FooPlayer.decide: selected action -> {repr(chosen)} with score {max_score}')
            except Exception:
                print('FooPlayer.decide: selected an action (repr failed)')

        return chosen

    def _evaluate_state(self, game: Game) -> float:
        """Static evaluation of a game state from this player's perspective.

        Robust player lookup and extraction plan implemented here. This
        function follows the Strategizer's recommendations for attribute
        probing and emits a one-time diagnostic dump if probing fails to
        find useful information.
        """
        global _DUMPED_PLAYER_SCHEMA

        # Default metric values
        vp = 0
        settlements = 0
        cities = 0
        roads = 0
        dev_vp = 0
        army = 0

        # Defensive player container lookup
        players = None
        try:
            players = getattr(game, 'state', None)
            if players is not None:
                players = getattr(players, 'players', None) or getattr(game, 'players', None)
        except Exception:
            players = None

        if players is None:
            try:
                players = getattr(game, 'players', None)
            except Exception:
                players = None

        if players is None:
            try:
                players = getattr(game, 'player_state', None)
            except Exception:
                players = None

        # Helper: attempt to canonicalize keys we will probe
        def _candidate_keys():
            keys = []
            keys.append(getattr(self, 'color', None))
            try:
                keys.append(str(getattr(self, 'color', None)))
            except Exception:
                pass
            keys.append(getattr(getattr(self, 'color', None), 'name', None))
            try:
                keys.append(int(getattr(self, 'color', None)))
            except Exception:
                pass
            return [k for k in keys if k is not None]

        player_obj = None
        player_key_used = None

        # If players is a dict-like mapping, try direct key access then fallbacks
        try:
            if isinstance(players, dict):
                for key in _candidate_keys():
                    try:
                        if key in players:
                            player_obj = players[key]
                            player_key_used = key
                            break
                    except Exception:
                        # Some keys may not be valid for 'in' checks; ignore
                        continue
                # Fallback: iterate values and match by attributes
                if player_obj is None:
                    for p in players.values():
                        try:
                            if (hasattr(p, 'color') and getattr(p, 'color', None) == getattr(self, 'color', None)):
                                player_obj = p
                                break
                            if isinstance(p, dict) and ('color' in p and p.get('color') == getattr(self, 'color', None)):
                                player_obj = p
                                break
                            if hasattr(p, 'name') and getattr(p, 'name', None) == getattr(self, 'name', None):
                                player_obj = p
                                break
                        except Exception:
                            continue

            # If players is a list/tuple/iterable, iterate and match by attributes
            elif isinstance(players, (list, tuple)):
                for p in players:
                    try:
                        if (hasattr(p, 'color') and getattr(p, 'color', None) == getattr(self, 'color', None)):
                            player_obj = p
                            break
                        if hasattr(p, 'name') and getattr(p, 'name', None) == getattr(self, 'name', None):
                            player_obj = p
                            break
                        if isinstance(p, dict) and ('color' in p and p.get('color') == getattr(self, 'color', None)):
                            player_obj = p
                            break
                    except Exception:
                        continue
                # Fallback to index mapping if available
                if player_obj is None and hasattr(self, 'index'):
                    try:
                        idx = getattr(self, 'index')
                        player_obj = players[idx]
                        player_key_used = idx
                    except Exception:
                        player_obj = None

            # If players is a single object (not mapping/list), treat as the player container
            else:
                # If game exposes a direct player object
                if players is not None:
                    player_obj = players

        except Exception:
            player_obj = None

        # As a last resort choose a first-entry fallback to avoid crashing
        if player_obj is None:
            try:
                # If mapping-like
                if isinstance(players, dict):
                    vals = list(players.values())
                    if vals:
                        player_obj = vals[0]
                        player_key_used = list(players.keys())[0]
                elif isinstance(players, (list, tuple)) and len(players) > 0:
                    player_obj = players[0]
                    player_key_used = 0
                else:
                    # Give up; player_obj remains None
                    player_obj = None
            except Exception:
                player_obj = None

        # Now attempt to extract metrics from player_obj using ordered attempts
        def _to_int(x: Any) -> Optional[int]:
            try:
                if x is None:
                    return None
                if isinstance(x, bool):
                    return int(x)
                if isinstance(x, (list, tuple, set)):
                    return len(x)
                # If it's callable, call it and then try convert
                if callable(x):
                    x = x()
                return int(x)
            except Exception:
                return None

        try:
            p = player_obj
            # Victory Points (vp)
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

            # If game exposes a helper, try it
            if vp == 0:
                try:
                    if hasattr(game, 'get_victory_points'):
                        try:
                            # Try passing player object
                            val = game.get_victory_points(p)
                            vv = _to_int(val)
                            if vv is not None:
                                vp = vv
                        except Exception:
                            # Maybe get_victory_points expects a player index/color
                            try:
                                val = game.get_victory_points(getattr(self, 'color', None))
                                vv = _to_int(val)
                                if vv is not None:
                                    vp = vv
                            except Exception:
                                pass
                except Exception:
                    pass

            # Settlements
            for attr in ('settlements', 'settlement_positions', 'settlement_count', 'settle_list', 'settles'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        settlements = iv
                        break
                except Exception:
                    continue

            # Cities
            for attr in ('cities', 'city_count'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        cities = iv
                        break
                except Exception:
                    continue

            # Roads
            for attr in ('roads', 'road_count'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        roads = iv
                        break
                except Exception:
                    continue

            # Dev VP
            for attr in ('dev_vp', 'dev_points'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        dev_vp = iv
                        break
                except Exception:
                    continue
            # If not found, try counting vp-like dev cards
            if dev_vp == 0:
                try:
                    if hasattr(p, 'dev_cards'):
                        cards = getattr(p, 'dev_cards')
                        if callable(cards):
                            cards = cards()
                        # Count cards that look like victory VPs
                        count = 0
                        for d in cards:
                            try:
                                if getattr(d, 'is_victory', False) or getattr(d, 'type', None) == 'vp':
                                    count += 1
                            except Exception:
                                continue
                        if count:
                            dev_vp = count
                except Exception:
                    pass

            # Army
            for attr in ('army_size', 'largest_army'):
                try:
                    if isinstance(p, dict) and attr in p:
                        val = p[attr]
                    else:
                        val = getattr(p, attr, None)
                    if callable(val):
                        val = val()
                    iv = _to_int(val)
                    if iv is not None:
                        army = iv
                        break
                except Exception:
                    continue

        except Exception as e:
            if DEBUG:
                print('FooPlayer._evaluate_state: exception during probing:', e, file=sys.stderr)
                traceback.print_exc()
            # In the event of unexpected errors, return a very low score to
            # discourage picking states we couldn't evaluate.
            return float(-1e6)

        # If we failed to extract useful metrics, emit a one-time diagnostic
        # dump to help adjust the probing logic. This prints to stderr and
        # is gated by a process-level flag so it only happens once.
        try:
            if DEBUG and not _DUMPED_PLAYER_SCHEMA and vp == 0 and settlements == 0 and cities == 0 and roads == 0:
                print('\n=== DIAGNOSTIC DUMP (FooPlayer) ===', file=sys.stderr)
                try:
                    print(f'Game type: {type(game)}', file=sys.stderr)
                    print(f'Game.state type: {type(getattr(game, "state", None))}', file=sys.stderr)
                    print(f'Players container type: {type(players)}', file=sys.stderr)
                    try:
                        plen = len(players) if players is not None else 'N/A'
                    except Exception:
                        plen = 'N/A'
                    print(f"Players length: {plen}", file=sys.stderr)

                    # If it's a mapping, show keys and a sample of values
                    if isinstance(players, dict):
                        print('Player keys:', list(players.keys())[:10], file=sys.stderr)
                        cnt = 0
                        for k, v in list(players.items())[:4]:
                            print(f'-- Player key: {k} type: {type(v)}', file=sys.stderr)
                            try:
                                preview = repr(v)
                                print('   repr:', preview[:200], file=sys.stderr)
                            except Exception:
                                print('   repr: <unrepr-able>', file=sys.stderr)
                            try:
                                attrs = [a for a in dir(v) if not a.startswith('_')]
                                print('   attrs sample:', attrs[:40], file=sys.stderr)
                            except Exception:
                                print('   attrs: <failed>', file=sys.stderr)
                            cnt += 1
                    elif isinstance(players, (list, tuple)):
                        for idx, v in enumerate(list(players)[:4]):
                            print(f'-- Player idx: {idx} type: {type(v)}', file=sys.stderr)
                            try:
                                preview = repr(v)
                                print('   repr:', preview[:200], file=sys.stderr)
                            except Exception:
                                print('   repr: <unrepr-able>', file=sys.stderr)
                            try:
                                attrs = [a for a in dir(v) if not a.startswith('_')]
                                print('   attrs sample:', attrs[:40], file=sys.stderr)
                            except Exception:
                                print('   attrs: <failed>', file=sys.stderr)
                    else:
                        # Print a small repr of the players object
                        try:
                            print('Players repr:', repr(players)[:400], file=sys.stderr)
                        except Exception:
                            print('Players repr: <failed>', file=sys.stderr)

                except Exception:
                    print('Diagnostic dump failed to fully collect details', file=sys.stderr)
                    traceback.print_exc()
                # mark dumped so we don't flood logs
                _DUMPED_PLAYER_SCHEMA = True
        except Exception:
            # If diagnostic printing causes an issue, swallow it -- do not
            # crash the harness for debugging output.
            try:
                traceback.print_exc()
            except Exception:
                pass

        # Build a composite score. Primary contributor is victory points.
        # Use the Strategizer's recommended formula (VP prioritized):
        # score = vp*1000 + cities*100 + settlements*10 + roads*3 + dev_vp*50 + army*50
        try:
            score = float(vp * 1000 + cities * 100 + settlements * 10 + roads * 3 + dev_vp * 50 + army * 50)
        except Exception:
            # Defensive fallback
            score = float(vp)

        if DEBUG:
            try:
                print(f'FooPlayer._evaluate_state: vp={vp}, cities={cities}, settlements={settlements}, roads={roads}, dev_vp={dev_vp}, army={army} -> score={score}')
            except Exception:
                print('FooPlayer._evaluate_state: computed a score (repr failed)')

        return score
