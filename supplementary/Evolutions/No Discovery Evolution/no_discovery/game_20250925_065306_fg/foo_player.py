import math
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
ITERATIONS = 300              # MCTS iterations per decision (budget)
MAX_DEPTH = 8                 # rollout / simulation depth
MAX_ACTIONS_TO_EVALUATE = 12  # progressive widening cap at root
UCT_C = 1.4                   # exploration constant for UCT
DEBUG = True                  # enable verbose debugging prints

# One-time diagnostic guard to avoid log flooding
_DUMPED_PLAYER_SCHEMA = False


class FooPlayer(Player):
    """A stronger FooPlayer that performs MCTS (budgeted UCT) with a
    robust static evaluator fallback.

    Behavior summary:
    - If only one legal action -> return it.
    - Otherwise run a budgeted MCTS search from the current state using
      the adapters-exposed game copy/execute APIs. The MCTS uses the
      static evaluator (self._evaluate_state) or make_value_fn (if
      available) at leaf nodes.
    - All adapter calls are defensive: multiple possible method names
      are tried (copy/clone, execute/apply, get_playable_actions/legal_actions).
    - If MCTS fails for any reason, fall back to a 1-ply static eval
      over a sampled set of actions (existing behavior).

    The evaluator extracts VP/settlements/cities/roads/dev_vp/army using
    an ordered probing plan and emits a one-time diagnostic dump if it
    cannot find any of the core metrics.
    """

    def __init__(self, name: Optional[str] = None):
        # Try to initialize the base Player with a default color if
        # constructor signatures differ across harness versions.
        try:
            super().__init__(Color.BLUE, name)
        except Exception:
            try:
                super().__init__()
            except Exception:
                # Last resort: ignore and hope harness injects necessary fields
                pass

        # Local RNG for tie-breaking and sampling
        random.seed(None)

    # ----------------------------- MCTS -----------------------------
    class _MCTSNode:
        def __init__(self, game: Game, parent: Optional['FooPlayer._MCTSNode'] = None, action: Any = None):
            self.game = game
            self.parent = parent
            self.action = action
            self.children: list[FooPlayer._MCTSNode] = []
            self.untried_actions: list = []  # to be populated at expansion
            self.visits = 0
            self.total_value = 0.0

        def avg_value(self) -> float:
            return self.total_value / self.visits if self.visits > 0 else 0.0

        def uct_score(self, c: float = UCT_C) -> float:
            # UCT value; if unvisited return +inf to ensure exploration
            if self.visits == 0:
                return float('inf')
            if self.parent is None or self.parent.visits == 0:
                return self.avg_value()
            return self.avg_value() + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def _get_playable_actions(self, game: Game):
        """Defensive attempt to query playable actions from a game state.

        Tries several common API names and returns a list (may be empty).
        """
        # Try a few common method/attribute names; return list(actions)
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
                if callable(attr):
                    res = attr()
                else:
                    res = attr
                if res is None:
                    continue
                # Ensure it's iterable/list
                try:
                    return list(res)
                except Exception:
                    return [res]
            except Exception:
                continue

        # As a last resort, try to look inside game.state for a helper
        try:
            st = getattr(game, 'state', None)
            if st is not None:
                for name in ('get_playable_actions', 'playable_actions', 'legal_actions'):
                    try:
                        attr = getattr(st, name, None)
                        if callable(attr):
                            res = attr()
                            if res is not None:
                                return list(res)
                    except Exception:
                        continue
        except Exception:
            pass

        return []

    def _copy_game(self, game: Game) -> Optional[Game]:
        """Defensive copy of the game state using several possible APIs."""
        try:
            # Preferred
            return game.copy()
        except Exception:
            pass
        try:
            # Alternative name
            clone = getattr(game, 'clone', None)
            if callable(clone):
                return clone()
        except Exception:
            pass
        try:
            # Try a deeper copy as last resort
            import copy

            return copy.deepcopy(game)
        except Exception:
            return None

    def _apply_action(self, game: Game, action: Any) -> bool:
        """Try to apply an action on the given game; return True on success."""
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

    def _is_terminal(self, game: Game) -> bool:
        """Detect terminal/finished game state."""
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
        try:
            # some engines expose winning_color or similar
            if hasattr(game, 'winning_color'):
                wc = getattr(game, 'winning_color')
                if wc is not None:
                    return True
        except Exception:
            pass
        return False

    def _reward_from_game(self, game: Game, my_color: Any) -> float:
        """Compute a normalized reward in [0,1] for my_color in game.

        Prefer make_value_fn if available; otherwise attempt to extract
        victory points via the static _evaluate_state and normalize.
        Terminal wins yield reward 1.0.
        """
        # Terminal check: if game declares a winner, give 1.0 for win
        try:
            if hasattr(game, 'winning_color'):
                wc = getattr(game, 'winning_color')
                if wc is not None:
                    try:
                        if wc == my_color:
                            return 1.0
                        else:
                            return 0.0
                    except Exception:
                        # fallback to string/enum compare
                        try:
                            if str(wc) == str(my_color):
                                return 1.0 if str(wc) == str(my_color) else 0.0
                        except Exception:
                            pass
        except Exception:
            pass

        # Use value function if present
        if HAVE_VALUE_FN:
            try:
                vfn = make_value_fn(game)
                try:
                    val = vfn(game, my_color)
                except Exception:
                    val = vfn(game)
                # Normalize val to [0,1] assuming scale roughly 0..10 VP
                try:
                    v = float(val)
                    return max(0.0, min(1.0, v / 10.0))
                except Exception:
                    pass
            except Exception:
                # Fall through to static eval
                if DEBUG:
                    print('FooPlayer._reward_from_game: make_value_fn failed; falling back', file=sys.stderr)

        # Fall back: use static evaluator to estimate VP and normalize
        try:
            score = float(self._evaluate_state(game))
            # Our static score is not VPs but a weighted sum; attempt to convert
            # back to an approximate VP by dividing by 1000 (since VP*1000 is dominant)
            approx_vp = score / 1000.0
            return max(0.0, min(1.0, approx_vp / 10.0))
        except Exception:
            return 0.0

    def _simulate_rollout(self, root_game: Game, max_depth: int, my_color: Any) -> float:
        """Perform a random (or value-guided) rollout and return normalized reward."""
        try:
            game = self._copy_game(root_game)
            if game is None:
                return 0.0
            depth = 0
            while depth < max_depth and not self._is_terminal(game):
                actions = self._get_playable_actions(game)
                if not actions:
                    break
                # If we have a value function we can do a greedy pick with
                # some randomness (epsilon-greedy). Otherwise pick uniformly.
                if HAVE_VALUE_FN:
                    try:
                        # try scoring each action quickly by applying on a copy
                        best_a = None
                        best_v = -float('inf')
                        for a in actions:
                            try:
                                g2 = self._copy_game(game)
                                if g2 is None:
                                    continue
                                applied = self._apply_action(g2, a)
                                if not applied:
                                    continue
                                vfn = make_value_fn(g2)
                                try:
                                    v = vfn(g2, my_color)
                                except Exception:
                                    v = vfn(g2)
                                v = float(v)
                                if v > best_v:
                                    best_v = v
                                    best_a = a
                            except Exception:
                                continue
                        if best_a is None:
                            action = random.choice(actions)
                        else:
                            # epsilon-greedy: small chance to explore
                            if random.random() < 0.1:
                                action = random.choice(actions)
                            else:
                                action = best_a
                    except Exception:
                        action = random.choice(actions)
                else:
                    action = random.choice(actions)

                applied = self._apply_action(game, action)
                if not applied:
                    # If we couldn't apply the chosen action, break to avoid infinite loop
                    break
                depth += 1

            # Compute reward for my_color
            return self._reward_from_game(game, my_color)
        except Exception:
            if DEBUG:
                print('FooPlayer._simulate_rollout: exception during rollout', file=sys.stderr)
                traceback.print_exc()
            return 0.0

    def _run_mcts(self, game: Game, playable_actions: list, iterations: int, max_depth: int, my_color: Any):
        """Run a simple MCTS (UCT) search and return action statistics.

        Returns a dict mapping action -> (visits, total_value, avg_value).
        """
        root_game_copy = self._copy_game(game)
        if root_game_copy is None:
            raise RuntimeError('FooPlayer._run_mcts: failed to copy root game')

        root = FooPlayer._MCTSNode(root_game_copy)

        # Initialize root.untried_actions with a sample (progressive widening)
        root_actions = playable_actions
        if len(root_actions) > MAX_ACTIONS_TO_EVALUATE:
            try:
                sampled = random.sample(root_actions, MAX_ACTIONS_TO_EVALUATE)
            except Exception:
                sampled = root_actions[:MAX_ACTIONS_TO_EVALUATE]
            root.untried_actions = list(sampled)
        else:
            root.untried_actions = list(root_actions)

        for it in range(iterations):
            node = root
            # SELECTION & EXPANSION
            # Select until a node with untried actions or a leaf is reached
            while True:
                if node.untried_actions:
                    # Expand one action from untried_actions
                    try:
                        a = node.untried_actions.pop()
                    except Exception:
                        a = None
                    if a is None:
                        break
                    # Apply action on a copy of the node's game
                    gcopy = self._copy_game(node.game)
                    if gcopy is None:
                        break
                    applied = self._apply_action(gcopy, a)
                    if not applied:
                        # skip this action
                        continue
                    child = FooPlayer._MCTSNode(gcopy, parent=node, action=a)
                    # populate child's untried_actions lazily
                    try:
                        acts = self._get_playable_actions(gcopy)
                        if len(acts) > MAX_ACTIONS_TO_EVALUATE:
                            child.untried_actions = random.sample(acts, min(len(acts), MAX_ACTIONS_TO_EVALUATE))
                        else:
                            child.untried_actions = list(acts)
                    except Exception:
                        child.untried_actions = []
                    node.children.append(child)
                    node = child
                    break
                else:
                    # No untried actions: descend to best child
                    if not node.children:
                        break
                    # pick child with highest UCT
                    try:
                        node = max(node.children, key=lambda n: n.uct_score(UCT_C))
                    except Exception:
                        # fallback to visits
                        node = max(node.children, key=lambda n: n.visits)

            # SIMULATION from node.game
            reward = self._simulate_rollout(node.game, max_depth, my_color)

            # BACKPROPAGATION
            while node is not None:
                node.visits += 1
                node.total_value += reward
                node = node.parent

        # Aggregate stats for root's children
        stats = {}
        for child in root.children:
            try:
                visits = child.visits
                total = child.total_value
                avg = (total / visits) if visits > 0 else 0.0
                stats[child.action] = (visits, total, avg)
            except Exception:
                continue
        return stats

    # ------------------------ Decide (entry point) ------------------------
    def decide(self, game: Game, playable_actions):
        # Defensive: no actions
        if not playable_actions:
            if DEBUG:
                print('FooPlayer.decide: no playable_actions -> returning None')
            return None

        # Ensure list
        try:
            actions = list(playable_actions)
        except Exception:
            try:
                return playable_actions[0]
            except Exception:
                return None

        # Trivial case
        if len(actions) == 1:
            if DEBUG:
                print('FooPlayer.decide: only one action -> returning it')
            return actions[0]

        # Try running MCTS; if it fails, fall back to 1-ply static evaluation
        try:
            if DEBUG:
                print(f'FooPlayer.decide: starting MCTS with ITERATIONS={ITERATIONS}, MAX_DEPTH={MAX_DEPTH}')
            stats = self._run_mcts(game, actions, ITERATIONS, MAX_DEPTH, getattr(self, 'color', None))
            if not stats:
                raise RuntimeError('MCTS produced no child stats')

            # Choose action by highest visit count, tie-break by avg value
            best_action = None
            best_visits = -1
            best_avg = -float('inf')
            for a, (visits, total, avg) in stats.items():
                if visits > best_visits or (visits == best_visits and avg > best_avg):
                    best_action = a
                    best_visits = visits
                    best_avg = avg

            if best_action is None:
                raise RuntimeError('MCTS failed to select an action')

            if DEBUG:
                print(f'FooPlayer.decide: MCTS selected action {repr(best_action)} visits={best_visits} avg={best_avg}')

            return best_action

        except Exception as e:
            if DEBUG:
                print(f'FooPlayer.decide: MCTS failed with error: {e}; falling back to 1-ply eval', file=sys.stderr)
                traceback.print_exc()

            # Fall back: evaluate up to MAX_ACTIONS_TO_EVALUATE actions via static eval
            # (this code mirrors the previous implementation but is local here)
            if len(actions) > MAX_ACTIONS_TO_EVALUATE:
                try:
                    candidates = random.sample(actions, MAX_ACTIONS_TO_EVALUATE)
                except Exception:
                    candidates = actions[:MAX_ACTIONS_TO_EVALUATE]
                if DEBUG:
                    print(f'FooPlayer.decide: sampled {len(candidates)} of {len(actions)} actions to evaluate')
            else:
                candidates = actions
                if DEBUG:
                    print(f'FooPlayer.decide: evaluating all {len(candidates)} actions')

            scores = []
            for i, action in enumerate(candidates):
                try:
                    new_game = self._copy_game(game)
                    if new_game is None:
                        if DEBUG:
                            print(f'FooPlayer.decide: unable to copy game for action #{i}; marking -inf')
                        scores.append((action, float('-inf')))
                        continue

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
                        if DEBUG:
                            print(f'FooPlayer.decide: failed to execute candidate action #{i}; marking score -inf')
                        scores.append((action, float('-inf')))
                        continue

                    if HAVE_VALUE_FN:
                        try:
                            vfn = make_value_fn(new_game)
                            try:
                                val = vfn(new_game, getattr(self, 'color', None))
                            except Exception:
                                val = vfn(new_game)
                            score = float(val)
                            scores.append((action, score))
                            if DEBUG:
                                print(f'FooPlayer.decide: action #{i} -> value_fn score {score}')
                            continue
                        except Exception:
                            if DEBUG:
                                print(f'FooPlayer.decide: make_value_fn failed for action #{i}; falling back to static eval', file=sys.stderr)

                    score = self._evaluate_state(new_game)
                    scores.append((action, score))
                    if DEBUG:
                        print(f'FooPlayer.decide: action #{i} -> score {score}')

                except Exception as e2:
                    if DEBUG:
                        print(f'FooPlayer.decide: exception while evaluating action #{i}: {e2}! Marking -inf', file=sys.stderr)
                        traceback.print_exc()
                    scores.append((action, float('-inf')))

            if not scores:
                if DEBUG:
                    print('FooPlayer.decide: no scores produced -> defaulting to first action')
                return actions[0]

            try:
                max_score = max(score for (_, score) in scores)
            except Exception:
                max_score = float('-inf')

            best_candidates = [a for (a, s) in scores if s == max_score]
            if not best_candidates or max_score == float('-inf'):
                if DEBUG:
                    print('FooPlayer.decide: all evaluations failed -> defaulting to first action')
                return actions[0]

            chosen = random.choice(best_candidates)
            if DEBUG:
                try:
                    print(f'FooPlayer.decide: selected action -> {repr(chosen)} with score {max_score}')
                except Exception:
                    print('FooPlayer.decide: selected an action (repr failed)')
            return chosen

    # ------------------- Static evaluation (copied and hardened) -------------------
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
                # Prefer game.state.players but guard against different shapes
                try:
                    players = getattr(players, 'players', None) or getattr(game, 'players', None)
                except Exception:
                    players = getattr(game, 'players', None) or getattr(players, 'players', None)
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
