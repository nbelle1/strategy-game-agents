import random
import traceback
from typing import Iterable, List, Optional, Tuple, Dict, Any

# Must import adapters via the provided thin wrapper. Do NOT import catanatron directly.
from .adapters import (
    Game,
    Player,
    Color,
    ActionType,
    copy_game,
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    base_fn,
    list_prunned_actions,
)


class FooPlayer(Player):
    """A player that uses a selective 2-ply adversarial lookahead built on adapters.

    Key features:
    - Phase-aware 1-ply sampling/pruning to keep runtime bounded.
    - Chance-aware expected values using execute_spectrum/expand_spectrum when available.
    - Selective 2-ply adversarial (min over opponent responses) evaluation for top-K 1-ply
      candidates to improve robustness against counters.
    - Deterministic sampling/tie-breaking via seeded RNG.

    All interactions with the engine use only the adapters surface.
    Set self.debug = True on the instance to enable diagnostic printing.
    """

    # Tunable class defaults (STRATEGIZER recommendations)
    MAX_ACTIONS_TO_EVAL: int = 60
    SAMPLE_PER_ACTION_TYPE: int = 3
    SPECTRUM_MAX_OUTCOMES: int = 8
    EARLY_TURN_THRESHOLD: int = 30

    # New selective 2-ply parameters
    TOP_K_1PLY: int = 12
    OP_MAX_ACTIONS: int = 20
    OP_SAMPLE_PER_ACTION_TYPE: int = 2

    # reserved/compat
    TOP_K_DEEP: int = 0  # disabled by default
    RNG_SEED: int = 0

    def __init__(self, name: Optional[str] = None):
        # Initialize as BLUE by default (preserve original behavior)
        super().__init__(Color.BLUE, name)
        # Toggle to True to get per-turn diagnostic prints
        self.debug: bool = False
        # Pre-create the value function from adapters.base_fn factory if possible.
        # base_fn returns a callable: (game, color) -> float.
        try:
            self._value_fn = base_fn()
        except Exception:
            # If the factory has a different signature, lazily resolve in evaluation.
            self._value_fn = None

    # ------------------ Helper methods ------------------
    def _stable_color_hash(self, color: Color) -> int:
        """Stable small hash for a Color used to seed RNG deterministically.

        We keep this deterministic across runs by summing character ordinals of the color's
        string representation. This avoids relying on Python's randomized hash().
        """
        try:
            return sum(ord(c) for c in str(color)) & 0xFFFFFFFF
        except Exception:
            return 0

    def _action_type_key(self, action) -> str:
        """Return a stable grouping key for an action.

        Prefer action.action_type, then other attributes, then class name or string.
        """
        k = getattr(action, "action_type", None)
        if k is not None:
            return str(k)
        for attr in ("type", "name"):
            k = getattr(action, attr, None)
            if k is not None:
                return str(k)
        try:
            return action.__class__.__name__
        except Exception:
            return str(action)

    def _is_build_or_upgrade(self, action) -> bool:
        """Detect actions that build or upgrade (settlement, city, road, upgrade).

        This function is defensive: it checks action_type when available and falls back
        to class name matching so grouping remains robust.
        """
        at = getattr(action, "action_type", None)
        try:
            return at in {
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD,
            }
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            name_str = str(name).lower()
            return any(k in name_str for k in ("build", "settle", "city", "road", "upgrade"))

    def _is_robber_or_chance(self, action) -> bool:
        """Detect robber placement or development-card (chance) actions.

        Uses action_type when available; otherwise checks common name tokens.
        """
        at = getattr(action, "action_type", None)
        try:
            return at in {
                ActionType.PLAY_DEV_CARD,
                ActionType.PLACE_ROBBER,
                ActionType.DRAW_DEV_CARD,
            }
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            name_str = str(name).lower()
            return any(k in name_str for k in ("robber", "dev", "development", "draw"))

    def _get_visible_vp(self, game: Game, my_color: Color) -> int:
        """Try to extract a visible/observable victory point count for my_color.

        This is intentionally defensive: if no visible metric exists, return 0.
        """
        try:
            vp_map = getattr(game, "visible_vp", None)
            if isinstance(vp_map, dict):
                return int(vp_map.get(my_color, 0))
        except Exception:
            pass
        try:
            vp_map = getattr(game, "visible_victory_points", None)
            if isinstance(vp_map, dict):
                return int(vp_map.get(my_color, 0))
        except Exception:
            pass
        return 0

    def _sample_actions(self, playable_actions: Iterable, game: Game) -> List:
        """Phase-aware sampling: prioritize builds early, VP actions late.

        Returns a deterministic, pruned list of candidate actions up to MAX_ACTIONS_TO_EVAL.
        """
        actions = list(playable_actions)
        n = len(actions)
        if n <= self.MAX_ACTIONS_TO_EVAL:
            return actions

        # Determine phase using available heuristics on game. Use tick or current_turn if present.
        current_turn = getattr(game, "current_turn", None)
        if current_turn is None:
            current_turn = getattr(game, "tick", 0)
        early_game = (current_turn <= self.EARLY_TURN_THRESHOLD)

        # Group actions by stable key
        groups: Dict[str, List] = {}
        for a in actions:
            key = self._action_type_key(a)
            groups.setdefault(key, []).append(a)

        # Deterministic RNG seeded with a combination of RNG_SEED and player's color
        color_seed = self._stable_color_hash(self.color)
        rng = random.Random(self.RNG_SEED + color_seed)

        sampled: List = []
        # Iterate through groups in a stable order to keep behavior deterministic
        for key in sorted(groups.keys()):
            group = list(groups[key])
            # Determine how many to sample from this group, with phase-aware bias
            sample_count = self.SAMPLE_PER_ACTION_TYPE
            try:
                if early_game and any(self._is_build_or_upgrade(a) for a in group):
                    sample_count += 1
                elif not early_game and any(
                    getattr(a, "action_type", None) in {ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT}
                    for a in group
                ):
                    sample_count += 1
            except Exception:
                pass

            # Deterministic shuffle and pick
            rng.shuffle(group)
            take = min(sample_count, len(group))
            sampled.extend(group[:take])
            if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                break

        # If under budget, fill deterministically from remaining actions
        if len(sampled) < self.MAX_ACTIONS_TO_EVAL:
            for a in actions:
                if a not in sampled:
                    sampled.append(a)
                    if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                        break

        if self.debug:
            phase = "early" if early_game else "late"
            print(f"_sample_actions: phase={phase}, pruned {n} -> {len(sampled)} actions (cap={self.MAX_ACTIONS_TO_EVAL})")
        return sampled

    def _sample_opponent_actions(self, playable_actions: Iterable, game: Game, opponent_color: Color) -> List:
        """Opponent-specific sampling that respects OP_SAMPLE_PER_ACTION_TYPE and OP_MAX_ACTIONS.

        Uses a deterministic RNG seeded with opponent color so opponent sampling is reproducible.
        """
        actions = list(playable_actions)
        n = len(actions)
        if n <= self.OP_MAX_ACTIONS:
            return actions

        # Phase detection reused from our own sampling
        current_turn = getattr(game, "current_turn", None)
        if current_turn is None:
            current_turn = getattr(game, "tick", 0)
        early_game = (current_turn <= self.EARLY_TURN_THRESHOLD)

        groups: Dict[str, List] = {}
        for a in actions:
            key = self._action_type_key(a)
            groups.setdefault(key, []).append(a)

        color_seed = self._stable_color_hash(opponent_color)
        rng = random.Random(self.RNG_SEED + color_seed)

        sampled: List = []
        for key in sorted(groups.keys()):
            group = list(groups[key])
            # opponent sampling budget
            sample_count = self.OP_SAMPLE_PER_ACTION_TYPE
            try:
                if early_game and any(self._is_build_or_upgrade(a) for a in group):
                    sample_count += 1
            except Exception:
                pass
            rng.shuffle(group)
            take = min(sample_count, len(group))
            sampled.extend(group[:take])
            if len(sampled) >= self.OP_MAX_ACTIONS:
                break

        if len(sampled) < self.OP_MAX_ACTIONS:
            for a in actions:
                if a not in sampled:
                    sampled.append(a)
                    if len(sampled) >= self.OP_MAX_ACTIONS:
                        break

        if self.debug:
            print(f"_sample_opponent_actions: pruned {n} -> {len(sampled)} actions (cap={self.OP_MAX_ACTIONS})")
        return sampled

    def _normalize_and_cap_spectrum(self, spectrum: Iterable, cap: int) -> List[Tuple[Game, float]]:
        """Normalize spectrum outcomes and cap to `cap` entries.

        Accepts iterables like those returned by execute_spectrum or expand_spectrum entry lists.
        Returns a list of (game, prob) with probabilities summing to 1.
        """
        try:
            lst = list(spectrum)
            if not lst:
                return []
            capped = lst[:cap]
            probs = []
            games = []
            for entry in capped:
                try:
                    g, p = entry
                except Exception:
                    # Unexpected shape: skip
                    continue
                games.append(g)
                probs.append(float(p))
            if not games:
                return []
            total = sum(probs)
            if total > 0.0:
                normalized = [(g, p / total) for g, p in zip(games, probs)]
            else:
                n = len(games)
                normalized = [(g, 1.0 / n) for g in games]
            return normalized
        except Exception:
            if self.debug:
                print("_normalize_and_cap_spectrum: failed to normalize spectrum")
                traceback.print_exc()
            return []

    def _determine_opponent_color(self, game: Game, my_color: Color) -> Color:
        """Try to determine the opponent's color from the game state.

        This is defensive: it checks common attributes and falls back to a two-player assumption.
        """
        try:
            cur = getattr(game, "current_player", None)
            if cur is not None:
                # If cur is a Player instance, extract its color attribute when possible
                try:
                    if cur != my_color:
                        return cur
                except Exception:
                    pass
        except Exception:
            pass

        # As a simple fallback, assume a two-player game and pick a different color deterministically
        try:
            colors = [c for c in list(Color)]
            if len(colors) >= 2:
                for c in colors:
                    if c != my_color:
                        return c
        except Exception:
            pass
        # Last resort: return my_color (harmless, though less correct)
        return my_color

    def _derive_opponent_actions(self, game: Game, opponent_color: Color) -> List:
        """Obtain a list of opponent actions with several fallbacks.

        Order:
        1) adapters.list_prunned_actions(game)
        2) game.playable_actions() if present
        3) empty list (conservative)
        """
        try:
            # Preferred: adapters-provided pruned action list (designed for search)
            pruned = list_prunned_actions(game)
            if pruned:
                return pruned
        except Exception:
            if self.debug:
                print("_derive_opponent_actions: list_prunned_actions failed")
                traceback.print_exc()

        try:
            pa = getattr(game, "playable_actions", None)
            if callable(pa):
                res = pa()
                if res:
                    return list(res)
        except Exception:
            if self.debug:
                print("_derive_opponent_actions: game.playable_actions() failed")
                traceback.print_exc()

        # As a conservative fallback, return empty list so we evaluate the post-action state directly
        return []

    def _safe_eval_base_fn(self, g: Game, color: Color) -> Optional[float]:
        """Safely call the adapters' base value function in its possible forms.

        Tries self._value_fn(g,color) if available; otherwise attempts base_fn()(g,color) and
        finally base_fn(g,color). Returns None on failure and logs when debug=True.
        """
        try:
            if self._value_fn is not None:
                return float(self._value_fn(g, color))
        except Exception:
            if self.debug:
                print("_safe_eval_base_fn: self._value_fn failed")
                traceback.print_exc()
        # Try factory form
        try:
            vf = base_fn()
            try:
                return float(vf(g, color))
            except Exception:
                if self.debug:
                    print("_safe_eval_base_fn: vf(g,color) failed")
                    traceback.print_exc()
        except Exception:
            # Maybe base_fn itself accepts (g,color)
            pass
        try:
            return float(base_fn(g, color))
        except Exception:
            if self.debug:
                print("_safe_eval_base_fn: all attempts to call base_fn failed")
                traceback.print_exc()
            return None

    def _simulate_and_evaluate(self, game: Game, action, my_color: Color) -> Optional[float]:
        """Simulate `action` from `game` and return a numeric expected score for my_color.

        If action is None, simply evaluate the provided game state.
        This function handles spectrum (chance) outcomes when available and falls back to
        deterministic execution. Returns None on failure for the given simulation.
        """
        # Copy the game to avoid mutating caller's state
        try:
            game_copy = copy_game(game)
        except Exception as e:
            if self.debug:
                print("_simulate_and_evaluate: copy_game failed:", e)
                traceback.print_exc()
            return None

        # If action is None, just evaluate the provided state
        if action is None:
            return self._safe_eval_base_fn(game_copy, my_color)

        # Chance-aware path
        if self._is_robber_or_chance(action):
            try:
                spec = None
                try:
                    spec = execute_spectrum(game_copy, action)
                except Exception:
                    # Try expand_spectrum single-action expansion
                    try:
                        spec_map = expand_spectrum(game_copy, [action])
                        if isinstance(spec_map, dict):
                            spec = spec_map.get(action, None)
                    except Exception:
                        spec = None

                if spec:
                    outcomes = self._normalize_and_cap_spectrum(spec, self.SPECTRUM_MAX_OUTCOMES)
                    if not outcomes:
                        # Fall through to deterministic
                        pass
                    else:
                        total_score = 0.0
                        for og, prob in outcomes:
                            sc = self._safe_eval_base_fn(og, my_color)
                            if sc is None:
                                # If any outcome can't be evaluated reliably, abort spectrum path
                                total_score = None
                                break
                            total_score += prob * sc
                        if total_score is None:
                            if self.debug:
                                print("_simulate_and_evaluate: spectrum had unscorable outcomes; falling back")
                        else:
                            return float(total_score)
            except Exception as e:
                if self.debug:
                    print("_simulate_and_evaluate: execute_spectrum/expand_spectrum failed:", e)
                    traceback.print_exc()
                # fall through to deterministic

        # Deterministic fallback
        try:
            outcomes = execute_deterministic(game_copy, action)
        except Exception as e:
            if self.debug:
                print("_simulate_and_evaluate: execute_deterministic failed:", e)
                traceback.print_exc()
            return None

        try:
            if not outcomes:
                if self.debug:
                    print("_simulate_and_evaluate: execute_deterministic returned no outcomes")
                return None
            first = outcomes[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                resultant_game = first[0]
            else:
                resultant_game = first
        except Exception:
            resultant_game = game_copy

        return self._safe_eval_base_fn(resultant_game, my_color)

    # ------------------ NEW missing method: _evaluate_action ------------------
    def _evaluate_action(self, game: Game, action, my_color: Color) -> Optional[Tuple[float, float]]:
        """Evaluate a candidate action and return (score, vp_delta) or None on failure.

        This method unifies spectrum-based chance evaluation and deterministic execution
        and returns both the numeric score (from base_fn) and the visible VP delta (after - before).
        It is defensive to adapter signature differences and logs traces when self.debug is True.
        """
        # Helper: safe eval using existing wrapper
        def safe_eval(g: Game) -> Optional[float]:
            return self._safe_eval_base_fn(g, my_color)

        # Helper: visible vp extraction (use existing helper)
        def get_vp(g: Game) -> float:
            try:
                return float(self._get_visible_vp(g, my_color))
            except Exception:
                if self.debug:
                    print("_evaluate_action: _get_visible_vp failed")
                    traceback.print_exc()
                return 0.0

        # Step A: copy game
        try:
            game_copy = copy_game(game)
        except Exception:
            if self.debug:
                print("_evaluate_action: copy_game failed:")
                traceback.print_exc()
            return None

        # original visible vp
        try:
            vp_orig = get_vp(game)
        except Exception:
            vp_orig = 0.0

        # Step B: if chance-like, try spectrum expansion
        if self._is_robber_or_chance(action):
            try:
                spec = None
                try:
                    spec = execute_spectrum(game_copy, action)
                except Exception:
                    try:
                        spec_map = expand_spectrum(game_copy, [action])
                        if isinstance(spec_map, dict):
                            spec = spec_map.get(action, None)
                    except Exception:
                        spec = None

                if spec:
                    outcomes = self._normalize_and_cap_spectrum(spec, self.SPECTRUM_MAX_OUTCOMES)
                    if outcomes:
                        weighted_score = 0.0
                        weighted_vp_delta = 0.0
                        any_scored = False
                        for og, prob in outcomes:
                            sc = safe_eval(og)
                            if sc is None:
                                # skip unscorable outcomes
                                continue
                            any_scored = True
                            vp_out = get_vp(og)
                            weighted_score += prob * sc
                            weighted_vp_delta += prob * (vp_out - vp_orig)
                        if any_scored:
                            return (float(weighted_score), float(weighted_vp_delta))
                        # else fall through to deterministic
            except Exception:
                if self.debug:
                    print("_evaluate_action: spectrum evaluation failed:")
                    traceback.print_exc()
                # fall through

        # Step C: deterministic execution fallback
        try:
            res = execute_deterministic(game_copy, action)
        except Exception:
            if self.debug:
                print("_evaluate_action: execute_deterministic failed:")
                traceback.print_exc()
            return None

        try:
            # normalize to a single resultant game
            resultant_game = None
            if res is None:
                resultant_game = game_copy
            elif isinstance(res, (list, tuple)):
                first = res[0]
                if isinstance(first, tuple) and len(first) >= 1:
                    resultant_game = first[0]
                else:
                    resultant_game = first
            else:
                # could be a single game object
                resultant_game = res if hasattr(res, "state") or hasattr(res, "current_player") else game_copy

            score = safe_eval(resultant_game)
            if score is None:
                return None
            vp_after = get_vp(resultant_game)
            vp_delta = float(vp_after - vp_orig)
            return (float(score), float(vp_delta))
        except Exception:
            if self.debug:
                print("_evaluate_action: normalize/eval failed:")
                traceback.print_exc()
            return None

    # ------------------ Decision method (public) ------------------
    def decide(self, game: Game, playable_actions: Iterable):
        """Choose an action using selective 2-ply adversarial lookahead.

        Flow:
        1) Run phase-aware 1-ply sampling and evaluation across candidates.
        2) Keep top TOP_K_1PLY candidates by 1-ply score and deepen each with opponent modeling.
        3) For each candidate, compute expected adversarial value = E_outcomes[min_opponent_response(score)].
        4) Pick candidate maximizing (expected_value, 1-ply vp_delta, repr action tie-break).

        All adapter calls are protected with try/except. On catastrophic failure we fall back to
        returning the best 1-ply candidate or the first playable action as a last resort.
        """
        actions = list(playable_actions)

        if not actions:
            if self.debug:
                print("decide: no playable_actions provided")
            return None

        if len(actions) == 1:
            if self.debug:
                print("decide: single playable action, returning it")
            return actions[0]

        # Stage 1: 1-ply evaluation
        candidates = self._sample_actions(actions, game)
        if self.debug:
            print(f"decide: sampled {len(candidates)} candidates from {len(actions)} actions")

        one_ply_results: List[Tuple[Any, float, float]] = []  # (action, score, vp_delta)
        for idx, a in enumerate(candidates, start=1):
            res = self._evaluate_action(game, a, self.color)
            if self.debug:
                print(f"1-ply [{idx}/{len(candidates)}]: {repr(a)} -> {res}")
            if res is None:
                continue
            sc, vpd = res
            one_ply_results.append((a, float(sc), float(vpd)))

        if not one_ply_results:
            # Nothing evaluated successfully; fallback deterministically
            if self.debug:
                print("decide: no 1-ply evaluations succeeded; falling back to first playable action")
            return actions[0]

        # Stage 2: pick top-K 1-ply candidates
        one_ply_results.sort(key=lambda t: (t[1], t[2]), reverse=True)
        top_k = [t[0] for t in one_ply_results[: self.TOP_K_1PLY]]

        if self.debug:
            print("Top 1-ply candidates:")
            for a, s, v in one_ply_results[: self.TOP_K_1PLY]:
                print(f"  candidate: {repr(a)} score={s} vp_delta={v}")

        # Stage 3: selective 2-ply adversarial evaluation
        best_action = None
        best_value = -float("inf")
        best_vp_delta = -float("inf")
        best_repr = None

        # Simulation budget guard: do not exceed too many simulated nodes
        sim_count = 0
        SIMULATION_HARD_LIMIT = 10000  # safety cap to protect time; adjustable

        try:
            for a in top_k:
                if sim_count >= SIMULATION_HARD_LIMIT:
                    if self.debug:
                        print("decide: reached simulation hard limit; stopping deepening")
                    break

                # Simulate our action a to produce outcome branches
                try:
                    game_copy = copy_game(game)
                except Exception as e:
                    if self.debug:
                        print("decide: copy_game failed for candidate", repr(a), e)
                        traceback.print_exc()
                    continue

                # Obtain outcome branches: prefer spectrum for chance actions
                outcomes: List[Tuple[Game, float]] = []
                try:
                    if self._is_robber_or_chance(a):
                        spec = None
                        try:
                            spec = execute_spectrum(game_copy, a)
                        except Exception:
                            try:
                                spec_map = expand_spectrum(game_copy, [a])
                                if isinstance(spec_map, dict):
                                    spec = spec_map.get(a, None)
                            except Exception:
                                spec = None

                        if spec:
                            outcomes = self._normalize_and_cap_spectrum(spec, self.SPECTRUM_MAX_OUTCOMES)
                    # Fallback to deterministic
                    if not outcomes:
                        det = execute_deterministic(game_copy, a)
                        if not det:
                            if self.debug:
                                print("decide: execute_deterministic returned empty for", repr(a))
                            continue
                        # det is list of (game, prob) often; take as provided
                        # normalize shape defensively
                        normalized = []
                        for entry in det[: self.SPECTRUM_MAX_OUTCOMES]:
                            try:
                                g, p = entry
                            except Exception:
                                g = entry
                                p = 1.0
                            normalized.append((g, float(p)))
                        # If probabilities not summing to 1, normalize
                        total_p = sum(p for _, p in normalized)
                        if total_p <= 0:
                            # assign uniform
                            n = len(normalized)
                            outcomes = [(g, 1.0 / n) for (g, _) in normalized]
                        else:
                            outcomes = [(g, p / total_p) for (g, p) in normalized]

                except Exception as e:
                    if self.debug:
                        print("decide: failed to obtain outcomes for candidate", repr(a), "error:", e)
                        traceback.print_exc()
                    continue

                # Cap outcomes just in case
                if len(outcomes) > self.SPECTRUM_MAX_OUTCOMES:
                    outcomes = outcomes[: self.SPECTRUM_MAX_OUTCOMES]

                if self.debug:
                    print(f"Candidate {repr(a)} produced {len(outcomes)} outcome(s) to evaluate")

                expected_value_a = 0.0
                # find 1-ply vp delta for tie-break usage
                one_ply_vp_delta = next((v for (act, s, v) in one_ply_results if act == a), 0.0)

                for og, p_i in outcomes:
                    if sim_count >= SIMULATION_HARD_LIMIT:
                        break
                    # Derive opponent color
                    opp_color = self._determine_opponent_color(og, self.color)
                    # Get opponent actions with robust fallbacks
                    try:
                        opp_actions = self._derive_opponent_actions(og, opp_color)
                    except Exception:
                        opp_actions = []

                    if not opp_actions:
                        # No opponent actions: evaluate the post-my-action state directly
                        val_i = self._simulate_and_evaluate(og, None, self.color)
                        if val_i is None:
                            # Skip this outcome in expectation if it couldn't be evaluated
                            continue
                        expected_value_a += p_i * val_i
                        sim_count += 1
                        continue

                    # Prune opponent actions deterministically and cap
                    opp_sampled = self._sample_opponent_actions(opp_actions, og, opp_color)[: self.OP_MAX_ACTIONS]

                    if self.debug:
                        print(f"  outcome p={p_i:.3f}: opp_actions={len(opp_actions)} -> sampled={len(opp_sampled)}")

                    # Adversarial opponent: they choose the action minimizing our final score
                    min_score_after_opp = float("inf")
                    for b in opp_sampled:
                        if sim_count >= SIMULATION_HARD_LIMIT:
                            break
                        val_after_b = self._simulate_and_evaluate(og, b, self.color)
                        sim_count += 1
                        if val_after_b is None:
                            continue
                        if val_after_b < min_score_after_opp:
                            min_score_after_opp = val_after_b

                    if min_score_after_opp == float("inf"):
                        # If no opponent simulation succeeded, evaluate the post-my-action state
                        min_score_after_opp = self._simulate_and_evaluate(og, None, self.color) or 0.0

                    expected_value_a += p_i * min_score_after_opp

                # Compare candidate expected value using deterministic tie-break
                if self.debug:
                    print(f"Candidate {repr(a)} expected_value={expected_value_a} (1-ply vp_delta={one_ply_vp_delta})")

                is_better = False
                if best_action is None:
                    is_better = True
                elif expected_value_a > best_value:
                    is_better = True
                elif expected_value_a == best_value:
                    if one_ply_vp_delta > best_vp_delta:
                        is_better = True
                    elif one_ply_vp_delta == best_vp_delta:
                        tie_repr = repr(a)
                        if best_repr is None or tie_repr < best_repr:
                            is_better = True

                if is_better:
                    best_action = a
                    best_value = expected_value_a
                    best_vp_delta = one_ply_vp_delta
                    best_repr = repr(a)

            # End loop over top_k
        except Exception:
            if self.debug:
                print("decide: unexpected error during 2-ply deepening")
                traceback.print_exc()
            # Fall back to 1-ply selection below

        # If 2-ply produced a valid selection, return it
        if best_action is not None:
            if self.debug:
                print("decide: selected (2-ply) action:", repr(best_action), "value:", best_value)
            return best_action

        # Otherwise, fall back to best 1-ply action using existing tie-break rules
        if self.debug:
            print("decide: falling back to best 1-ply action")
        best_action_1ply = None
        best_score = -float("inf")
        best_vp = -float("inf")
        best_repr = None
        for (a, s, v) in one_ply_results:
            tie_repr = repr(a)
            is_better = False
            if best_action_1ply is None:
                is_better = True
            elif s > best_score:
                is_better = True
            elif s == best_score:
                if v > best_vp:
                    is_better = True
                elif v == best_vp and (best_repr is None or tie_repr < best_repr):
                    is_better = True
            if is_better:
                best_action_1ply = a
                best_score = s
                best_vp = v
                best_repr = tie_repr

        if best_action_1ply is not None:
            if self.debug:
                print("decide: chosen action (1-ply fallback):", repr(best_action_1ply), "score:", best_score, "vp_delta:", best_vp)
            return best_action_1ply

        # Last resort: return first playable action
        if self.debug:
            print("decide: no choice after fallbacks; returning first playable action")
        return actions[0]
