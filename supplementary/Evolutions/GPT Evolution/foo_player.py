import random
import traceback
from typing import Iterable, List, Optional, Tuple

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
)


class FooPlayer(Player):
    """A player that uses a phase-aware 1-ply lookahead with expected-value for chance actions.

    Strategy summary:
    - Phase-aware sampling/pruning of playable actions to keep runtime bounded.
    - For each sampled candidate:
        - Copy the game state (copy_game).
        - For chance-like actions (robber/dev-card): use execute_spectrum/expand_spectrum to compute expected value.
        - Otherwise execute deterministically (execute_deterministic).
        - Evaluate resulting states with the adapters base value function (base_fn()).
    - Select the action maximizing (score, vp_delta) with a deterministic tie-break on repr(action).

    Interactions with the engine are done through the adapters surface only.
    Debug printing is available by setting self.debug = True on the instance.
    """

    # Tunable class defaults (updated per STRATEGIZER recommendations)
    MAX_ACTIONS_TO_EVAL: int = 60
    SAMPLE_PER_ACTION_TYPE: int = 3
    SPECTRUM_MAX_OUTCOMES: int = 8
    EARLY_TURN_THRESHOLD: int = 30
    TOP_K_DEEP: int = 0  # reserved for future opponent-aware refinement (disabled by default)
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
            # Compare against ActionType enum values when possible
            return at in {
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD,
                # Some code-bases may expose upgrade as a separate type; include common names
            }
        except Exception:
            # Fallback to name-based detection
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
        # As a conservative fallback, check for an attribute `visible_victory_points` or similar
        try:
            vp_map = getattr(game, "visible_victory_points", None)
            if isinstance(vp_map, dict):
                return int(vp_map.get(my_color, 0))
        except Exception:
            pass
        # If nothing is available, return 0 — we avoid inventing game internals
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
        groups = {}
        for a in actions:
            key = self._action_type_key(a)
            groups.setdefault(key, []).append(a)

        # Deterministic RNG seeded with a combination of RNG_SEED and player's color
        color_seed = sum(ord(c) for c in str(self.color))
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
                # If any checks fail, fall back to default sample_count
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

    def _evaluate_action(self, game: Game, action, my_color: Color) -> Optional[Tuple[float, float]]:
        """Evaluate an action and return (score, vp_delta) or None on failure.

        - For robber/chance actions, attempt to use execute_spectrum/expand_spectrum to compute expected value.
        - Otherwise run execute_deterministic and score the single resulting state.

        Any exception during evaluation for a specific action results in None so other actions
        can still be considered.
        """
        # 1) copy the game state
        try:
            game_copy = copy_game(game)
        except Exception as e:
            if self.debug:
                print("copy_game failed:", e)
                traceback.print_exc()
            return None

        # Ensure we have a value function callable
        if self._value_fn is None:
            try:
                self._value_fn = base_fn()
            except Exception as e:
                if self.debug:
                    print("base_fn() factory failed during evaluate_action:", e)
                    traceback.print_exc()
                return None

        # Helper to safely compute numeric score from value function
        def score_for(g: Game) -> Optional[float]:
            try:
                s = self._value_fn(g, my_color)
                return float(s)
            except Exception:
                if self.debug:
                    print("value function failed on game state for action", repr(action))
                    traceback.print_exc()
                return None

        # If this is a robber/chance-like action, try to compute expected value
        if self._is_robber_or_chance(action):
            try:
                # Prefer execute_spectrum if available
                spectrum = None
                try:
                    spectrum = execute_spectrum(game_copy, action)
                except Exception:
                    # Try expand_spectrum with a single-action list and extract
                    try:
                        spec_map = expand_spectrum(game_copy, [action])
                        if isinstance(spec_map, dict):
                            spectrum = spec_map.get(action, [])
                    except Exception:
                        spectrum = None

                if spectrum:
                    # Cap outcomes for runtime
                    spectrum_list = list(spectrum)[: self.SPECTRUM_MAX_OUTCOMES]
                    weighted_score = 0.0
                    weighted_vp_delta = 0.0
                    base_vp = self._get_visible_vp(game, my_color)
                    for entry in spectrum_list:
                        # entry expected to be (game_state, prob) but be defensive
                        try:
                            outcome_game, prob = entry
                        except Exception:
                            # Unexpected shape; skip this outcome
                            continue
                        sc = score_for(outcome_game)
                        if sc is None:
                            # If any outcome cannot be scored, abort spectrum evaluation
                            weighted_score = None
                            break
                        weighted_score += prob * sc
                        vp_after = self._get_visible_vp(outcome_game, my_color)
                        weighted_vp_delta += prob * (vp_after - base_vp)

                    if weighted_score is None:
                        # Fall back to deterministic evaluation below
                        if self.debug:
                            print("Spectrum evaluation produced an unscorable outcome; falling back to deterministic for", repr(action))
                    else:
                        if self.debug:
                            print(
                                f"Spectrum eval for {repr(action)}: expected_score={weighted_score}, expected_vp_delta={weighted_vp_delta}, outcomes={len(spectrum_list)}"
                            )
                        return (float(weighted_score), float(weighted_vp_delta))
            except Exception as e:
                if self.debug:
                    print("execute_spectrum/expand_spectrum failed for action", repr(action), "error:", e)
                    traceback.print_exc()
                # Fall through to deterministic handling

        # Default deterministic evaluation
        try:
            outcomes = execute_deterministic(game_copy, action)
        except Exception as e:
            if self.debug:
                print("execute_deterministic failed for action:", repr(action), "error:", e)
                traceback.print_exc()
            return None

        # Normalize to a single resulting game state (pick the first outcome deterministically)
        try:
            if not outcomes:
                if self.debug:
                    print("execute_deterministic returned empty outcomes for", repr(action))
                return None
            first = outcomes[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                resultant_game = first[0]
            else:
                resultant_game = first
        except Exception:
            # As a last resort, use the mutated game_copy
            resultant_game = game_copy

        # Score and vp delta
        sc = score_for(resultant_game)
        if sc is None:
            return None
        try:
            base_vp = self._get_visible_vp(game, my_color)
            after_vp = self._get_visible_vp(resultant_game, my_color)
            vp_delta = float(after_vp - base_vp)
        except Exception:
            vp_delta = 0.0

        return (float(sc), float(vp_delta))

    # ------------------ Decision method (public) ------------------
    def decide(self, game: Game, playable_actions: Iterable):
        """Choose an action from playable_actions using the refined 1-ply lookahead.

        The selection prioritizes (score, vp_delta) and breaks ties deterministically by
        lexicographic repr(action).
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

        # Sample/prune with phase awareness
        candidates = self._sample_actions(actions, game)

        if self.debug:
            print(f"decide: evaluating {len(candidates)} candidate(s) out of {len(actions)} playable action(s)")

        best_action = None
        best_score = -float("inf")
        best_vp_delta = -float("inf")
        best_tie_repr = None

        evaluated = 0
        for action in candidates:
            evaluated += 1
            eval_res = self._evaluate_action(game, action, self.color)
            if self.debug:
                print(f"Evaluated action [{evaluated}/{len(candidates)}]: {repr(action)} -> {eval_res}")

            if eval_res is None:
                continue
            score, vp_delta = eval_res

            tie_repr = repr(action)
            # Compare by (score, vp_delta, -repr) where repr smaller is preferred deterministically
            is_better = False
            if best_action is None:
                is_better = True
            elif score > best_score:
                is_better = True
            elif score == best_score:
                if vp_delta > best_vp_delta:
                    is_better = True
                elif vp_delta == best_vp_delta:
                    if best_tie_repr is None or tie_repr < best_tie_repr:
                        is_better = True

            if is_better:
                best_action = action
                best_score = score
                best_vp_delta = vp_delta
                best_tie_repr = tie_repr

            # Optional budget guard: stop early if we've evaluated MAX_ACTIONS_TO_EVAL candidates
            if evaluated >= self.MAX_ACTIONS_TO_EVAL:
                if self.debug:
                    print("decide: reached evaluation budget; stopping early")
                break

        if best_action is None:
            if self.debug:
                print("decide: no evaluated candidate succeeded; falling back to first playable action")
            return actions[0]

        if self.debug:
            print("decide: chosen action:", repr(best_action), "score:", best_score, "vp_delta:", best_vp_delta)

        return best_action
