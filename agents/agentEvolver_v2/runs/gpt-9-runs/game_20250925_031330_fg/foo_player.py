import random
import traceback
from typing import Iterable, List, Optional

# Must import adapters via the provided thin wrapper. Do NOT import catanatron directly.
from .adapters import (
    Game,
    Player,
    Color,
    copy_game,
    execute_deterministic,
    base_fn,
)


class FooPlayer(Player):
    """A simple player that uses a 1-ply deterministic lookahead via adapters.

    Strategy summary:
    - Sample/prune the playable actions if there are too many (to bound runtime).
    - For each candidate action:
        - Make a deep copy of the game state (copy_game).
        - Execute the action deterministically (execute_deterministic).
        - Evaluate the resulting state with the base value function (base_fn()).
    - Choose the action with the highest evaluation score. Tie-break deterministically.

    Notes:
    - All interactions with the engine are done through the adapters surface.
    - Debug printing is available by setting self.debug = True on the instance.
    """

    # Tunable class defaults
    MAX_ACTIONS_TO_EVAL: int = 30
    SAMPLE_PER_ACTION_TYPE: int = 2
    RNG_SEED: int = 0

    def __init__(self, name: Optional[str] = None):
        # Initialize as BLUE by default (preserve original behavior)
        super().__init__(Color.BLUE, name)
        # Toggle to True to get per-turn diagnostic prints (keeps test runs quieter by default)
        self.debug: bool = False
        # Pre-create the value function from adapters.base_fn factory.
        # base_fn returns a callable: (game, color) -> float.
        try:
            self._value_fn = base_fn()
        except Exception:
            # If the factory has a different signature, lazily resolve in evaluation.
            self._value_fn = None

    # ------------------ Helper methods ------------------
    def _action_type_key(self, action) -> str:
        """Return a stable grouping key for an action.

        Prefer action.action_type (present on Action namedtuples), then fall back to
        class name or string representation. This keeps grouping robust across action
        shapes.
        """
        # Common attribute on actions in this environment
        k = getattr(action, "action_type", None)
        if k is not None:
            return str(k)
        # Try other possible names
        for attr in ("type", "name"):
            k = getattr(action, attr, None)
            if k is not None:
                return str(k)
        # Fall back to class name or string form
        try:
            return action.__class__.__name__
        except Exception:
            return str(action)

    def _sample_actions(self, playable_actions: Iterable) -> List:
        """Return a pruned list of candidate actions to evaluate.

        - If the number of actions is below MAX_ACTIONS_TO_EVAL, return them all.
        - Otherwise group actions by type and take up to SAMPLE_PER_ACTION_TYPE from each
          group (deterministic sampling using a seeded RNG). If still under the cap,
          fill the remainder deterministically.
        """
        actions = list(playable_actions)
        n = len(actions)
        if n <= self.MAX_ACTIONS_TO_EVAL:
            return actions

        # Group actions
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
            rng.shuffle(group)
            take = min(self.SAMPLE_PER_ACTION_TYPE, len(group))
            sampled.extend(group[:take])
            if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                break

        # If we haven't reached the cap, append remaining actions deterministically
        if len(sampled) < self.MAX_ACTIONS_TO_EVAL:
            for a in actions:
                if a not in sampled:
                    sampled.append(a)
                    if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                        break

        if self.debug:
            print(f"_sample_actions: pruned {n} -> {len(sampled)} actions")
        return sampled

    def _evaluate_action(self, game: Game, action, my_color: Color) -> Optional[float]:
        """Evaluate a single action by applying it to a copied game state and scoring it.

        Returns:
            float score if successful, otherwise None.
        """
        # 1) copy the game state
        try:
            game_copy = copy_game(game)
        except Exception as e:
            if self.debug:
                print("copy_game failed:", e)
                traceback.print_exc()
            return None

        # 2) execute the action deterministically; adapters.execute_deterministic returns
        #    a list of (game_copy, probability) tuples according to adapters.py docstring.
        try:
            outcomes = execute_deterministic(game_copy, action)
        except Exception as e:
            if self.debug:
                print("execute_deterministic failed for action:", repr(action), "error:", e)
                traceback.print_exc()
            return None

        # Normalize to a single resulting game state deterministically: pick the first outcome.
        try:
            if not outcomes:
                # Nothing returned => treat as failure
                if self.debug:
                    print("execute_deterministic returned empty outcomes for", repr(action))
                return None
            # outcomes is expected to be List[Tuple[Game, float]]
            first = outcomes[0]
            # If tuple-like, take first element
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                resultant_game = first[0]
            else:
                # If the adapter returned a Game directly, use it
                resultant_game = first
        except Exception:
            # As a last resort, assume game_copy was mutated in place
            resultant_game = game_copy

        # 3) evaluate with the base value function
        try:
            if self._value_fn is None:
                # Attempt to create the value function on-demand
                try:
                    self._value_fn = base_fn()
                except Exception as e:
                    if self.debug:
                        print("base_fn() factory failed:", e)
                        traceback.print_exc()
                    return None

            score = self._value_fn(resultant_game, my_color)
        except TypeError:
            # base_fn might have a different calling convention; catch and log
            if self.debug:
                print("base_fn evaluation TypeError for action", repr(action))
                traceback.print_exc()
            return None
        except Exception as e:
            if self.debug:
                print("base_fn evaluation failed for action", repr(action), "error:", e)
                traceback.print_exc()
            return None

        # Ensure numeric result
        try:
            return float(score)
        except Exception:
            if self.debug:
                print("Non-numeric score returned for action", repr(action), "score:", score)
            return None

    # ------------------ Decision method (public) ------------------
    def decide(self, game: Game, playable_actions: Iterable):
        """Choose an action from playable_actions using a 1-ply lookahead.

        This method follows the adapter-based strategy specified in META:
        - Sample/prune actions to keep runtime bounded.
        - Evaluate each candidate deterministically.
        - Select the highest-scoring action with a deterministic tie-break.
        """
        # Convert to list for stable indexing and reporting
        actions = list(playable_actions)

        # Defensive: no actions
        if not actions:
            if self.debug:
                print("decide: no playable_actions provided")
            return None

        # Quick-win: only one legal action
        if len(actions) == 1:
            if self.debug:
                print("decide: single playable action, returning it")
            return actions[0]

        my_color = self.color

        # Sample/prune to a candidate set
        candidates = self._sample_actions(actions)

        if self.debug:
            print(f"decide: evaluating {len(candidates)} candidate(s) out of {len(actions)} playable action(s)")

        best_action = None
        best_score = -float("inf")
        best_tie_repr = None

        # Evaluate candidates
        evaluated = 0
        for action in candidates:
            score = self._evaluate_action(game, action, my_color)
            evaluated += 1
            if self.debug:
                print(f"Evaluated action [{evaluated}/{len(candidates)}]: {repr(action)} -> {score}")

            if score is None:
                continue

            tie_repr = repr(action)
            # Deterministic tie-break: prefer numerically higher score; if equal, choose the
            # action with lexicographically smaller repr(action) to keep behavior stable.
            if (
                best_action is None
                or score > best_score
                or (score == best_score and (best_tie_repr is None or tie_repr < best_tie_repr))
            ):
                best_action = action
                best_score = score
                best_tie_repr = tie_repr

        # Fallbacks if evaluation failed for all candidates
        if best_action is None:
            if self.debug:
                print("decide: no candidate produced a valid evaluation; falling back to first playable action")
            # Preserve original deterministic behavior as a safe fallback
            return actions[0]

        if self.debug:
            print("decide: chosen action:", repr(best_action), "score:", best_score)

        return best_action
