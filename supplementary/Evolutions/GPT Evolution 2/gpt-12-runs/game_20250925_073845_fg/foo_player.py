import random
from typing import Iterable, List, Optional

# MUST use the adapters surface to interact with the game environment
from .adapters import (
    Game,
    Player,
    Color,
    execute_deterministic,
    execute_spectrum,
    base_fn,
)


class FooPlayer(Player):
    """A simple Foo player that uses a 1-ply lookahead and heuristic fallback.

    Strategy summary:
    - For each candidate action, expand deterministic (and fallback to spectrum) outcomes
      using adapters.execute_deterministic / execute_spectrum.
    - Evaluate resulting game states with adapters.base_fn() when available (preferred),
      otherwise fall back to a lightweight heuristic that inspects the player's state.
    - Select the action with the highest expected value. Break ties randomly.

    Notes:
    - We call only the adapter functions (no direct imports from catanatron internals).
    - We include verbose prints to help debug runs. These can be reduced later.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(Color.BLUE, name)
        # Cache a value function factory if available. We will attempt to use adapters.base_fn()
        try:
            self._value_fn = base_fn()
            print("FooPlayer: Using adapters.base_fn() for state evaluation")
        except Exception as e:
            self._value_fn = None
            print("FooPlayer: adapters.base_fn() not available, falling back to heuristic. Error:", e)

    # ------------------- Helper functions -------------------
    def _get_player_color(self) -> Color:
        """Return this player's color. Try common attribute names."""
        # Player class from adapters should set a color attribute. We defensively handle naming.
        if hasattr(self, "color"):
            return getattr(self, "color")
        if hasattr(self, "_color"):
            return getattr(self, "_color")
        # Fallback to the Color assigned in constructor
        return Color.BLUE

    def _heuristic_value(self, game: Game, color: Color) -> float:
        """A fast heuristic to score a game state for the given player color.

        This heuristic is intentionally simple and robust to missing/variant attributes.
        Weighted sum:
            VP * 100 + settlements * 20 + cities * 50 + roads * 5 + resources * 1
        If exact attributes are missing, we attempt several common attribute names and
        fall back to zero.
        """
        # Attempt to locate the player's state inside the game object
        player_state = None
        try:
            players_container = getattr(game.state, "players", None)
            if players_container is None:
                # sometimes game may store players as a list directly on the game
                players_container = getattr(game, "players", None)

            # If players is a dict keyed by Color, try that first
            if isinstance(players_container, dict):
                player_state = players_container.get(color) or players_container.get(str(color))
            elif isinstance(players_container, (list, tuple)):
                # Attempt to find by color attribute
                for p in players_container:
                    if getattr(p, "color", None) == color or getattr(p, "color", None) == str(color):
                        player_state = p
                        break
            else:
                player_state = None
        except Exception:
            player_state = None

        # Extract common metrics defensively
        def _safe_get(obj, *names, default=0):
            if obj is None:
                return default
            for name in names:
                try:
                    val = getattr(obj, name)
                    if val is not None:
                        return val
                except Exception:
                    try:
                        # maybe it's a dict-like
                        val = obj[name]
                        if val is not None:
                            return val
                    except Exception:
                        continue
            return default

        vp = _safe_get(player_state, "victory_points", "vp", default=0)
        # settlements/cities/roads may be stored as lists or counts
        settlements = _safe_get(player_state, "settlements", "settle_count", default=0)
        if isinstance(settlements, (list, tuple)):
            settlements = len(settlements)
        cities = _safe_get(player_state, "cities", "city_count", default=0)
        if isinstance(cities, (list, tuple)):
            cities = len(cities)
        roads = _safe_get(player_state, "roads", "road_count", default=0)
        if isinstance(roads, (list, tuple)):
            roads = len(roads)

        # resources might be dict-like
        resources_obj = _safe_get(player_state, "resources", default=0)
        resources_total = 0
        try:
            if isinstance(resources_obj, dict):
                resources_total = sum(resources_obj.values())
            elif isinstance(resources_obj, (list, tuple)):
                resources_total = sum(resources_obj)
            else:
                resources_total = int(resources_obj)
        except Exception:
            resources_total = 0

        score = (
            float(vp) * 100.0
            + float(settlements) * 20.0
            + float(cities) * 50.0
            + float(roads) * 5.0
            + float(resources_total) * 1.0
        )
        return score

    def _evaluate_game_state(self, game: Game, color: Color) -> float:
        """Evaluate a single game state for the given player color.

        Prefer adapters.base_fn() if available (cached in self._value_fn). If that fails for any
        reason, use the heuristic fallback implemented above.
        """
        if self._value_fn is not None:
            try:
                # value functions in adapters typically take (game, color) and return float
                return float(self._value_fn(game, color))
            except Exception as e:
                # If the value function fails, print debug and fallback
                print("FooPlayer: value_fn failed, falling back to heuristic. Error:", e)

        # fallback heuristic
        return float(self._heuristic_value(game, color))

    def _evaluate_action_expectation(self, game: Game, action) -> float:
        """Compute expected value of taking `action` in `game` for this player.

        We attempt to expand deterministic outcomes via adapters.execute_deterministic.
        If that returns multiple outcomes or fails, we try execute_spectrum. If all fail,
        return a very low score so the action is unlikely to be chosen.
        """
        color = self._get_player_color()

        # First try deterministic expansion (adapter should produce copies internally)
        try:
            outcomes = execute_deterministic(game, action)
            # outcomes: List[Tuple[Game, float]]
            if not outcomes:
                raise RuntimeError("execute_deterministic returned no outcomes")
        except Exception as e_det:
            # Try broader spectrum expansion as fallback
            try:
                print("FooPlayer: execute_deterministic failed for action, trying spectrum. Error:", e_det)
                outcomes = execute_spectrum(game, action)
                if not outcomes:
                    raise RuntimeError("execute_spectrum returned no outcomes")
            except Exception as e_spec:
                print("FooPlayer: Both deterministic and spectrum execution failed for action. Errors:", e_det, e_spec)
                # Return a very low score to make this action unattractive
                return float("-inf")

        # Compute expected value over outcomes
        expected = 0.0
        total_prob = 0.0
        for outcome_game, prob in outcomes:
            try:
                val = self._evaluate_game_state(outcome_game, color)
            except Exception as e:
                print("FooPlayer: evaluation of outcome failed, using heuristic 0. Error:", e)
                val = self._heuristic_value(outcome_game, color)
            expected += val * float(prob)
            total_prob += float(prob)

        # Normalize if probabilities don't sum exactly to 1
        if total_prob > 0:
            expected = expected / total_prob
        return expected

    # ------------------- Main decision function -------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Optional[object]:
        """Choose an action from playable_actions using a 1-ply expected-value lookahead.

        - If evaluation fails for all actions, fall back to the first playable action.
        - To limit runtime, if the action set is very large we sample a subset.
        """
        playable_actions = list(playable_actions)
        if not playable_actions:
            print("FooPlayer: No playable actions available, returning None")
            return None

        # If there are many actions, sample a subset to keep runtime reasonable
        MAX_SIMULATIONS = 16
        actions_to_evaluate: List = playable_actions
        if len(playable_actions) > MAX_SIMULATIONS:
            # Take a mix of highest-priority heuristically good actions and random samples
            # Quick heuristic: score actions by applying them *naively* by evaluating their string repr
            # This is cheap and only used to pick candidates to simulate.
            random.shuffle(playable_actions)
            actions_to_evaluate = playable_actions[:MAX_SIMULATIONS]
            print(f"FooPlayer: Large action space ({len(playable_actions)}), sampling {len(actions_to_evaluate)} actions")

        best_score = float("-inf")
        best_actions: List = []

        # Evaluate each candidate action
        for a in actions_to_evaluate:
            try:
                score = self._evaluate_action_expectation(game, a)
            except Exception as e:
                print("FooPlayer: Exception during action evaluation, skipping action. Error:", e)
                score = float("-inf")

            print(f"FooPlayer: Action {a} -> expected score {score}")

            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score == best_score:
                best_actions.append(a)

        # If everything failed and best_score is -inf, fall back to first playable action
        if not best_actions:
            print("FooPlayer: All action evaluations failed, defaulting to first playable action")
            return playable_actions[0]

        chosen = random.choice(best_actions)
        print(f"FooPlayer: Chosen action {chosen} with expected score {best_score}")
        return chosen
