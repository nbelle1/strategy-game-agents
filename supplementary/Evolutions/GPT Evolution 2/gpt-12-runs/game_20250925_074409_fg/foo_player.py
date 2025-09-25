import random
from typing import Iterable, List, Optional, Any

# MUST use the adapters surface to interact with the game environment
from .adapters import (
    Game,
    Player,
    Color,
    copy_game,
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    list_prunned_actions,
    prune_robber_actions,
    base_fn,
    value_production,
    get_value_fn,
)


class FooPlayer(Player):
    """A Foo player with improved action sampling, 1-ply expansion + short rollouts,
    and an enhanced heuristic fallback.

    Key ideas implemented:
    - Prefilter actions to always include high-impact moves (city/settlement/robber/etc.).
    - Use adapters.execute_spectrum / execute_deterministic to expand chance outcomes.
    - For each outcome, evaluate with adapters.base_fn() when available, otherwise a
      richer heuristic that includes production potential and city-upgrade progress.
    - Run a short greedy rollout (depth-limited) from each outcome to estimate
      downstream value.
    - Keep many robust try/except fallbacks so the player still returns an action
      if parts of the adapters API differ in the environment.

    Notes on adapters usage:
    - We only call functions exposed by .adapters. If the next run raises AttributeError
      for an adapters function used here (e.g., expand_spectrum), report the traceback
      so RESEARCHER can map the exact adapter surface.
    """

    # Tunable constants
    MAX_SIMULATIONS = 24  # cap actions to evaluate per decision
    PREFILTER_TOP_K = 8  # after must-includes, take this many top cheap-scored actions
    ROLLOUT_DEPTH = 2  # depth for short greedy rollout
    SIMULATION_BUDGET = 60  # approximate cap on total expanded branches across actions
    DEBUG = False  # Set True to enable debug printing

    # Action tokens we consider high-impact (match against action type/name/str)
    MUST_INCLUDE_TOKENS = {
        "build_city",
        "build_settlement",
        "build_road",
        "buy_dev",
        "buy_dev_card",
        "play_knight",
        "move_robber",
        "move_robber_action",
        "trade",
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(Color.BLUE, name)
        # Try to cache a base value function from adapters
        try:
            self._value_fn = base_fn()
            self.debug_print("FooPlayer: Using adapters.base_fn() for evaluation")
        except Exception as e:
            self._value_fn = None
            self.debug_print("FooPlayer: adapters.base_fn() not available, will use heuristic. Error:", e)

    # ------------------- Debug helper -------------------
    def debug_print(self, *args: Any) -> None:
        if self.DEBUG:
            print(*args)

    # ------------------- Utility helpers -------------------
    def _get_player_color(self) -> Color:
        """Return this player's color. Try common attribute names."""
        if hasattr(self, "color"):
            return getattr(self, "color")
        if hasattr(self, "_color"):
            return getattr(self, "_color")
        return Color.BLUE

    def _safe_action_name(self, action: Any) -> str:
        """Produce a lowercase string name for the action for robust matching."""
        try:
            # action.action_type may be an enum with .name
            at = getattr(action, "action_type", None)
            if at is None:
                at = getattr(action, "type", None)
            if at is not None:
                try:
                    # enum values often have .name
                    return str(at.name).lower()
                except Exception:
                    return str(at).lower()
        except Exception:
            pass
        # Fallback to stringifying the action
        try:
            return str(action).lower()
        except Exception:
            return ""

    # ------------------- Heuristic / evaluation -------------------
    def _heuristic_value(self, game: Game, color: Color) -> float:
        """Enhanced heuristic including production potential and city-upgrade progress.

        This is deliberately defensive to handle variations in the game model.
        """
        # Die probabilities for numbers 2..12 ignoring 7
        die_prob = {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36}

        # Helper to find player object/state
        player_state = None
        try:
            players_container = getattr(getattr(game, "state", game), "players", None)
            if players_container is None:
                players_container = getattr(game, "players", None)

            if isinstance(players_container, dict):
                # Keys might be Color or string
                player_state = players_container.get(color) or players_container.get(str(color))
            elif isinstance(players_container, (list, tuple)):
                for p in players_container:
                    if getattr(p, "color", None) == color or getattr(p, "color", None) == str(color):
                        player_state = p
                        break
        except Exception:
            player_state = None

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
                        val = obj[name]
                        if val is not None:
                            return val
                    except Exception:
                        continue
            return default

        # Base counts
        vp = _safe_get(player_state, "victory_points", "vp", default=0)
        settlements = _safe_get(player_state, "settlements", "settle_count", default=0)
        if isinstance(settlements, (list, tuple)):
            settlements = len(settlements)
        cities = _safe_get(player_state, "cities", "city_count", default=0)
        if isinstance(cities, (list, tuple)):
            cities = len(cities)
        roads = _safe_get(player_state, "roads", "road_count", default=0)
        if isinstance(roads, (list, tuple)):
            roads = len(roads)
        dev_vp = _safe_get(player_state, "dev_vp", "dev_victory_points", default=0)

        # Resources summary
        resources_obj = _safe_get(player_state, "resources", default=0)
        resources_total = 0
        resource_diversity = 0
        try:
            if isinstance(resources_obj, dict):
                resources_total = sum(resources_obj.values())
                resource_diversity = sum(1 for v in resources_obj.values() if v > 0)
            elif isinstance(resources_obj, (list, tuple)):
                resources_total = sum(resources_obj)
                resource_diversity = sum(1 for v in resources_obj if v > 0)
            else:
                resources_total = int(resources_obj)
                resource_diversity = 1 if resources_total > 0 else 0
        except Exception:
            resources_total = 0
            resource_diversity = 0

        # Production potential estimation: look for player's settlements/cities on hexes
        prod_value = 0.0
        try:
            # Try common structures: game.state.board.hexes or game.state.board
            board = getattr(getattr(game, "state", game), "board", None) or getattr(game, "board", None)
            if board is not None:
                # hexes might be a list or dict keyed by index
                hexes = getattr(board, "hexes", None) or getattr(board, "tiles", None) or []
                # Player locations might be stored on player_state as lists of vertex indices
                settlements_list = _safe_get(player_state, "settlements", "settle_locations", default=[])
                if isinstance(settlements_list, (list, tuple)):
                    for s in settlements_list:
                        # try to map s -> adjacent hex indices
                        try:
                            # Heuristic: if hexes is list and s is indexable, check neighbors attribute
                            # Many implementations store vertex->hex adjacency; this is best-effort.
                            hex_indices = []
                            if isinstance(hexes, (list, tuple)):
                                # Search hexes for ones annotated with adjacency to this vertex
                                for h in hexes:
                                    neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                                    if s in neighbors:
                                        num = getattr(h, "roll", None) or getattr(h, "number", None) or getattr(h, "value", None)
                                        try:
                                            num = int(num)
                                        except Exception:
                                            num = None
                                        if num in die_prob:
                                            prod_value += die_prob[num] * 1.0  # settlement weight
                            else:
                                # hexes may be dict-like; attempt adjacency lookup
                                pass
                        except Exception:
                            continue
                # City production double-weight (best-effort: if cities stored separately)
                cities_list = _safe_get(player_state, "cities", "city_locations", default=[])
                if isinstance(cities_list, (list, tuple)):
                    for c in cities_list:
                        try:
                            for h in hexes:
                                neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                                if c in neighbors:
                                    num = getattr(h, "roll", None) or getattr(h, "number", None) or getattr(h, "value", None)
                                    try:
                                        num = int(num)
                                    except Exception:
                                        num = None
                                    if num in die_prob:
                                        prod_value += die_prob[num] * 2.0  # city weight
                        except Exception:
                            continue
        except Exception:
            prod_value = 0.0

        # City upgrade progress heuristic: reward having resources that contribute to city (ore + wheat)
        city_resource_val = 0.0
        try:
            if isinstance(resources_obj, dict):
                wheat = resources_obj.get("wheat", 0) + resources_obj.get("grain", 0)
                ore = resources_obj.get("ore", 0) + resources_obj.get("metal", 0)
                city_resource_val = min(wheat, ore)  # rough proxy towards ability to upgrade
        except Exception:
            city_resource_val = 0.0

        # Compose weighted sum - tuned to prefer VPs and production
        score = (
            float(vp) * 100.0
            + float(settlements) * 25.0
            + float(cities) * 60.0
            + float(roads) * 6.0
            + float(dev_vp) * 50.0
            + float(resources_total) * 1.0
            + float(resource_diversity) * 2.0
            + float(city_resource_val) * 5.0
            + float(prod_value) * 40.0
        )

        return float(score)

    def _evaluate_game_state(self, game: Game, color: Color) -> float:
        """Evaluate a single game state for the given player color.

        Prefer adapters.base_fn() if available (cached in self._value_fn). If available, combine
        it with the heuristic for stability: 0.85*value_fn + 0.15*heuristic.
        """
        heuristic = self._heuristic_value(game, color)
        if self._value_fn is not None:
            try:
                vf_val = float(self._value_fn(game, color))
                # Blend for stability
                return 0.85 * vf_val + 0.15 * heuristic
            except Exception as e:
                self.debug_print("FooPlayer: value_fn failed during evaluate_game_state, falling back to heuristic. Error:", e)
        return float(heuristic)

    # ------------------- Action sampling / prefilter -------------------
    def cheap_pre_score(self, action: Any, game: Game, color: Color) -> float:
        """Cheap, very fast scoring used to prioritize actions for simulation.

        This must be fast and not perform copying or heavy simulation.
        """
        s = 0.0
        name = self._safe_action_name(action)
        # Reward direct VP gains
        if any(tok in name for tok in ("build_city", "build_settlement", "build_sett")):
            s += 100.0
        if "buy_dev" in name or "buycard" in name or "buy_dev_card" in name:
            s += 60.0
        if "build_road" in name or "road" in name:
            s += 20.0
        if "knight" in name or "play_knight" in name or "play_kn" in name:
            s += 70.0
        if "robber" in name or "move_robber" in name:
            s += 50.0
        if "trade" in name or "offer_trade" in name:
            s += 10.0

        # Minor random tie-break to diversify decisions when cheap scores equal
        s += random.random() * 1e-3
        return s

    def prefilter_actions(self, actions: List[Any], game: Game, color: Color) -> List[Any]:
        """Return a bounded list of candidate actions to evaluate thoroughly.

        Steps:
        - Always include must-include actions by token match.
        - Score remaining actions with cheap_pre_score and pick top PREFILTER_TOP_K.
        - Fill up with random samples up to MAX_SIMULATIONS to keep diversity.
        """
        if not actions:
            return []

        # Normalize action list
        all_actions = list(actions)

        musts = []
        others = []
        for a in all_actions:
            name = self._safe_action_name(a)
            if any(tok in name for tok in self.MUST_INCLUDE_TOKENS):
                musts.append(a)
            else:
                others.append(a)

        # Score others quickly
        scored = [(self.cheap_pre_score(a, game, color), a) for a in others]
        scored.sort(key=lambda x: x[0], reverse=True)

        top_k = [a for (_s, a) in scored[: self.PREFILTER_TOP_K]]

        # Combine unique musts + top_k preserving order with uniqueness
        candidates = []
        for a in musts + top_k:
            if a not in candidates:
                candidates.append(a)

        # Fill up with random remaining samples until MAX_SIMULATIONS or out of actions
        remaining = [a for a in all_actions if a not in candidates]
        random.shuffle(remaining)
        while len(candidates) < min(len(all_actions), self.MAX_SIMULATIONS) and remaining:
            candidates.append(remaining.pop())

        # If still empty for some reason, fallback to a small random sample
        if not candidates and all_actions:
            candidates = random.sample(all_actions, min(len(all_actions), self.MAX_SIMULATIONS))

        self.debug_print(f"FooPlayer: Prefilter selected {len(candidates)} candidates (musts={len(musts)})")
        return candidates

    # ------------------- Playable actions extraction -------------------
    def get_playable_actions_from_game(self, game: Game) -> List[Any]:
        """Try a number of adapters/game methods to extract playable actions in this state.

        We prefer adapters.list_prunned_actions(game) if available.
        """
        try:
            actions = list_prunned_actions(game)
            if actions:
                return actions
        except Exception as e:
            self.debug_print("FooPlayer: list_prunned_actions unavailable or failed. Error:", e)

        # Try common game-provided methods/attributes
        try:
            if hasattr(game, "get_playable_actions"):
                return list(game.get_playable_actions())
        except Exception:
            pass
        try:
            if hasattr(game, "playable_actions"):
                return list(getattr(game, "playable_actions"))
        except Exception:
            pass
        try:
            state = getattr(game, "state", None)
            if state is not None and hasattr(state, "playable_actions"):
                return list(getattr(state, "playable_actions"))
        except Exception:
            pass

        return []

    # ------------------- Rollout logic -------------------
    def rollout_value(self, game: Game, color: Color, depth: int) -> float:
        """Perform a short greedy rollout from `game` for `depth` steps and return an evaluation.

        The rollout picks the best cheap_pre_score action for the active player at each step,
        simulates a deterministic branch, and continues. This is fast and approximate.
        """
        try:
            if depth <= 0:
                return self._evaluate_game_state(game, color)

            # Get playable actions for the current active player
            actions = self.get_playable_actions_from_game(game)
            if not actions:
                return self._evaluate_game_state(game, color)

            # Rank actions cheaply and try to simulate the top few until one succeeds
            actions_sorted = sorted(actions, key=lambda a: self.cheap_pre_score(a, game, color), reverse=True)
            # Limit branching inside rollout to a small constant for speed
            for a in actions_sorted[:4]:
                try:
                    outcomes = execute_deterministic(game, a)
                except Exception:
                    try:
                        outcomes = execute_spectrum(game, a)
                    except Exception:
                        outcomes = []
                if not outcomes:
                    continue
                # Choose the most probable branch
                best_branch = max(outcomes, key=lambda bp: float(bp[1]))
                next_game = best_branch[0]
                # Recurse
                return self.rollout_value(next_game, color, depth - 1)

            # If none simulated, fallback to evaluation
            return self._evaluate_game_state(game, color)
        except Exception as e:
            self.debug_print("FooPlayer: rollout_value exception, falling back to evaluate_game_state. Error:", e)
            return self._evaluate_game_state(game, color)

    # ------------------- Evaluate action expectation (enhanced) -------------------
    def _evaluate_action_expectation(self, game: Game, action: Any, per_action_branch_limit: int = 8) -> float:
        """Compute expected value of taking `action` in `game` for this player.

        Expands chance outcomes (execute_spectrum preferred), evaluates each branch and
        adds a short rollout estimate to approximate downstream value.
        """
        color = self._get_player_color()

        # Try spectrum first for a full branching view
        branches = None
        try:
            branches = execute_spectrum(game, action)
            if not branches:
                raise RuntimeError("execute_spectrum returned no branches")
        except Exception as e_s:
            self.debug_print("FooPlayer: execute_spectrum failed or unavailable for action; trying deterministic. Error:", e_s)
            try:
                branches = execute_deterministic(game, action)
                if not branches:
                    raise RuntimeError("execute_deterministic returned no outcomes")
            except Exception as e_d:
                self.debug_print("FooPlayer: Both execute_spectrum and execute_deterministic failed for action. Errors:", e_s, e_d)
                return float("-inf")

        # Limit branches to keep runtime bounded
        if len(branches) > per_action_branch_limit:
            # Keep most probable branches
            branches = sorted(branches, key=lambda bp: float(bp[1]), reverse=True)[:per_action_branch_limit]

        expected = 0.0
        total_prob = 0.0
        for (out_game, prob) in branches:
            try:
                # Immediate evaluation
                immediate = self._evaluate_game_state(out_game, color)
                # Add rollout estimate from this branch (depth-1)
                rollout_est = self.rollout_value(out_game, color, max(0, self.ROLLOUT_DEPTH - 1))
                branch_val = 0.6 * immediate + 0.4 * rollout_est
            except Exception as e:
                self.debug_print("FooPlayer: evaluation failed for branch, using heuristic. Error:", e)
                branch_val = self._heuristic_value(out_game, color)
            expected += float(prob) * float(branch_val)
            total_prob += float(prob)

        if total_prob > 0:
            expected = expected / total_prob
        return float(expected)

    # ------------------- Main decision function -------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Optional[object]:
        """Choose an action from playable_actions using enhanced sampling + rollout estimation."""
        try:
            playable_actions = list(playable_actions)
            if not playable_actions:
                self.debug_print("FooPlayer: No playable actions available, returning None")
                return None

            color = self._get_player_color()

            # Prefilter candidate actions to evaluate
            candidates = self.prefilter_actions(playable_actions, game, color)

            # If many candidates remain, cap to MAX_SIMULATIONS
            if len(candidates) > self.MAX_SIMULATIONS:
                candidates = candidates[: self.MAX_SIMULATIONS]

            # If still empty, fall back to random subset of playable_actions
            if not candidates:
                candidates = random.sample(playable_actions, min(len(playable_actions), self.MAX_SIMULATIONS))

            # Distribute simulation budget across candidates
            per_action_budget = max(1, self.SIMULATION_BUDGET // max(1, len(candidates)))

            best_score = float("-inf")
            best_actions: List[Any] = []
            scores_debug = []

            for a in candidates:
                try:
                    score = self._evaluate_action_expectation(game, a, per_action_branch_limit=per_action_budget)
                except Exception as e:
                    self.debug_print("FooPlayer: Exception during action evaluation, skipping action. Error:", e)
                    score = float("-inf")

                scores_debug.append((score, a))
                self.debug_print(f"FooPlayer: Action {a} -> expected score {score}")

                if score > best_score:
                    best_score = score
                    best_actions = [a]
                elif score == best_score:
                    best_actions.append(a)

            # If no action had a finite score, fallback to first playable action
            if not best_actions:
                self.debug_print("FooPlayer: All evaluations failed, defaulting to first playable action")
                return playable_actions[0]

            # Log top 3 candidates when debugging
            if self.DEBUG:
                scores_debug.sort(key=lambda x: x[0], reverse=True)
                topn = scores_debug[:3]
                self.debug_print("FooPlayer: Top candidates:")
                for sc, act in topn:
                    self.debug_print(f"  score={sc:.2f} action={act}")

            chosen = random.choice(best_actions)
            self.debug_print(f"FooPlayer: Chosen action {chosen} with expected score {best_score}")
            return chosen
        except Exception as e:
            # Protect against unexpected errors in the decision pipeline
            print("FooPlayer: Unexpected error in decide(), defaulting to first playable action. Error:", e)
            try:
                return list(playable_actions)[0]
            except Exception:
                return None
