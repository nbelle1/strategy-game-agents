import random
from typing import Iterable, List, Optional, Any, Tuple

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
    """A Foo player with game-phase aware decisioning, improved sampling,
    short rollouts, and richer heuristics.

    This implementation is defensive: it uses only the adapters surface and
    contains many fallbacks when attributes or adapter helpers are missing.

    Key new features vs earlier versions:
    - Game-phase detection (early/mid/late) to bias settlement/road vs city/dev-card choices
    - Settlement & road potential heuristics to encourage early expansion
    - Robber/knight evaluation to value disruptive moves and steals
    - Must-include guarantees for critical action types (settlement/road/robber/dev)
    - Rollout policy biased by phase to approximate downstream effects

    NOTE: Many game model attribute names vary across environments. This code
    attempts multiple common attribute names and falls back to string-based
    heuristics when necessary. If the next run raises AttributeError for an
    adapters function or a specific attribute, provide the traceback so it can
    be patched to the concrete environment.
    """

    # Tunable constants (exposed to edit for experimentation)
    MAX_SIMULATIONS = 24
    PREFILTER_TOP_K = 8
    ROLLOUT_DEPTH = 2
    SIMULATION_BUDGET = 60
    DEBUG = False

    # Phase thresholds
    EARLY_TURN_THRESHOLD = 25
    EARLY_VP_THRESHOLD = 5

    # Must-include action tokens (robust, lowercase matching)
    MUST_INCLUDE_TOKENS = {
        "build_city",
        "build_settlement",
        "build_sett",
        "build_road",
        "buy_dev",
        "buy_dev_card",
        "buycard",
        "play_knight",
        "knight",
        "move_robber",
        "move_robber_action",
        "robber",
        "trade",
        "offer_trade",
    }

    # Robber scoring base
    ROBBER_BASE_SCORE = 40.0

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
            at = getattr(action, "action_type", None)
            if at is None:
                at = getattr(action, "type", None)
            if at is not None:
                try:
                    return str(at.name).lower()
                except Exception:
                    return str(at).lower()
        except Exception:
            pass
        try:
            # Some Action objects have a .name or .action_name
            name = getattr(action, "name", None) or getattr(action, "action_name", None)
            if name is not None:
                return str(name).lower()
        except Exception:
            pass
        try:
            return str(action).lower()
        except Exception:
            return ""

    # ------------------- Phase detection -------------------
    def is_early_game(self, game: Game) -> bool:
        """Heuristic to detect early phase of the game.

        Prefer explicit turn counts if available, else fallback to max VP.
        """
        # Try turn/tick fields
        try:
            state = getattr(game, "state", game)
            turn_count = getattr(state, "turn", None) or getattr(state, "tick", None) or getattr(state, "turn_count", None) or getattr(state, "tick_count", None)
            if isinstance(turn_count, (int, float)):
                return int(turn_count) < self.EARLY_TURN_THRESHOLD
        except Exception:
            pass

        # Fallback: use maximum VP among players
        try:
            players = getattr(state, "players", None) or getattr(game, "players", None) or []
            max_vp = 0
            if isinstance(players, dict):
                for p in players.values():
                    vp = getattr(p, "victory_points", None) or getattr(p, "vp", None) or 0
                    try:
                        vp = int(vp)
                    except Exception:
                        vp = 0
                    if vp > max_vp:
                        max_vp = vp
            else:
                for p in players:
                    vp = getattr(p, "victory_points", None) or getattr(p, "vp", None) or 0
                    try:
                        vp = int(vp)
                    except Exception:
                        vp = 0
                    if vp > max_vp:
                        max_vp = vp
            return max_vp < self.EARLY_VP_THRESHOLD
        except Exception:
            # Conservative fallback: treat as mid-game
            return False

    # ------------------- Heuristic / evaluation (phase-aware) -------------------
    def _heuristic_value(self, game: Game, color: Color) -> float:
        """Phase-aware heuristic including production potential and city-upgrade progress.

        Many attribute names are attempted to be robust across different game models.
        """
        # Die probabilities for numbers 2..12 ignoring 7
        die_prob = {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36}

        # Player lookup
        player_state = None
        try:
            state = getattr(game, "state", game)
            players = getattr(state, "players", None) or getattr(game, "players", None)
            if isinstance(players, dict):
                player_state = players.get(color) or players.get(str(color))
            elif isinstance(players, (list, tuple)):
                for p in players:
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

        vp = _safe_get(player_state, "victory_points", "vp", default=0)
        settlements = _safe_get(player_state, "settlements", "settle_count", "settle_locations", default=0)
        if isinstance(settlements, (list, tuple)):
            settlements = len(settlements)
        cities = _safe_get(player_state, "cities", "city_count", "city_locations", default=0)
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

        # Production potential estimation
        prod_value = 0.0
        try:
            board = getattr(state, "board", None) or getattr(game, "board", None)
            hexes = getattr(board, "hexes", None) or getattr(board, "tiles", None) or []
            settlements_list = _safe_get(player_state, "settlements", "settle_locations", default=[])
            if isinstance(settlements_list, (list, tuple)):
                for s in settlements_list:
                    try:
                        for h in hexes:
                            neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                            if s in neighbors:
                                num = getattr(h, "roll", None) or getattr(h, "number", None) or getattr(h, "value", None)
                                try:
                                    num = int(num)
                                except Exception:
                                    num = None
                                if num in die_prob:
                                    prod_value += die_prob[num] * 1.0
                    except Exception:
                        continue
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
                                    prod_value += die_prob[num] * 2.0
                    except Exception:
                        continue
        except Exception:
            prod_value = 0.0

        # City upgrade progress heuristic
        city_resource_val = 0.0
        try:
            if isinstance(resources_obj, dict):
                wheat = resources_obj.get("wheat", 0) + resources_obj.get("grain", 0)
                ore = resources_obj.get("ore", 0) + resources_obj.get("metal", 0)
                city_resource_val = min(wheat, ore)
        except Exception:
            city_resource_val = 0.0

        # Phase multipliers
        early = self.is_early_game(game)
        if early:
            settlement_mul = 1.6
            road_mul = 1.4
            city_mul = 0.9
            dev_mul = 1.1
            prod_weight = 60.0
        else:
            settlement_mul = 0.9
            road_mul = 0.9
            city_mul = 1.4
            dev_mul = 1.0
            prod_weight = 40.0

        # Compose weighted sum
        score = (
            float(vp) * 100.0
            + float(settlements) * 25.0 * settlement_mul
            + float(cities) * 60.0 * city_mul
            + float(roads) * 6.0 * road_mul
            + float(dev_vp) * 50.0
            + float(resources_total) * 1.0
            + float(resource_diversity) * 2.0
            + float(city_resource_val) * 5.0
            + float(prod_value) * prod_weight
        )

        return float(score)

    def _evaluate_game_state(self, game: Game, color: Color) -> float:
        """Evaluate a single game state for the given player color.

        Prefer adapters.base_fn() if available (cached in self._value_fn). If available, combine
        it with the heuristic for stability. We keep phase multipliers inside the heuristic so
        they influence the final blended value.
        """
        heuristic = self._heuristic_value(game, color)
        if self._value_fn is not None:
            try:
                vf_val = float(self._value_fn(game, color))
                return 0.85 * vf_val + 0.15 * heuristic
            except Exception as e:
                self.debug_print("FooPlayer: value_fn failed during evaluate_game_state, falling back to heuristic. Error:", e)
        return float(heuristic)

    # ------------------- Cheap scoring & potentials -------------------
    def settlement_potential(self, action: Any, game: Game, color: Color) -> float:
        """Estimate benefit of a settlement action: new resource types and production.

        Best-effort: try to parse adjacent hexes from action or fallback to zero.
        """
        bonus = 0.0
        try:
            name = self._safe_action_name(action)
            # If action string contains resource hints or vertex index, give tiny bonus
            if any(tok in name for tok in ("build_settlement", "build_sett", "settle")):
                bonus += 5.0
        except Exception:
            pass
        return bonus

    def road_connection_potential(self, action: Any, game: Game, color: Color) -> float:
        """Estimate if a road action helps expansion. Best-effort string heuristic."""
        bonus = 0.0
        try:
            name = self._safe_action_name(action)
            if "build_road" in name or "road" in name:
                bonus += 2.0
        except Exception:
            pass
        return bonus

    def cheap_pre_score(self, action: Any, game: Game, color: Color) -> float:
        """Cheap, fast scoring used to prioritize actions for simulation (phase-aware)."""
        s = 0.0
        name = self._safe_action_name(action)

        early = self.is_early_game(game)
        # multipliers
        if early:
            settlement_mul = 1.6
            road_mul = 1.4
            city_mul = 0.9
            dev_mul = 1.1
        else:
            settlement_mul = 0.9
            road_mul = 0.9
            city_mul = 1.4
            dev_mul = 1.0

        # Reward direct VP gains
        if any(tok in name for tok in ("build_city",)):
            s += 100.0 * city_mul
        if any(tok in name for tok in ("build_settlement", "build_sett")):
            s += 90.0 * settlement_mul
        if "buy_dev" in name or "buycard" in name or "buy_dev_card" in name:
            s += 60.0 * dev_mul
        if "build_road" in name or ("road" in name and "build" in name):
            s += 20.0 * road_mul
        if "knight" in name or "play_knight" in name:
            s += 70.0
        if "robber" in name or "move_robber" in name:
            s += 50.0
        if "trade" in name or "offer_trade" in name:
            s += 10.0

        # Add small potentials
        s += self.settlement_potential(action, game, color)
        s += self.road_connection_potential(action, game, color)

        # Minor random tie-break
        s += random.random() * 1e-3
        return s

    # ------------------- Prefilter actions (phase-aware guarantees) -------------------
    def prefilter_actions(self, actions: List[Any], game: Game, color: Color) -> List[Any]:
        """Return a bounded list of candidate actions to evaluate thoroughly.

        Guarantees inclusion of must-include tokens and early-game settlement/road actions.
        """
        if not actions:
            return []

        all_actions = list(actions)
        early = self.is_early_game(game)

        musts = []
        others = []
        for a in all_actions:
            name = self._safe_action_name(a)
            if any(tok in name for tok in self.MUST_INCLUDE_TOKENS):
                # Early-game guarantee: ensure settlement/road included
                if early and any(tok in name for tok in ("build_settlement", "build_sett", "build_road", "road")):
                    if a not in musts:
                        musts.append(a)
                else:
                    if a not in musts:
                        musts.append(a)
            else:
                others.append(a)

        # Score and pick top-K from others
        scored = [(self.cheap_pre_score(a, game, color), a) for a in others]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [a for (_s, a) in scored[: self.PREFILTER_TOP_K]]

        # Combine unique musts + top_k preserving order
        candidates = []
        for a in musts + top_k:
            if a not in candidates:
                candidates.append(a)

        # Fill up with random remaining samples until MAX_SIMULATIONS
        remaining = [a for a in all_actions if a not in candidates]
        random.shuffle(remaining)
        while len(candidates) < min(len(all_actions), self.MAX_SIMULATIONS) and remaining:
            candidates.append(remaining.pop())

        if not candidates and all_actions:
            candidates = random.sample(all_actions, min(len(all_actions), self.MAX_SIMULATIONS))

        self.debug_print(f"FooPlayer: Prefilter selected {len(candidates)} candidates (musts={len(musts)}, early={early})")
        return candidates

    # ------------------- Playable actions extraction -------------------
    def get_playable_actions_from_game(self, game: Game) -> List[Any]:
        """Try adapters.list_prunned_actions first, then common game attributes."""
        try:
            acts = list_prunned_actions(game)
            if acts:
                return acts
        except Exception as e:
            self.debug_print("FooPlayer: list_prunned_actions unavailable or failed. Error:", e)

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

    # ------------------- Robber / Knight evaluation -------------------
    def evaluate_robber_action(self, action: Any, game: Game, color: Color) -> float:
        """Estimate the value of moving the robber (best-effort)."""
        score = 0.0
        try:
            # Base preference to include robber moves
            score += self.ROBBER_BASE_SCORE
            # Try to detect a target hex id from action string
            name = self._safe_action_name(action)
            # If action contains a hex index, reward it slightly
            # (Concrete parsing depends on action repr; this is best-effort)
            for tok in name.split():
                if tok.isdigit():
                    score += 5.0
                    break
        except Exception:
            pass
        return score

    def evaluate_play_knight(self, action: Any, game: Game, color: Color) -> float:
        """Estimate the value of playing a knight (best-effort)."""
        score = 10.0
        # Stronger if it could lead to Largest Army or steal
        try:
            name = self._safe_action_name(action)
            if "steal" in name or "rob" in name:
                score += 10.0
        except Exception:
            pass
        return score

    # ------------------- Rollout logic (phase-biased) -------------------
    def rollout_value(self, game: Game, color: Color, depth: int) -> float:
        """Short greedy rollout with phase bias."""
        try:
            if depth <= 0:
                return self._evaluate_game_state(game, color)

            actions = self.get_playable_actions_from_game(game)
            if not actions:
                return self._evaluate_game_state(game, color)

            # Bias selection by phase: early -> settlement/road favored; late -> city/dev
            early = self.is_early_game(game)
            sorted_actions = sorted(actions, key=lambda a: self.cheap_pre_score(a, game, color), reverse=True)

            # Try top few actions until one simulates
            for a in sorted_actions[:6]:
                # If early, prefer settlement/road in the slice
                if early and not any(tok in self._safe_action_name(a) for tok in ("build_settlement", "settle", "road")):
                    continue
                try:
                    branches = execute_deterministic(game, a)
                except Exception:
                    try:
                        branches = execute_spectrum(game, a)
                    except Exception:
                        branches = []
                if not branches:
                    continue
                # pick most probable branch
                best_branch = max(branches, key=lambda bp: float(bp[1]))
                next_game = best_branch[0]
                return self.rollout_value(next_game, color, depth - 1)

            # fallback to picking any action that simulates
            for a in sorted_actions[:6]:
                try:
                    branches = execute_deterministic(game, a)
                except Exception:
                    try:
                        branches = execute_spectrum(game, a)
                    except Exception:
                        branches = []
                if branches:
                    next_game = max(branches, key=lambda bp: float(bp[1]))[0]
                    return self.rollout_value(next_game, color, depth - 1)

            return self._evaluate_game_state(game, color)
        except Exception as e:
            self.debug_print("FooPlayer: rollout_value exception, falling back to evaluate_game_state. Error:", e)
            return self._evaluate_game_state(game, color)

    # ------------------- Evaluate action expectation (enhanced) -------------------
    def _evaluate_action_expectation(self, game: Game, action: Any, per_action_branch_limit: int = 8) -> float:
        """Compute expected value of taking `action` in `game` for this player.

        Uses execute_spectrum when available then adds a rollout estimate for depth-1.
        """
        color = self._get_player_color()

        # Special-case robber/knight/dev scoring quick boost before heavy sim
        name = self._safe_action_name(action)
        preboost = 0.0
        if any(tok in name for tok in ("move_robber", "robber")):
            preboost += self.evaluate_robber_action(action, game, color)
        if any(tok in name for tok in ("knight", "play_knight")):
            preboost += self.evaluate_play_knight(action, game, color)

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
            branches = sorted(branches, key=lambda bp: float(bp[1]), reverse=True)[:per_action_branch_limit]

        expected = 0.0
        total_prob = 0.0
        # Determine rollout depth possibly reduced if budget is tight
        rollout_depth = max(0, self.ROLLOUT_DEPTH - 1)
        for (out_game, prob) in branches:
            try:
                immediate = self._evaluate_game_state(out_game, color)
                rollout_est = self.rollout_value(out_game, color, rollout_depth)
                branch_val = 0.6 * immediate + 0.4 * rollout_est
            except Exception as e:
                self.debug_print("FooPlayer: evaluation failed for branch, using heuristic. Error:", e)
                branch_val = self._heuristic_value(out_game, color)
            expected += float(prob) * float(branch_val)
            total_prob += float(prob)

        if total_prob > 0:
            expected = expected / total_prob

        # Add any preboost (robber/knight quick estimate)
        expected += preboost
        return float(expected)

    # ------------------- Main decision function -------------------
    def decide(self, game: Game, playable_actions: Iterable) -> Optional[object]:
        """Choose an action from playable_actions using phase-aware sampling + rollouts."""
        try:
            playable_actions = list(playable_actions)
            if not playable_actions:
                self.debug_print("FooPlayer: No playable actions available, returning None")
                return None

            color = self._get_player_color()
            early = self.is_early_game(game)

            # Prefilter candidate actions
            candidates = self.prefilter_actions(playable_actions, game, color)

            # Cap to MAX_SIMULATIONS
            if len(candidates) > self.MAX_SIMULATIONS:
                candidates = candidates[: self.MAX_SIMULATIONS]

            if not candidates:
                candidates = random.sample(playable_actions, min(len(playable_actions), self.MAX_SIMULATIONS))

            # Distribute simulation budget adaptively
            per_action_budget = max(1, self.SIMULATION_BUDGET // max(1, len(candidates)))

            best_score = float("-inf")
            best_actions: List[Any] = []
            scores_debug: List[Tuple[float, Any]] = []

            for a in candidates:
                try:
                    score = self._evaluate_action_expectation(game, a, per_action_branch_limit=per_action_budget)
                except Exception as e:
                    self.debug_print("FooPlayer: Exception during action evaluation, skipping action. Error:", e)
                    score = float("-inf")

                scores_debug.append((score, a))

                if score > best_score:
                    best_score = score
                    best_actions = [a]
                elif score == best_score:
                    best_actions.append(a)

            # If no action had a finite score, fallback to first playable action
            if not best_actions:
                self.debug_print("FooPlayer: All evaluations failed, defaulting to first playable action")
                return playable_actions[0]

            # Debug logging: phase and top candidates
            if self.DEBUG:
                phase = "EARLY" if early else "MID/LATE"
                self.debug_print(f"FooPlayer: Phase={phase}, chosen multipliers: settlement/road/city based on phase")
                scores_debug.sort(key=lambda x: x[0], reverse=True)
                topn = scores_debug[:3]
                self.debug_print("FooPlayer: Top candidates:")
                for sc, act in topn:
                    self.debug_print(f"  score={sc:.2f} action={act}")

            chosen = random.choice(best_actions)
            self.debug_print(f"FooPlayer: Chosen action {chosen} with expected score {best_score}")
            return chosen
        except Exception as e:
            # Protect against unexpected errors
            print("FooPlayer: Unexpected error in decide(), defaulting to first playable action. Error:", e)
            try:
                return list(playable_actions)[0]
            except Exception:
                return None
