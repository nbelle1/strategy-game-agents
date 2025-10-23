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

    Key features:
    - Game-phase detection (early/mid/late) to bias settlement/road vs city/dev-card
    - Settlement & road potential heuristics to encourage early expansion
    - Robber/knight evaluation to value disruption and steals
    - Must-include guarantees for critical action types (settlement/road/robber/dev)
    - Rollout policy biased by phase and includes a light opponent-response

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

    # Phase thresholds (used by get_game_phase)
    EARLY_TURN_THRESHOLD = 20
    MID_TURN_THRESHOLD = 45

    # Phase multipliers matrix (explicit)
    MULTS = {
        "EARLY": {"settlement": 2.0, "road": 1.8, "city": 0.8, "dev": 1.2},
        "MID": {"settlement": 1.0, "road": 1.0, "city": 1.25, "dev": 1.0},
        "LATE": {"settlement": 0.8, "road": 0.9, "city": 1.5, "dev": 1.0},
    }

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

    # Robber scoring base (increased)
    ROBBER_BASE_SCORE = 80.0

    # Settlement target in early game
    TARGET_SETTLEMENTS_EARLY = 3

    # Epsilon-greedy randomness to avoid predictability
    EPSILON_GREEDY = 0.04

    # Rollout bonuses for the very first rollout step
    ROLLOUT_SETTLEMENT_BONUS = 1.7
    ROLLOUT_ROAD_BONUS = 1.4

    # Tie tolerance
    TOLERANCE = 1e-6

    def __init__(self, name: Optional[str] = None):
        super().__init__(Color.BLUE, name)
        # Try to cache a base value function from adapters
        try:
            self._value_fn = base_fn()
            self.debug_print("FooPlayer: Using adapters.base_fn() for evaluation")
        except Exception as e:
            self._value_fn = None
            self.debug_print("FooPlayer: adapters.base_fn() not available, will use heuristic. Error:", e)

        # Diagnostic counters (quiet unless DEBUG)
        self._diag_forced_settlement = 0
        self._diag_forced_road = 0

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
    def get_game_phase(self, game: Game, color: Optional[Color] = None) -> str:
        """Return 'EARLY', 'MID', or 'LATE' based on turn counters or VP thresholds.

        Order of checks:
        1) turn/tick counters if available (preferred)
        2) max VP among players
        3) fallback to conservative MID
        """
        try:
            state = getattr(game, "state", game)
            turn_count = (
                getattr(state, "turn", None)
                or getattr(state, "tick", None)
                or getattr(state, "turn_count", None)
                or getattr(state, "tick_count", None)
            )
            if isinstance(turn_count, (int, float)):
                tc = int(turn_count)
                if tc < self.EARLY_TURN_THRESHOLD:
                    return "EARLY"
                if tc < self.MID_TURN_THRESHOLD:
                    return "MID"
                return "LATE"
        except Exception:
            pass

        # Fallback: use maximum VP among players
        try:
            state = getattr(game, "state", game)
            players = getattr(state, "players", None) or getattr(game, "players", None) or []
            max_vp = 0
            if isinstance(players, dict):
                for p in players.values():
                    vp = getattr(p, "victory_points", None) or getattr(p, "vp", None) or 0
                    try:
                        vp = int(vp)
                    except Exception:
                        vp = 0
                    max_vp = max(max_vp, vp)
            else:
                for p in players:
                    vp = getattr(p, "victory_points", None) or getattr(p, "vp", None) or 0
                    try:
                        vp = int(vp)
                    except Exception:
                        vp = 0
                    max_vp = max(max_vp, vp)
            if max_vp < 4:
                return "EARLY"
            if max_vp < 8:
                return "MID"
            return "LATE"
        except Exception:
            # Conservative fallback to MID
            return "MID"

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
        phase = self.get_game_phase(game, color)
        mults = self.MULTS.get(phase, self.MULTS["MID"])
        settlement_mul = mults["settlement"]
        road_mul = mults["road"]
        city_mul = mults["city"]
        dev_mul = mults["dev"]

        # Adjust production weight by phase
        prod_weight = 80.0 if phase == "EARLY" else 45.0 if phase == "MID" else 30.0

        # Compose weighted sum (city reward scaled by city_mul)
        score = (
            float(vp) * 100.0
            + float(settlements) * 25.0 * settlement_mul
            + float(cities) * 60.0 * city_mul
            + float(roads) * 6.0 * road_mul
            + float(dev_vp) * 50.0
            + float(resources_total) * 1.0
            + float(resource_diversity) * 3.0
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
    def _get_player_state(self, game: Game, color: Color) -> Any:
        """Return the player_state object from the game state (best-effort)."""
        try:
            state = getattr(game, "state", game)
            players = getattr(state, "players", None) or getattr(game, "players", None)
            if isinstance(players, dict):
                return players.get(color) or players.get(str(color))
            elif isinstance(players, (list, tuple)):
                for p in players:
                    if getattr(p, "color", None) == color or getattr(p, "color", None) == str(color):
                        return p
        except Exception:
            return None
        return None

    def settlement_potential(self, action: Any, game: Game, color: Color) -> float:
        """Estimate benefit of a settlement action: new resource types and production.

        Best-effort: try to parse adjacent hexes from action or fallback to string heuristics.
        """
        bonus = 0.0
        try:
            name = self._safe_action_name(action)
            # Quick check: if action indicates a settlement, give base
            if any(tok in name for tok in ("build_settlement", "build_sett", "settle")):
                bonus += 5.0

            # Try to parse a vertex index from the action string
            digits = [int(tok) for tok in name.split() if tok.isdigit()]
            vertex = digits[0] if digits else None

            state = getattr(game, "state", game)
            board = getattr(state, "board", None) or getattr(game, "board", None)
            hexes = getattr(board, "hexes", None) or getattr(board, "tiles", None) or []

            # Player's current resource types
            player_state = self._get_player_state(game, color)
            player_types = set()
            try:
                settlements_list = getattr(player_state, "settlements", None) or getattr(player_state, "settle_locations", None) or []
                if isinstance(settlements_list, (list, tuple)):
                    for s in settlements_list:
                        for h in hexes:
                            neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                            if s in neighbors:
                                rtype = getattr(h, "resource", None) or getattr(h, "type", None)
                                if rtype is not None:
                                    player_types.add(str(rtype).lower())
            except Exception:
                player_types = set()

            # Adjacent resources for proposed vertex
            adj_resources = set()
            prod_sum = 0.0
            die_prob = {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36}
            if vertex is not None:
                for h in hexes:
                    try:
                        neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                        if vertex in neighbors:
                            rtype = getattr(h, "resource", None) or getattr(h, "type", None)
                            if rtype is not None:
                                adj_resources.add(str(rtype).lower())
                            num = getattr(h, "roll", None) or getattr(h, "number", None) or getattr(h, "value", None)
                            try:
                                num = int(num)
                            except Exception:
                                num = None
                            if num in die_prob:
                                prod_sum += die_prob[num]
                    except Exception:
                        continue
            # New types
            new_types = adj_resources - player_types
            bonus += float(len(new_types)) * 12.0
            bonus += float(prod_sum) * 8.0
        except Exception:
            pass
        return float(bonus)

    def road_connection_potential(self, action: Any, game: Game, color: Color) -> float:
        """Estimate if a road action helps expansion. Best-effort using indices."""
        bonus = 0.0
        try:
            name = self._safe_action_name(action)
            # try to extract numbers from action name
            digits = [int(tok) for tok in name.split() if tok.isdigit()]
            # player's settlement/city vertices
            player_state = self._get_player_state(game, color)
            player_nodes = set()
            try:
                settles = getattr(player_state, "settlements", None) or getattr(player_state, "settle_locations", None) or []
                cities = getattr(player_state, "cities", None) or getattr(player_state, "city_locations", None) or []
                if isinstance(settles, (list, tuple)):
                    player_nodes.update(settles)
                if isinstance(cities, (list, tuple)):
                    player_nodes.update(cities)
            except Exception:
                player_nodes = set()

            if digits:
                # if any digit matches a player node, give higher bonus
                if any(d in player_nodes for d in digits):
                    bonus += 6.0
                else:
                    bonus += 3.0
            else:
                # fallback string heuristics
                if "build_road" in name or ("road" in name and "build" in name):
                    bonus += 2.0
        except Exception:
            pass
        return float(bonus)

    def evaluate_buy_dev_card(self, action: Any, game: Game, color: Color) -> bool:
        """Decide whether buying a dev card is currently a good idea (best-effort)."""
        try:
            player_state = self._get_player_state(game, color)
            resources = getattr(player_state, "resources", None)
            if isinstance(resources, dict):
                ore = resources.get("ore", 0) + resources.get("metal", 0)
                wheat = resources.get("wheat", 0) + resources.get("grain", 0)
                others = sum(v for k, v in resources.items() if k not in ("ore", "metal", "wheat", "grain"))
                # if have ore+wheat+another, prefer dev card; or if no settlement/road/city affordable
                if ore >= 1 and wheat >= 1 and others >= 1:
                    return True
                # fallback: if early game and we have some resources but no settlement potential, allow dev buy
                phase = self.get_game_phase(game, color)
                if phase == "EARLY" and (ore + wheat + others) >= 3:
                    return True
        except Exception:
            pass
        return False

    def build_urgency(self, game: Game, color: Color) -> Tuple[float, float, float]:
        """Return (city_bonus, settlement_bonus, road_bonus) depending on resources and phase."""
        city_bonus = 0.0
        settlement_bonus = 0.0
        road_bonus = 0.0
        try:
            player_state = self._get_player_state(game, color)
            resources = getattr(player_state, "resources", None) or {}
            if not isinstance(resources, dict):
                # try to coerce
                try:
                    total = sum(resources)
                    resources = {"res": total}
                except Exception:
                    resources = {}

            # simple can_afford_city_soon heuristic
            ore = resources.get("ore", 0) + resources.get("metal", 0)
            wheat = resources.get("wheat", 0) + resources.get("grain", 0)
            settlements_list = getattr(player_state, "settlements", None) or getattr(player_state, "settle_locations", None) or []
            settlements_owned = len(settlements_list) if isinstance(settlements_list, (list, tuple)) else 0

            phase = self.get_game_phase(game, color)
            # If mid/late and can afford city soon, large city urgency
            if phase in ("MID", "LATE") and ore >= 2 and wheat >= 1:
                city_bonus += 40.0
            # If early and lacking settlements target, encourage settlements strongly
            if phase == "EARLY" and settlements_owned < self.TARGET_SETTLEMENTS_EARLY:
                settlement_bonus += 35.0
            # Road potential: check if any road potential exists via road_connection_potential on sample actions
            # We cannot inspect every action here cheaply; provide a conservative small bonus
            road_bonus += 10.0
        except Exception:
            pass
        return city_bonus, settlement_bonus, road_bonus

    def cheap_pre_score(self, action: Any, game: Game, color: Color) -> float:
        """Cheap, fast scoring used to prioritize actions for simulation (phase-aware)."""
        s = 0.0
        name = self._safe_action_name(action)

        phase = self.get_game_phase(game, color)
        mults = self.MULTS.get(phase, self.MULTS["MID"])
        settlement_mul = mults["settlement"]
        road_mul = mults["road"]
        city_mul = mults["city"]
        dev_mul = mults["dev"]

        # urgency bonuses
        city_urgency, sett_urgency, road_urgency = self.build_urgency(game, color)

        # Reward direct VP gains but adjust city bias early
        if any(tok in name for tok in ("build_city",)):
            base_city = max(50.0, 100.0 * city_mul - 15.0)
            # penalize city if early and still below settlement target
            try:
                player_state = self._get_player_state(game, color)
                settles = getattr(player_state, "settlements", None) or getattr(player_state, "settle_locations", None) or []
                curr_settlements = len(settles) if isinstance(settles, (list, tuple)) else 0
                if phase == "EARLY" and curr_settlements < self.TARGET_SETTLEMENTS_EARLY:
                    base_city *= 0.6
            except Exception:
                pass
            s += base_city + city_urgency

        if any(tok in name for tok in ("build_settlement", "build_sett")):
            s += 90.0 * settlement_mul
            # add settlement potential (resource diversity / production)
            s += self.settlement_potential(action, game, color) * (1.0 if phase != "EARLY" else settlement_mul)
            s += sett_urgency

        if "buy_dev" in name or "buycard" in name or "buy_dev_card" in name:
            buy_pref = self.evaluate_buy_dev_card(action, game, color)
            if buy_pref:
                s += 30.0 * dev_mul + 30.0
            else:
                s += 20.0 * dev_mul

        if "build_road" in name or ("road" in name and "build" in name):
            s += 20.0 * road_mul
            s += self.road_connection_potential(action, game, color) * (1.0 if phase != "EARLY" else road_mul)
            s += road_urgency

        if "knight" in name or "play_knight" in name:
            # raise baseline but let evaluate_play_knight add army/steal bonuses
            s += 70.0
            s += self.evaluate_play_knight(action, game, color)

        if "robber" in name or "move_robber" in name:
            s += 50.0
            s += self.evaluate_robber_action(action, game, color)

        if "trade" in name or "offer_trade" in name:
            s += 10.0

        # Encourage hitting settlement target early
        try:
            player_state = self._get_player_state(game, color)
            curr_settlements = 0
            settles = getattr(player_state, "settlements", None) or getattr(player_state, "settle_locations", None) or []
            if isinstance(settles, (list, tuple)):
                curr_settlements = len(settles)
            if phase == "EARLY" and curr_settlements < self.TARGET_SETTLEMENTS_EARLY and any(tok in name for tok in ("build_settlement", "build_sett")):
                s += 30.0
        except Exception:
            pass

        # small settlement/road potentials for other actions
        if not any(tok in name for tok in ("build_settlement", "build_sett")):
            s += self.settlement_potential(action, game, color) * 0.1
        if not any(tok in name for tok in ("build_road",)):
            s += self.road_connection_potential(action, game, color) * 0.1

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
        phase = self.get_game_phase(game, color)

        musts = []
        others = []
        found_settlement = None
        found_road = None
        for a in all_actions:
            name = self._safe_action_name(a)
            if any(tok in name for tok in self.MUST_INCLUDE_TOKENS):
                if a not in musts:
                    musts.append(a)
            else:
                others.append(a)
            if found_settlement is None and any(tok in name for tok in ("build_settlement", "build_sett", "settle")):
                found_settlement = a
            if found_road is None and any(tok in name for tok in ("build_road", "road")):
                found_road = a

        # Phase-based forced includes: ensure at least one settlement and one road action if present in EARLY
        if phase == "EARLY":
            if found_settlement is not None and found_settlement not in musts:
                musts.append(found_settlement)
                self._diag_forced_settlement += 1
            if found_road is not None and found_road not in musts:
                musts.append(found_road)
                self._diag_forced_road += 1

        # Include recommended dev-card buys if conservative
        for a in all_actions:
            name = self._safe_action_name(a)
            if any(tok in name for tok in ("buy_dev", "buycard", "buy_dev_card")):
                try:
                    if self.evaluate_buy_dev_card(a, game, color) and a not in musts:
                        # include only if no immediate settlement/road affordable (conservative)
                        musts.append(a)
                except Exception:
                    pass

        # Ensure robber/knight actions are present
        for a in all_actions:
            name = self._safe_action_name(a)
            if any(tok in name for tok in ("robber", "move_robber", "knight", "play_knight")):
                if a not in musts:
                    musts.append(a)

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

        self.debug_print(f"FooPlayer: Prefilter selected {len(candidates)} candidates (musts={len(musts)}, phase={phase})")
        if self.DEBUG and phase == "EARLY":
            self.debug_print(f"  Forced includes: settlement={'yes' if found_settlement else 'no'}, road={'yes' if found_road else 'no'}")
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
            name = self._safe_action_name(action)
            # Try to parse a target hex id
            digits = [int(tok) for tok in name.split() if tok.isdigit()]
            target = digits[0] if digits else None

            # Die probabilities
            die_prob = {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36}

            state = getattr(game, "state", game)
            board = getattr(state, "board", None) or getattr(game, "board", None)
            hexes = getattr(board, "hexes", None) or getattr(board, "tiles", None) or []

            # Map hex identifier to object (best-effort: use index or id)
            hex_map = {}
            for idx, h in enumerate(hexes):
                try:
                    hid = getattr(h, "id", None) or getattr(h, "index", None) or idx
                except Exception:
                    hid = idx
                try:
                    key = int(hid) if isinstance(hid, int) or (isinstance(hid, str) and hid.isdigit()) else idx
                except Exception:
                    key = idx
                hex_map[key] = h

            # Compute production loss on opponents
            opponents = []
            players = getattr(state, "players", None) or getattr(game, "players", None) or []
            if isinstance(players, dict):
                for k, p in players.items():
                    # skip self
                    if k == color or getattr(p, "color", None) == color:
                        continue
                    opponents.append(p)
            else:
                for p in players:
                    if getattr(p, "color", None) == color or p == color:
                        continue
                    opponents.append(p)

            total_prod_loss = 0.0
            steal_expected = 0.0
            resource_value = {"ore": 3.0, "metal": 3.0, "wheat": 3.0, "grain": 3.0, "brick": 2.0, "lumber": 2.0, "wood": 2.0, "sheep": 2.0}

            if target in hex_map:
                h = hex_map[target]
                num = getattr(h, "roll", None) or getattr(h, "number", None) or getattr(h, "value", None)
                try:
                    num = int(num)
                except Exception:
                    num = None
                prob = die_prob.get(num, 0)
                # For each opponent, check their settlements/cities adjacent to this hex
                for opp in opponents:
                    opp_settles = getattr(opp, "settlements", None) or getattr(opp, "settle_locations", None) or []
                    opp_cities = getattr(opp, "cities", None) or getattr(opp, "city_locations", None) or []
                    mult = 0.0
                    try:
                        for s in opp_settles:
                            neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                            if s in neighbors:
                                mult += 1.0
                        for c in opp_cities:
                            neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                            if c in neighbors:
                                mult += 2.0
                    except Exception:
                        continue
                    total_prod_loss += prob * mult
                    # Estimate steal expected value as portion of opponent resources times average resource value
                    try:
                        opp_resources = getattr(opp, "resources", None) or {}
                        if isinstance(opp_resources, dict) and opp_resources:
                            total_res = sum(opp_resources.values())
                            if total_res > 0:
                                avg_val = sum(resource_value.get(r, 1.5) * (opp_resources.get(r, 0) / total_res) for r in opp_resources)
                                steal_expected += avg_val * 0.5
                    except Exception:
                        pass

            # Aggressive scaling per latest tuning
            score += total_prod_loss * 40.0
            score += steal_expected * 12.0
            # Extra bonus if multiple opponent cities affected
            try:
                if target in hex_map:
                    h = hex_map[target]
                    city_count = 0
                    for opp in opponents:
                        for c in getattr(opp, "cities", []) or getattr(opp, "city_locations", []) or []:
                            neighbors = getattr(h, "vertices", None) or getattr(h, "adjacent_vertices", None) or []
                            if c in neighbors:
                                city_count += 1
                    if city_count > 0:
                        score += 20.0 * city_count
            except Exception:
                pass

        except Exception:
            pass
        return float(score)

    def evaluate_play_knight(self, action: Any, game: Game, color: Color) -> float:
        """Estimate the value of playing a knight (best-effort)."""
        score = 20.0
        try:
            name = self._safe_action_name(action)
            if "steal" in name or "rob" in name:
                score += 10.0

            # army progress
            player_state = self._get_player_state(game, color)
            army = getattr(player_state, "army", None) or getattr(player_state, "army_size", None) or getattr(player_state, "knights_played", None) or 0
            try:
                army = int(army)
            except Exception:
                army = 0

            # try detect largest army threshold (best-effort fallback to 3)
            largest_threshold = 3
            try:
                # if available, find other players' armies
                state = getattr(game, "state", game)
                players = getattr(state, "players", None) or getattr(game, "players", None) or []
                max_other = 0
                if isinstance(players, dict):
                    for k, p in players.items():
                        if getattr(p, "color", None) == color or k == color:
                            continue
                        other_army = getattr(p, "army", None) or getattr(p, "army_size", None) or getattr(p, "knights_played", None) or 0
                        try:
                            other_army = int(other_army)
                        except Exception:
                            other_army = 0
                        max_other = max(max_other, other_army)
                else:
                    for p in players:
                        if getattr(p, "color", None) == color:
                            continue
                        other_army = getattr(p, "army", None) or getattr(p, "army_size", None) or getattr(p, "knights_played", None) or 0
                        try:
                            other_army = int(other_army)
                        except Exception:
                            other_army = 0
                        max_other = max(max_other, other_army)
                largest_threshold = max(3, max_other + 1)
            except Exception:
                largest_threshold = 3

            if army + 1 >= largest_threshold:
                score += 40.0
            else:
                score += 20.0
        except Exception:
            pass
        return float(score)

    # ------------------- Helper: determine active player color -------------------
    def _get_active_player_color(self, game: Game) -> Optional[Color]:
        """Best-effort to detect which Color is to move in the given game state."""
        try:
            state = getattr(game, "state", game)
            cp = getattr(state, "current_player", None) or getattr(state, "active_player", None) or getattr(state, "turn_color", None)
            if cp is None:
                cp = getattr(game, "current_player", None)
            # cp might be index, player object, or Color
            if isinstance(cp, Color):
                return cp
            if isinstance(cp, int):
                players = getattr(state, "players", None) or getattr(game, "players", None) or []
                try:
                    if isinstance(players, (list, tuple)) and 0 <= cp < len(players):
                        return getattr(players[cp], "color", None)
                except Exception:
                    pass
            # If cp is a player object
            if hasattr(cp, "color"):
                return getattr(cp, "color")

            # Fallback: pick first player in players whose color != our color
            players = getattr(state, "players", None) or getattr(game, "players", None) or []
            my_color = self._get_player_color()
            if isinstance(players, dict):
                for k, p in players.items():
                    try:
                        c = getattr(p, "color", None) or k
                        if c != my_color:
                            return c
                    except Exception:
                        continue
            else:
                for p in players:
                    try:
                        c = getattr(p, "color", None)
                        if c != my_color:
                            return c
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    # ------------------- Rollout logic with opponent-response -------------------
    def rollout_value(self, game: Game, color: Color, depth: int, initial: bool = True) -> float:
        """Short greedy rollout with phase bias and light opponent-response.

        initial: True for the first step of rollout so we can bias toward expansion early.
        """
        try:
            if depth <= 0:
                return self._evaluate_game_state(game, color)

            actions = self.get_playable_actions_from_game(game)
            if not actions:
                return self._evaluate_game_state(game, color)

            phase = self.get_game_phase(game, color)

            def score_for_rollout(a, g, c, is_initial):
                base = self.cheap_pre_score(a, g, c)
                if is_initial and phase == "EARLY":
                    name = self._safe_action_name(a)
                    if any(tok in name for tok in ("build_settlement", "build_sett", "settle")):
                        base *= self.ROLLOUT_SETTLEMENT_BONUS
                    if any(tok in name for tok in ("build_road", "road")):
                        base *= self.ROLLOUT_ROAD_BONUS
                return base

            sorted_actions = sorted(actions, key=lambda a: score_for_rollout(a, game, color, initial), reverse=True)

            # Try top actions to simulate
            for a in sorted_actions[:6]:
                branches = []
                try:
                    branches = execute_deterministic(game, a)
                except Exception:
                    try:
                        branches = execute_spectrum(game, a)
                    except Exception:
                        branches = []
                if not branches:
                    continue
                # pick the most probable branch
                next_game = max(branches, key=lambda bp: float(bp[1]))[0]

                # Light opponent-response: if opponent to move next, simulate their greedy action once
                opp_color = self._get_active_player_color(next_game)
                my_color = color
                if opp_color is not None and opp_color != my_color and depth >= 2:
                    try:
                        opp_actions = self.get_playable_actions_from_game(next_game)
                        if opp_actions:
                            # filter out robber/knight for opponent response unless all are robber/knight
                            non_disrupt = [oa for oa in opp_actions if not any(tok in self._safe_action_name(oa) for tok in ("knight", "robber", "move_robber"))]
                            candidate_ops = non_disrupt if non_disrupt else opp_actions
                            # pick opponent best action by cheap_pre_score from their perspective
                            best_opp = max(candidate_ops, key=lambda oa: self.cheap_pre_score(oa, next_game, opp_color))
                            # simulate opponent action deterministically if possible
                            opp_branches = []
                            try:
                                opp_branches = execute_deterministic(next_game, best_opp)
                            except Exception:
                                try:
                                    opp_branches = execute_spectrum(next_game, best_opp)
                                except Exception:
                                    opp_branches = []
                            if opp_branches:
                                next_game = max(opp_branches, key=lambda bp: float(bp[1]))[0]
                    except Exception:
                        pass

                return self.rollout_value(next_game, color, depth - 1, initial=False)

            # fallback: try any action that simulates
            for a in sorted_actions[:10]:
                branches = []
                try:
                    branches = execute_deterministic(game, a)
                except Exception:
                    try:
                        branches = execute_spectrum(game, a)
                    except Exception:
                        branches = []
                if branches:
                    next_game = max(branches, key=lambda bp: float(bp[1]))[0]
                    return self.rollout_value(next_game, color, depth - 1, initial=False)

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

        # Quick boosts for robber/knight/dev before heavy sim
        name = self._safe_action_name(action)
        preboost = 0.0
        try:
            if any(tok in name for tok in ("move_robber", "robber")):
                preboost += self.evaluate_robber_action(action, game, color)
            if any(tok in name for tok in ("knight", "play_knight")):
                preboost += self.evaluate_play_knight(action, game, color)
            if any(tok in name for tok in ("buy_dev", "buycard", "buy_dev_card")):
                if self.evaluate_buy_dev_card(action, game, color):
                    preboost += 20.0
        except Exception:
            preboost += 0.0

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
        rollout_depth = max(0, self.ROLLOUT_DEPTH - 1)
        for (out_game, prob) in branches:
            try:
                immediate = self._evaluate_game_state(out_game, color)
                rollout_est = self.rollout_value(out_game, color, rollout_depth, initial=True)
                branch_val = 0.6 * immediate + 0.4 * rollout_est
            except Exception as e:
                self.debug_print("FooPlayer: evaluation failed for branch, using heuristic. Error:", e)
                branch_val = self._heuristic_value(out_game, color)
            expected += float(prob) * float(branch_val)
            total_prob += float(prob)

        if total_prob > 0:
            expected = expected / total_prob

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
            phase = self.get_game_phase(game, color)

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

                if score > best_score + self.TOLERANCE:
                    best_score = score
                    best_actions = [a]
                elif abs(score - best_score) <= self.TOLERANCE:
                    best_actions.append(a)

            # If no action had a finite score, fallback to first playable action
            if not best_actions:
                self.debug_print("FooPlayer: All evaluations failed, defaulting to first playable action")
                return playable_actions[0]

            # Epsilon-greedy randomness to reduce predictability
            chosen: Any
            scores_debug.sort(key=lambda x: x[0], reverse=True)
            if random.random() < self.EPSILON_GREEDY and len(scores_debug) >= 2:
                # pick from top-3 weighted by score (or fewer if not available)
                top_k = scores_debug[: min(3, len(scores_debug))]
                weights = [max(0.0, s - top_k[-1][0] + 1e-6) for (s, a) in top_k]
                total_w = sum(weights)
                if total_w > 0:
                    r = random.random() * total_w
                    cum = 0.0
                    for w, (_s, a) in zip(weights, top_k):
                        cum += w
                        if r <= cum:
                            chosen = a
                            break
                    else:
                        chosen = top_k[0][1]
                else:
                    chosen = scores_debug[0][1]
                if self.DEBUG:
                    self.debug_print(f"FooPlayer: EPSILON pick triggered, chosen alternate action {chosen}")
                return chosen

            # If tie, break ties preferring settlement/road/resource diversity improvements
            if len(best_actions) > 1:
                tie_metrics = []
                for a in best_actions:
                    try:
                        metric = 0.0
                        metric += self.settlement_potential(a, game, color)
                        metric += self.road_connection_potential(a, game, color)
                        # small production proxy via heuristic
                        metric += 0.01 * self._heuristic_value(game, color)
                        tie_metrics.append((metric, a))
                    except Exception:
                        tie_metrics.append((0.0, a))
                tie_metrics.sort(key=lambda x: x[0], reverse=True)
                # pick the top metric actions (could still be multiple)
                top_metric = tie_metrics[0][0]
                filtered = [a for (m, a) in tie_metrics if abs(m - top_metric) <= self.TOLERANCE]
                if filtered:
                    chosen = random.choice(filtered)
                else:
                    chosen = random.choice(best_actions)
            else:
                chosen = best_actions[0]

            # Debug logging: phase and top candidates
            if self.DEBUG:
                self.debug_print(f"FooPlayer: Phase={phase}, SettlementsTarget={self.TARGET_SETTLEMENTS_EARLY}")
                topn = scores_debug[:3]
                self.debug_print("FooPlayer: Top candidates:")
                for sc, act in topn:
                    self.debug_print(f"  score={sc:.2f} action={act}")

            self.debug_print(f"FooPlayer: Chosen action {chosen} with expected score {best_score}")
            return chosen
        except Exception as e:
            # Protect against unexpected errors
            print("FooPlayer: Unexpected error in decide(), defaulting to first playable action. Error:", e)
            try:
                return list(playable_actions)[0]
            except Exception:
                return None