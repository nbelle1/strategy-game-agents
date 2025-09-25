import random
import traceback
from typing import Iterable, List, Optional, Tuple, Dict, Any

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
    # Parameters
    MAX_ACTIONS_TO_EVAL = 80
    SAMPLE_PER_ACTION_TYPE = 4
    SPECTRUM_MAX_OUTCOMES = 8
    EARLY_TURN_THRESHOLD = 30

    TOP_K_1PLY = 6
    OP_MAX_ACTIONS = 10
    OP_SAMPLE_PER_ACTION_TYPE = 2

    MAX_SIMULATION_NODES = 4000
    MIN_EVAL_SUCCESS_RATE_FOR_2PLY = 0.80
    MIN_SPECTRUM_SUCCESS_RATE = 0.60
    SCORE_AMBIGUITY_THRESHOLD = 0.05

    SELF_LOOKAHEAD_DEPTH = 3
    SELF_LOOKAHEAD_BUDGET = 200

    ROAD_ROLLOUTS = 20
    ROAD_ROLLOUT_DEPTH = 6
    ROAD_ROLLOUT_SIM_BUDGET = 600
    ROAD_ROLLOUT_CANDIDATES = 4
    ROAD_SETTLEMENT_PROB_THRESHOLD = 0.20

    RNG_SEED = 0

    def __init__(self, name: Optional[str] = None):
        super().__init__(Color.BLUE, name)
        self.debug = False
        try:
            self._value_fn = base_fn()
        except Exception:
            self._value_fn = None
        self._diag: Dict[str, int] = {
            "n_candidates": 0,
            "n_eval_attempts": 0,
            "n_eval_success": 0,
            "n_spectrum_calls": 0,
            "n_spectrum_success": 0,
            "n_det_calls": 0,
            "n_det_success": 0,
            "n_skipped": 0,
            "n_fallbacks_to_first_action": 0,
            "n_2ply_runs": 0,
            "n_2ply_skipped": 0,
            "n_road_candidates_included": 0,
            "simulated_nodes_total": 0,
            "n_road_rollouts_run": 0,
        }

    def _stable_color_hash(self, color: Color) -> int:
        try:
            return sum(ord(c) for c in str(color)) & 0xFFFFFFFF
        except Exception:
            return 0

    def _action_type_key(self, action) -> str:
        k = getattr(action, "action_type", None)
        if k is not None:
            return str(k)
        for attr in ("type", "name"):
            v = getattr(action, attr, None)
            if v is not None:
                return str(v)
        try:
            return action.__class__.__name__
        except Exception:
            return str(action)

    def _is_build_or_upgrade(self, action) -> bool:
        at = getattr(action, "action_type", None)
        try:
            return at in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD}
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            return any(x in str(name).lower() for x in ("build", "settle", "city", "road", "upgrade"))

    def _is_robber_or_chance(self, action) -> bool:
        at = getattr(action, "action_type", None)
        try:
            return at in {ActionType.PLAY_DEV_CARD, ActionType.PLACE_ROBBER, ActionType.DRAW_DEV_CARD}
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            return any(x in str(name).lower() for x in ("robber", "dev", "draw"))

    def _is_road_action(self, action) -> bool:
        at = getattr(action, "action_type", None)
        try:
            return at == ActionType.BUILD_ROAD
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            return "road" in str(name).lower()

    def _is_settlement_build(self, action) -> bool:
        at = getattr(action, "action_type", None)
        try:
            return at == ActionType.BUILD_SETTLEMENT
        except Exception:
            name = getattr(action, "name", None) or getattr(action, "type", None) or action.__class__.__name__
            return "settle" in str(name).lower()

    def _get_visible_vp(self, game: Game, my_color: Color) -> int:
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
        actions = list(playable_actions)
        n = len(actions)
        if n <= self.MAX_ACTIONS_TO_EVAL:
            return actions
        current_turn = getattr(game, "current_turn", None)
        if current_turn is None:
            current_turn = getattr(game, "tick", 0)
        early_game = (current_turn <= self.EARLY_TURN_THRESHOLD)
        mid_game = (self.EARLY_TURN_THRESHOLD < current_turn <= 2 * self.EARLY_TURN_THRESHOLD)
        groups: Dict[str, List] = {}
        for a in actions:
            key = self._action_type_key(a)
            groups.setdefault(key, []).append(a)
        rng = random.Random(self.RNG_SEED + self._stable_color_hash(self.color))
        sampled: List = []
        for key in sorted(groups.keys()):
            group = list(groups[key])
            sample_count = self.SAMPLE_PER_ACTION_TYPE
            try:
                if early_game and any(self._is_build_or_upgrade(a) for a in group):
                    sample_count += 1
                elif mid_game and any(self._is_road_action(a) for a in group):
                    sample_count += 1
            except Exception:
                pass
            rng.shuffle(group)
            take = min(sample_count, len(group))
            sampled.extend(group[:take])
            if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                break
        if len(sampled) < self.MAX_ACTIONS_TO_EVAL:
            for a in actions:
                if a not in sampled:
                    sampled.append(a)
                    if len(sampled) >= self.MAX_ACTIONS_TO_EVAL:
                        break
        if self.debug:
            phase = "early" if early_game else ("mid" if mid_game else "late")
            print(f"_sample_actions: phase={phase}, pruned {n} -> {len(sampled)} actions (cap={self.MAX_ACTIONS_TO_EVAL})")
        return sampled

    def _sample_opponent_actions(self, playable_actions: Iterable, game: Game, opponent_color: Color) -> List:
        actions = list(playable_actions)
        n = len(actions)
        if n <= self.OP_MAX_ACTIONS:
            return actions
        current_turn = getattr(game, "current_turn", None)
        if current_turn is None:
            current_turn = getattr(game, "tick", 0)
        early_game = (current_turn <= self.EARLY_TURN_THRESHOLD)
        groups: Dict[str, List] = {}
        for a in actions:
            key = self._action_type_key(a)
            groups.setdefault(key, []).append(a)
        rng = random.Random(self.RNG_SEED + self._stable_color_hash(opponent_color))
        sampled: List = []
        for key in sorted(groups.keys()):
            group = list(groups[key])
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
        try:
            lst = list(spectrum)
            if not lst:
                return []
            try:
                sorted_lst = sorted(lst, key=lambda x: float(x[1]) if len(x) > 1 else 0.0, reverse=True)
            except Exception:
                sorted_lst = lst
            capped = sorted_lst[:cap]
            games = []
            probs = []
            for entry in capped:
                try:
                    g, p = entry
                except Exception:
                    continue
                games.append(g)
                probs.append(float(p))
            if not games:
                return []
            total = sum(probs)
            if total > 0:
                return [(g, p / total) for g, p in zip(games, probs)]
            else:
                n = len(games)
                return [(g, 1.0 / n) for g in games]
        except Exception:
            if self.debug:
                print("_normalize_and_cap_spectrum: failed")
                traceback.print_exc()
            return []

    def _determine_opponent_color(self, game: Game, my_color: Color) -> Color:
        try:
            cur = getattr(game, "current_player", None)
            if cur is not None and cur != my_color:
                return cur
        except Exception:
            pass
        try:
            colors = [c for c in list(Color)]
            for c in colors:
                if c != my_color:
                    return c
        except Exception:
            pass
        return my_color

    def _derive_opponent_actions(self, game: Game, opponent_color: Color) -> List:
        try:
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
        return []

    def _safe_eval_base_fn(self, g: Game, color: Color) -> Optional[float]:
        try:
            if self._value_fn is not None:
                return float(self._value_fn(g, color))
        except Exception:
            if self.debug:
                print("_safe_eval_base_fn: _value_fn failed")
                traceback.print_exc()
        try:
            vf = base_fn()
            try:
                return float(vf(g, color))
            except Exception:
                if self.debug:
                    print("_safe_eval_base_fn: vf(g,color) failed")
                    traceback.print_exc()
        except Exception:
            pass
        try:
            return float(base_fn(g, color))
        except Exception:
            if self.debug:
                print("_safe_eval_base_fn: base_fn(g,color) failed")
                traceback.print_exc()
            return None

    def _simulate_action_branches(self, game: Game, action) -> List[Tuple[Game, float]]:
        try:
            game_copy = copy_game(game)
        except Exception:
            if self.debug:
                print("_simulate_action_branches: copy_game failed")
                traceback.print_exc()
            return []
        outcomes: List[Tuple[Game, float]] = []
        try:
            if self._is_robber_or_chance(action):
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
            else:
                det_res = execute_deterministic(game_copy, action)
                if det_res:
                    normalized: List[Tuple[Game, float]] = []
                    for entry in det_res[: self.SPECTRUM_MAX_OUTCOMES]:
                        try:
                            g, p = entry
                        except Exception:
                            g = entry
                            p = 1.0
                        normalized.append((g, float(p)))
                    total_p = sum(p for _, p in normalized)
                    if total_p > 0:
                        outcomes = [(g, p / total_p) for (g, p) in normalized]
                    else:
                        n = len(normalized)
                        if n > 0:
                            outcomes = [(g, 1.0 / n) for (g, _) in normalized]
        except Exception:
            if self.debug:
                print("_simulate_action_branches: failed to simulate")
                traceback.print_exc()
            return []
        return outcomes

    def _evaluate_action(self, game: Game, action, my_color: Color) -> Optional[Tuple[float, float]]:
        self._diag["n_eval_attempts"] = self._diag.get("n_eval_attempts", 0) + 1
        def safe_eval_fn(g: Game) -> Optional[float]:
            return self._safe_eval_base_fn(g, my_color)
        def get_vp(g: Game) -> float:
            try:
                return float(self._get_visible_vp(g, my_color))
            except Exception:
                if self.debug:
                    print("_evaluate_action: _get_visible_vp failed")
                    traceback.print_exc()
                return 0.0
        try:
            game_copy = copy_game(game)
        except Exception:
            if self.debug:
                print("_evaluate_action: copy_game failed")
                traceback.print_exc()
            self._diag["n_skipped"] = self._diag.get("n_skipped", 0) + 1
            return None
        try:
            vp_orig = get_vp(game)
        except Exception:
            vp_orig = 0.0
        if self._is_robber_or_chance(action):
            try:
                self._diag["n_spectrum_calls"] = self._diag.get("n_spectrum_calls", 0) + 1
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
                            sc = safe_eval_fn(og)
                            if sc is None:
                                continue
                            any_scored = True
                            vp_out = get_vp(og)
                            weighted_score += prob * sc
                            weighted_vp_delta += prob * (vp_out - vp_orig)
                        if any_scored:
                            self._diag["n_spectrum_success"] = self._diag.get("n_spectrum_success", 0) + 1
                            self._diag["n_eval_success"] = self._diag.get("n_eval_success", 0) + 1
                            return (float(weighted_score), float(weighted_vp_delta))
            except Exception:
                if self.debug:
                    print("_evaluate_action: spectrum failed")
                    traceback.print_exc()
        try:
            self._diag["n_det_calls"] = self._diag.get("n_det_calls", 0) + 1
            res = execute_deterministic(game_copy, action)
        except Exception:
            if self.debug:
                print("_evaluate_action: execute_deterministic failed")
                traceback.print_exc()
            self._diag["n_skipped"] = self._diag.get("n_skipped", 0) + 1
            return None
        try:
            if not res:
                resultant_game = game_copy
            else:
                first = res[0]
                if isinstance(first, tuple) and len(first) >= 1:
                    resultant_game = first[0]
                else:
                    resultant_game = first
            score = safe_eval_fn(resultant_game)
            if score is None:
                self._diag["n_skipped"] = self._diag.get("n_skipped", 0) + 1
                return None
            vp_after = get_vp(resultant_game)
            vp_delta = float(vp_after - vp_orig)
            self._diag["n_eval_success"] = self._diag.get("n_eval_success", 0) + 1
            self._diag["n_det_success"] = self._diag.get("n_det_success", 0) + 1
            return (float(score), float(vp_delta))
        except Exception:
            if self.debug:
                print("_evaluate_action: normalize/eval failed")
                traceback.print_exc()
            self._diag["n_skipped"] = self._diag.get("n_skipped", 0) + 1
            return None

    def _compute_expansion_potential(self, game: Game, action) -> float:
        try:
            game_copy = copy_game(game)
        except Exception:
            return -float("inf")
        outcomes = []
        try:
            if self._is_robber_or_chance(action):
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
            else:
                det_res = execute_deterministic(game_copy, action)
                if det_res:
                    normalized = []
                    for entry in det_res[: self.SPECTRUM_MAX_OUTCOMES]:
                        try:
                            g, p = entry
                        except Exception:
                            g = entry
                            p = 1.0
                        normalized.append((g, float(p)))
                    total_p = sum(p for _, p in normalized)
                    if total_p > 0:
                        outcomes = [(g, p / total_p) for (g, p) in normalized]
                    else:
                        n = len(normalized)
                        if n > 0:
                            outcomes = [(g, 1.0 / n) for (g, _) in normalized]
        except Exception:
            return -float("inf")
        if not outcomes:
            return -float("inf")
        total_expansion = 0.0
        for outcome_game, prob in outcomes:
            try:
                playable = self._derive_opponent_actions(outcome_game, self.color)
                expansion = len(playable) if playable else 0
                total_expansion += prob * expansion
            except Exception:
                return -float("inf")
        return total_expansion

    def _compute_expected_settlement_gain(self, game: Game, action) -> float:
        try:
            game_copy = copy_game(game)
        except Exception:
            return -float("inf")
        outcomes = self._simulate_action_branches(game_copy, action)
        if not outcomes:
            return -float("inf")
        total_gain = 0.0
        sim_nodes_used = 0
        for outcome_game, prob in outcomes:
            if sim_nodes_used >= self.SELF_LOOKAHEAD_BUDGET:
                break
            stack = [(outcome_game, 0, 0)]
            best_gain_for_branch = 0
            while stack and sim_nodes_used < self.SELF_LOOKAHEAD_BUDGET:
                state, depth, gained = stack.pop()
                sim_nodes_used += 1
                try:
                    playable = self._derive_opponent_actions(state, self.color) or []
                except Exception:
                    continue
                build_candidates = [act for act in playable if self._is_build_or_upgrade(act) or self._is_road_action(act)]
                for act in self._sample_actions(build_candidates, state)[:5]:
                    try:
                        det = execute_deterministic(copy_game(state), act)
                        if not det:
                            continue
                        first = det[0]
                        if isinstance(first, tuple) and len(first) >= 1:
                            next_state = first[0]
                        else:
                            next_state = first
                    except Exception:
                        continue
                    new_gained = gained + (1 if self._is_settlement_build(act) else 0)
                    if depth + 1 < self.SELF_LOOKAHEAD_DEPTH:
                        stack.append((next_state, depth + 1, new_gained))
                    else:
                        if new_gained > best_gain_for_branch:
                            best_gain_for_branch = new_gained
                if gained > best_gain_for_branch:
                    best_gain_for_branch = gained
            total_gain += prob * best_gain_for_branch
        return float(total_gain)

    def _sample_branch_by_prob(self, branches: List[Tuple[Game, float]], rng: random.Random) -> Optional[Game]:
        if not branches:
            return None
        try:
            total_p = sum(p for _, p in branches)
        except Exception:
            total_p = 0.0
        if total_p <= 0:
            return branches[0][0]
        r = rng.random() * total_p
        cumulative = 0.0
        for g, p in branches:
            cumulative += p
            if r <= cumulative:
                return g
        return branches[-1][0]

    def _get_current_player_color(self, game: Game) -> Color:
        try:
            cur = getattr(game, "current_player", None)
            return cur if cur is not None else self.color
        except Exception:
            return self.color

    def _choose_best_1ply_from_list(self, game: Game, playable: List) -> Optional[Any]:
        best_action = None
        best_score = -float("inf")
        for a in playable:
            try:
                res = self._evaluate_action(game, a, self.color)
            except Exception:
                res = None
            if res is None:
                continue
            sc, _ = res
            if sc > best_score:
                best_action = a
                best_score = sc
        return best_action or (playable[0] if playable else None)

    def _choose_opponent_action_deterministic(self, game: Game, playable: List, opp_color: Color) -> Optional[Any]:
        if not playable:
            return None
        try:
            best_action = None
            best_score = -float("inf")
            for a in playable:
                try:
                    game_copy = copy_game(game)
                    res = execute_deterministic(game_copy, a)
                    if not res:
                        continue
                    first = res[0]
                    outcome = first[0] if isinstance(first, tuple) else first
                    sc = self._safe_eval_base_fn(outcome, opp_color)
                    if sc is not None and sc > best_score:
                        best_action = a
                        best_score = sc
                except Exception:
                    continue
            return best_action or playable[0]
        except Exception:
            return playable[0]

    def _road_rollout_evaluator(self, game: Game, candidate: Any, sim_budget_remaining: int) -> Optional[Tuple[float, float, float, int]]:
        rng = random.Random(self.RNG_SEED + self._stable_color_hash(self.color))
        sims_used = 0
        successful_rollouts = 0
        settlement_count = 0
        roads_total = 0
        vp_total = 0.0
        try:
            base_value = self._safe_eval_base_fn(game, self.color)
        except Exception:
            base_value = None
        for _ in range(self.ROAD_ROLLOUTS):
            if sims_used >= sim_budget_remaining:
                break
            try:
                branches = self._simulate_action_branches(game, candidate)
                if not branches:
                    continue
                outcome_game = self._sample_branch_by_prob(branches, rng)
                if outcome_game is None:
                    continue
            except Exception:
                if self.debug:
                    print("_road_rollout_evaluator: simulate failed")
                    traceback.print_exc()
                continue
            success_this_rollout = False
            state = outcome_game
            roads_built = 0
            settlement_built = False
            for _ in range(self.ROAD_ROLLOUT_DEPTH):
                if sims_used >= sim_budget_remaining:
                    break
                try:
                    current_color = self._get_current_player_color(state)
                    playable = list(self._derive_opponent_actions(state, current_color) or [])
                except Exception:
                    break
                if current_color == self.color:
                    our_choices = [a for a in playable if self._is_road_action(a) or self._is_settlement_build(a)]
                    if our_choices:
                        chosen = rng.choice(our_choices)
                    else:
                        chosen = self._choose_best_1ply_from_list(state, playable)
                else:
                    chosen = self._choose_opponent_action_deterministic(state, playable, current_color)
                try:
                    if self._is_robber_or_chance(chosen):
                        try:
                            spec = execute_spectrum(copy_game(state), chosen)
                            chosen_state = self._sample_branch_by_prob(spec, rng)
                        except Exception:
                            det = execute_deterministic(copy_game(state), chosen)
                            first = det[0] if isinstance(det, (list, tuple)) and det else None
                            chosen_state = first[0] if isinstance(first, tuple) else (first if first is not None else state)
                    else:
                        det = execute_deterministic(copy_game(state), chosen)
                        first = det[0] if isinstance(det, (list, tuple)) and det else None
                        chosen_state = first[0] if isinstance(first, tuple) else (first if first is not None else state)
                except Exception:
                    if self.debug:
                        print("_road_rollout_evaluator: simulation failed during rollout")
                        traceback.print_exc()
                    break
                sims_used += 1
                if current_color == self.color:
                    if self._is_road_action(chosen):
                        roads_built += 1
                    if self._is_settlement_build(chosen):
                        settlement_built = True
                state = chosen_state
                success_this_rollout = True
            if success_this_rollout:
                successful_rollouts += 1
                settlement_count += 1 if settlement_built else 0
                roads_total += roads_built
                if base_value is not None:
                    final_value = self._safe_eval_base_fn(state, self.color) or 0.0
                    vp_total += (final_value - base_value)
        if successful_rollouts == 0:
            return None
        prob_settlement = settlement_count / successful_rollouts
        expected_roads = roads_total / successful_rollouts
        expected_vp = vp_total / successful_rollouts
        return (prob_settlement, expected_roads, expected_vp, sims_used)

    def decide(self, game: Game, playable_actions: Iterable):
        actions = list(playable_actions)
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]
        # reset diag
        for k in list(self._diag.keys()):
            self._diag[k] = 0
        # 1-ply
        candidates = self._sample_actions(actions, game)
        self._diag["n_candidates"] = len(candidates)
        one_ply_results: List[Tuple[Any, float, float]] = []
        eval_fn = getattr(self, "_evaluate_action", None) or getattr(self, "_simulate_and_evaluate", None)
        if eval_fn is None:
            self._diag["n_fallbacks_to_first_action"] += 1
            return actions[0]
        for a in candidates:
            try:
                res = eval_fn(game, a, self.color)
            except Exception:
                if self.debug:
                    print("decide: evaluator exception for", repr(a))
                    traceback.print_exc()
                res = None
            if res is None:
                self._diag["n_skipped"] += 1
                continue
            sc, vpd = res
            one_ply_results.append((a, float(sc), float(vpd)))
        if not one_ply_results:
            self._diag["n_fallbacks_to_first_action"] += 1
            return actions[0]
        # reliability
        eval_success_rate = self._diag.get("n_eval_success", 0) / max(1, self._diag.get("n_eval_attempts", 0))
        spectrum_success_rate = (
            self._diag.get("n_spectrum_success", 0) / max(1, self._diag.get("n_spectrum_calls", 0))
            if self._diag.get("n_spectrum_calls", 0) > 0
            else 1.0
        )
        one_ply_results.sort(key=lambda t: t[1], reverse=True)
        score_gap = one_ply_results[0][1] - one_ply_results[1][1] if len(one_ply_results) > 1 else float("inf")
        candidates_list = [t[0] for t in one_ply_results]
        road_candidates = [a for a in candidates_list if self._is_road_action(a)]
        robber_candidates = [a for a in candidates_list if self._is_robber_or_chance(a)]
        has_high_potential_road = any(self._compute_expansion_potential(game, a) >= 0 for a in road_candidates)
        has_high_potential_robber = any(self._compute_opponent_impact(game, a) >= 0 for a in robber_candidates)
        allow_2ply = (
            (eval_success_rate >= self.MIN_EVAL_SUCCESS_RATE_FOR_2PLY and spectrum_success_rate >= self.MIN_SPECTRUM_SUCCESS_RATE)
            or (score_gap < self.SCORE_AMBIGUITY_THRESHOLD)
            or has_high_potential_road
            or has_high_potential_robber
        )
        if self.debug:
            print(f"decide: eval_success_rate={eval_success_rate:.2f}, spectrum_success_rate={spectrum_success_rate:.2f}, score_gap={score_gap:.3f}, allow_2ply={allow_2ply}")
        if not allow_2ply:
            self._diag["n_2ply_skipped"] += 1
            # return best 1-ply
            best = max(one_ply_results, key=lambda t: (t[1], t[2], repr(t[0])))
            return best[0]
        # Stage 3: rollouts selection
        top_by_1ply = [t[0] for t in one_ply_results[:3]]
        remaining_candidates = [t[0] for t in one_ply_results[3:]]
        candidates_for_rollout = []
        candidates_for_rollout.extend(top_by_1ply)
        road_cands = [a for a in remaining_candidates if self._is_road_action(a)]
        settle_cands = [a for a in remaining_candidates if self._is_settlement_build(a)]
        candidates_for_rollout.extend(road_cands[:2])
        candidates_for_rollout.extend(settle_cands[:2])
        # dedupe cap
        seen = set(); roll_candidates = []
        for a in candidates_for_rollout:
            if a not in seen:
                seen.add(a); roll_candidates.append(a)
            if len(roll_candidates) >= self.ROAD_ROLLOUT_CANDIDATES:
                break
        rollout_metrics: Dict[Any, Tuple[float, float, float]] = {}
        sim_budget_remaining = min(self.ROAD_ROLLOUT_SIM_BUDGET, self.MAX_SIMULATION_NODES - self._diag.get("simulated_nodes_total", 0))
        for a in roll_candidates:
            if sim_budget_remaining <= 0:
                break
            try:
                metrics = self._road_rollout_evaluator(game, a, sim_budget_remaining)
            except Exception:
                if self.debug:
                    print("decide: _road_rollout_evaluator exception for", repr(a))
                    traceback.print_exc()
                metrics = None
            if metrics is not None:
                prob_settlement, expected_roads, expected_vp, sims_used = metrics
                rollout_metrics[a] = (prob_settlement, expected_roads, expected_vp)
                sim_budget_remaining -= sims_used
                self._diag["simulated_nodes_total"] += sims_used
                self._diag["n_road_rollouts_run"] += 1
            else:
                rollout_metrics[a] = (-float("inf"), -float("inf"), -float("inf"))
        # force road inclusion
        best_road_candidate = None; best_road_metrics = (-float("inf"), -float("inf"), -float("inf"))
        for a, metrics in rollout_metrics.items():
            if self._is_road_action(a) and metrics[0] > best_road_metrics[0]:
                best_road_candidate = a; best_road_metrics = metrics
        candidate_pool = [t[0] for t in one_ply_results[:3]]
        # add best settlement/expansion
        # compute settlement gains
        settlement_gain_scores: Dict[Any, float] = {}
        expansion_scores: Dict[Any, float] = {}
        for a in remaining_candidates:
            g = self._compute_expected_settlement_gain(game, a)
            if g != -float("inf"):
                settlement_gain_scores[a] = g
            e = self._compute_expansion_potential(game, a)
            if e != -float("inf"):
                expansion_scores[a] = e
        sorted_remaining = sorted(settlement_gain_scores.items(), key=lambda x: (x[1], expansion_scores.get(x[0], -float("inf"))), reverse=True)
        for a, _ in sorted_remaining[: max(0, self.TOP_K_1PLY - len(candidate_pool))]:
            candidate_pool.append(a)
        if best_road_candidate and best_road_metrics[0] >= self.ROAD_SETTLEMENT_PROB_THRESHOLD and best_road_candidate not in candidate_pool:
            candidate_pool.append(best_road_candidate)
            self._diag["n_road_candidates_included"] += 1
            if self.debug:
                print(f"decide: forced inclusion of road candidate {repr(best_road_candidate)} with prob_settlement={best_road_metrics[0]:.2f}")
        if self.debug:
            print("Candidate pool (with rollout metrics):")
            for a in candidate_pool:
                m = rollout_metrics.get(a, (-1, -1, -1))
                print(f"  {repr(a)} prob_settlement={m[0]:.2f} expected_roads={m[1]:.2f} expected_vp={m[2]:.2f}")
        # Stage 4: conservative adversarial 2-ply
        best_action = None
        best_tuple = None
        sim_count = 0
        SIMULATION_HARD_LIMIT = self.MAX_SIMULATION_NODES
        deep_successful = 0
        for a in candidate_pool:
            if sim_count >= SIMULATION_HARD_LIMIT:
                break
            try:
                game_copy = copy_game(game)
            except Exception:
                if self.debug:
                    print("decide: copy_game failed for", repr(a))
                    traceback.print_exc()
                continue
            # outcomes
            outcomes = []
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
                if not outcomes:
                    det = execute_deterministic(game_copy, a)
                    if not det:
                        continue
                    normalized = []
                    for entry in det[: self.SPECTRUM_MAX_OUTCOMES]:
                        try:
                            g, p = entry
                        except Exception:
                            g = entry; p = 1.0
                        normalized.append((g, float(p)))
                    total_p = sum(p for _, p in normalized)
                    if total_p <= 0:
                        n = len(normalized)
                        outcomes = [(g, 1.0 / n) for (g, _) in normalized]
                    else:
                        outcomes = [(g, p / total_p) for (g, p) in normalized]
            except Exception:
                if self.debug:
                    print("decide: failed to obtain outcomes for", repr(a))
                    traceback.print_exc()
                continue
            if not outcomes:
                continue
            if len(outcomes) > self.SPECTRUM_MAX_OUTCOMES:
                outcomes = outcomes[: self.SPECTRUM_MAX_OUTCOMES]
            expected_value_a = 0.0
            expansion_potential_a = 0.0
            one_ply_vp_delta = next((v for (act, s, v) in one_ply_results if act == a), 0.0)
            robber_impact_a = -float("inf")
            if self._is_robber_or_chance(a):
                try:
                    robber_impact_a = self._compute_opponent_impact(game, a)
                except Exception:
                    robber_impact_a = -float("inf")
            outcome_failures = 0
            for og, p_i in outcomes:
                if sim_count >= SIMULATION_HARD_LIMIT:
                    break
                try:
                    playable = self._derive_opponent_actions(og, self.color)
                    expansion = len(playable) if playable else 0
                    expansion_potential_a += p_i * expansion
                except Exception:
                    expansion_potential_a += p_i * -float("inf")
                opp_color = self._determine_opponent_color(og, self.color)
                try:
                    opp_actions = self._derive_opponent_actions(og, opp_color)
                except Exception:
                    opp_actions = []
                if not opp_actions:
                    val_i = self._simulate_and_evaluate(og, None, self.color)
                    if val_i is None:
                        outcome_failures += 1
                        continue
                    expected_value_a += p_i * val_i
                    sim_count += 1
                    continue
                opp_sampled = self._sample_opponent_actions(opp_actions, og, opp_color)[: self.OP_MAX_ACTIONS]
                min_score_after_opp = float("inf")
                opp_successes = 0
                for b in opp_sampled:
                    if sim_count >= SIMULATION_HARD_LIMIT:
                        break
                    val_after_b = self._simulate_and_evaluate(og, b, self.color)
                    sim_count += 1
                    if val_after_b is None:
                        continue
                    opp_successes += 1
                    if val_after_b < min_score_after_opp:
                        min_score_after_opp = val_after_b
                if opp_successes == 0:
                    tmp = self._simulate_and_evaluate(og, None, self.color)
                    if tmp is None:
                        outcome_failures += 1
                        continue
                    min_score_after_opp = tmp
                expected_value_a += p_i * min_score_after_opp
            if outcome_failures >= max(1, len(outcomes) // 2):
                continue
            deep_successful += 1
            # integrate rollout metrics into tie-break
            rollout_info = rollout_metrics.get(a, (-1, -1, -1))
            # Build comparison tuple per STRATEGIZER
            comp_tuple = (
                expected_value_a,
                settlement_gain_scores.get(a, -float("inf")),
                rollout_info[0],
                rollout_info[1],
                expansion_potential_a,
                robber_impact_a,
                self._count_build_actions(game, self.color),
                rollout_info[2],
                one_ply_vp_delta,
                repr(a),
            )
            if best_tuple is None or comp_tuple > best_tuple:
                best_tuple = comp_tuple
                best_action = a
        if deep_successful > 0:
            self._diag["n_2ply_runs"] += 1
        else:
            self._diag["n_2ply_skipped"] += 1
        self._diag["simulated_nodes_total"] += sim_count
        if self.debug:
            print("Road rollout diagnostics:")
            print(f"  n_road_rollouts_run: {self._diag.get('n_road_rollouts_run',0)}")
            print(f"  sim_budget_used: {self.ROAD_ROLLOUT_SIM_BUDGET - sim_budget_remaining}")
            print(f"  n_road_candidates_included: {self._diag.get('n_road_candidates_included',0)}")
        if best_action is not None:
            return best_action
        # fallback to 1-ply
        best = max(one_ply_results, key=lambda t: (t[1], t[2], repr(t[0])))
        return best[0]
