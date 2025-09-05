import os
import traceback
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType


class FooPlayer(Player):
    # Exact numeric constants (class attributes)
    weight_vp = 100.0
    weight_dev = 30.0
    dev_expected_vp = 0.2
    resource_penalty_scale = 5.0

    # Iteration-2 additional constants
    weight_settlement_spot = 2.0
    weight_road_extension = 8.0
    weight_block = 6.0
    hand_penalty_weight = 20.0

    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    # Fallback rules (word-for-word) used in helpers:
    # - Action type: getattr(action, "action_type", None) or getattr(action, "type", None)
    # - Action type name: getattr(at, "name", None) or getattr(at, "value", None) or str(at)
    # - Cost: getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
    # - Player resources: try callable game.get_player_resources(self) first, else getattr(self, "resources", None), else game.resources.get(self, {})
    # - If any expected attribute is absent or in unexpected format, do not raise; assume afford True or cost 0.

    def _extract_action_type_name(self, action):
        """Return a stable string name for an action's type using safe fallbacks."""
        at = getattr(action, "action_type", None) or getattr(action, "type", None)
        # handle enum-like or string-like types robustly
        name = None
        try:
            name = getattr(at, "name", None) or getattr(at, "value", None) or str(at)
        except Exception:
            try:
                name = str(at)
            except Exception:
                name = "UNKNOWN"
        return str(name)

    def can_afford(self, action, game) -> bool:
        """Return True if this player appears to have resources to pay for action's cost.

        Uses robust parsing and multiple get_player_resources signatures. Permissive when
        parsing fails to avoid crashes during the game.
        """
        raw_cost = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or getattr(action, "payment", None)
        if not raw_cost:
            # no-cost actions (ROLL, END_TURN, etc.) are affordable
            return True

        parsed = {}
        try:
            # try dict-like first
            if hasattr(raw_cost, "items"):
                parsed = dict(raw_cost)
            elif isinstance(raw_cost, (list, tuple)):
                for e in raw_cost:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        k = e[0]; v = e[1]
                        try:
                            parsed.setdefault(k, 0); parsed[k] += int(v)
                        except Exception:
                            pass
                    elif isinstance(e, dict):
                        if "resource" in e and "count" in e:
                            try:
                                parsed.setdefault(e["resource"], 0); parsed[e["resource"]] += int(e["count"])
                            except Exception:
                                pass
                        else:
                            for k, v in e.items():
                                if isinstance(k, str) and isinstance(v, (int, str)):
                                    try:
                                        parsed.setdefault(k, 0); parsed[k] += int(v)
                                    except Exception:
                                        pass
        except Exception:
            parsed = {}

        # get player resources robustly
        player_res = {}
        get_res = getattr(game, "get_player_resources", None)
        if callable(get_res):
            for arg in (self, getattr(self, "player_id", None), getattr(self, "index", None), getattr(self, "color", None)):
                if arg is None:
                    continue
                try:
                    pr = get_res(arg)
                    if pr:
                        player_res = pr
                        break
                except Exception:
                    pass
        if not player_res:
            player_res = getattr(self, "resources", None) or (getattr(game, "resources", None) or {}).get(self, {})

        if not parsed:
            print(f"[DIAG][can_afford] could not parse cost for action={self._extract_action_type_name(action)} raw_cost={repr(raw_cost)}")
            return True

        try:
            for r, need in parsed.items():
                try:
                    need_int = int(need)
                except Exception:
                    need_int = 0
                if player_res.get(r, 0) < need_int:
                    return False
        except Exception as e:
            print(f"[ERROR][can_afford] check exception: {e}")
            traceback.print_exc()
            return True
        return True

    def resource_spend_cost(self, action) -> int:
        """Return integer total of resource quantities required by action cost using safe fallbacks.

        Tries many common cost shapes and emits diagnostics when it cannot parse a cost.
        """
        raw = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or getattr(action, "payment", None) or None
        if raw is None:
            return 0
        total = 0
        try:
            if hasattr(raw, "items"):
                for r, amt in raw.items():
                    try:
                        total += int(amt)
                    except Exception:
                        pass
                if total == 0:
                    print(f"[DIAG][zero-cost] raw_cost_repr={repr(getattr(action,'cost', None))} alt_required_resources={repr(getattr(action,'required_resources',None))} alt_resources_needed={repr(getattr(action,'resources_needed',None))} action_repr={repr(action)}")
                return total
            if isinstance(raw, (list, tuple)):
                for entry in raw:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        try:
                            total += int(entry[1])
                        except Exception:
                            pass
                    elif isinstance(entry, dict):
                        if "count" in entry:
                            try:
                                total += int(entry["count"])
                            except Exception:
                                pass
                        elif "amount" in entry:
                            try:
                                total += int(entry["amount"])
                            except Exception:
                                pass
                        elif "qty" in entry:
                            try:
                                total += int(entry["qty"])
                            except Exception:
                                pass
                        else:
                            for v in entry.values():
                                if isinstance(v, (int, float, str)):
                                    try:
                                        total += int(v)
                                        break
                                    except Exception:
                                        pass
                if total == 0:
                    print(f"[DIAG][zero-cost] raw_cost_repr={repr(getattr(action,'cost', None))} alt_required_resources={repr(getattr(action,'required_resources',None))} alt_resources_needed={repr(getattr(action,'resources_needed',None))} action_repr={repr(action)}")
                return total
            for attr in ("resources", "cost_map", "costs", "_cost"):
                val = getattr(raw, attr, None)
                if val:
                    if hasattr(val, "items"):
                        for r, amt in val.items():
                            try:
                                total += int(amt)
                            except Exception:
                                pass
                        if total == 0:
                            print(f"[DIAG][zero-cost] unparsed_raw_attr={attr} raw={repr(raw)} action={repr(action)}")
                        return total
            try:
                for v in raw:
                    try:
                        total += int(v)
                    except Exception:
                        pass
                if total > 0:
                    return total
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR][resource_spend_cost] parse exception: {e}")
            import traceback
            traceback.print_exc()
        print(f"[DIAG][resource_spend_cost] unparsed_raw={repr(raw)} action_type={self._extract_action_type_name(action)}")
        return total

    def estimate_vp_gain(self, action, game) -> float:
        """Conservative immediate VP estimate for the action.

        Uses string-based action name checks to avoid AttributeError from missing enum members.
        """
        try:
            at = getattr(action, "action_type", None) or getattr(action, "type", None)
            at_name = getattr(at, "name", None) if at is not None else None
            action_name = str(at_name or at or "").upper()
            # Common canonical names to match
            if action_name in ("BUILD_SETTLEMENT", "BUILD_SETTLEMENT_ACTION"):
                return 1.0
            if action_name in ("BUILD_CITY", "BUILD_CITY_ACTION"):
                ups = getattr(action, "upgrades_settlement", None)
                if ups is not None:
                    return 1.0 if bool(ups) else 0.0
                return 1.0
            if action_name in ("BUY_DEV_CARD", "BUY_DEVELOPMENT_CARD", "DEVCARD_BUY"):
                return float(self.dev_expected_vp)
        except Exception as e:
            print(f"[ERROR] estimate_vp_gain exception for action raw_type={getattr(action,'action_type',None)} err={e}")
            traceback.print_exc()
        return 0.0

    def score_action(self, action, game) -> float:
        """Score an action using iteration-2 composition (vp, dev, spot, road, block, resource penalty, hand penalty).
        """
        try:
            f_vp = float(self.estimate_vp_gain(action, game))
            at = getattr(action, "action_type", None) or getattr(action, "type", None)
            at_name = getattr(at, "name", None) if at is not None else None
            action_name = str(at_name or at or "").upper()
            f_dev = 1.0 if action_name in ("BUY_DEV_CARD", "BUY_DEVELOPMENT_CARD", "DEVCARD_BUY") else 0.0
            f_res_cost = int(self.resource_spend_cost(action))
            # Iter-2 additions
            spot_value = self.spot_value(action, game) if action_name.startswith("BUILD_SETTLEMENT") and hasattr(self, "spot_value") else 0.0
            road_bonus = self.road_extension_value(action, game) if action_name.startswith("BUILD_ROAD") and hasattr(self, "road_extension_value") else 0.0
            block_bonus = self.blocking_value(action, game) if hasattr(self, "blocking_value") else 0.0
            try:
                hand_after = self.hand_count_after(action, game) if hasattr(self, "hand_count_after") else 0
                penalty = getattr(self, "hand_penalty_weight", 20.0) * max(0, hand_after - 6)
            except Exception:
                penalty = 0.0
            score = float(self.weight_vp * f_vp + self.weight_dev * f_dev + getattr(self, "weight_settlement_spot", 2.0) * spot_value + getattr(self, "weight_road_extension", 8.0) * road_bonus + getattr(self, "weight_block", 6.0) * block_bonus + (-1.0) * (f_res_cost / self.resource_penalty_scale) - penalty)
            return score
        except Exception as e:
            print(f"[ERROR] score_action exception for action raw_type={getattr(action,'action_type',None)} err={e}")
            traceback.print_exc()
            return -1e6

    def decide(self, game, playable_actions):
        """Choose one action from playable_actions using deterministic scoring and tie-breaks.

        Builds scored entries as tuples: (score, orig_index, action, action_name, res_cost)
        and sorts using the exact key:
        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[3], item[4], item[1]))

        Emits debug prints required for iteration 1.
        """
        # Defensive: if no playable actions, return None per environment expectations
        if not playable_actions:
            return None

        scored = []  # list of tuples: (score, orig_index, action, action_name, res_cost)
        for idx, action in enumerate(playable_actions):
            try:
                affordable = self.can_afford(action, game)
            except Exception as e:
                print(f"[ERROR] can_afford exception for action_idx={idx} action={self._extract_action_type_name(action)} err={e}")
                traceback.print_exc()
                # Default to permissive when can_afford fails
                affordable = True

            if not affordable:
                s = -1e6   # use large negative numeric instead of -inf for diagnostics
            else:
                try:
                    s = self.score_action(action, game)
                    # NaN and None guard
                    if s is None or (isinstance(s, float) and (s != s)):
                        print(f"[DEBUG] score_action returned non-finite for idx={idx} action={self._extract_action_type_name(action)}; treating as -1e6")
                        s = -1e6
                except Exception as e:
                    print(f"[ERROR] score_action exception for action_idx={idx} action={self._extract_action_type_name(action)} err={e}")
                    traceback.print_exc()
                    s = -1e6
            a_name = self._extract_action_type_name(action)
            a_cost = self.resource_spend_cost(action)
            scored.append((s, idx, action, a_name, a_cost))

        # Exact tie-break sort (copy this line)
        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[3], item[4], item[1]))
        if not scored_sorted:
            # safe fallback: return first playable action if none scored
            return playable_actions[0] if playable_actions else None
        chosen_score, chosen_index, chosen_action, chosen_name, chosen_cost = scored_sorted[0]
        chosen_vp = self.estimate_vp_gain(chosen_action, game)
        # If we are still picking a fallback-large-negative score, print raw type for debugging
        try:
            if chosen_score <= -1e5:
                raw_at = getattr(chosen_action, "action_type", None)
                print(f"[DEBUG][choose-fallback] chosen raw action_type={raw_at} action_name={chosen_name} score={chosen_score}")
        except Exception:
            pass
        # Exact debug prints to replace previous line
        print(f"[DEBUG] ChosenAction: type={chosen_name} score={chosen_score:.2f} cost={chosen_cost} vp_gain={chosen_vp:.2f}")
        print("[DEBUG] Top3 candidates:")
        for rank, (s, idx, a, a_name, a_cost) in enumerate(scored_sorted[:3], start=1):
            a_vp = self.estimate_vp_gain(a, game)
            print(f"[DEBUG]  {rank}) type={a_name} score={s:.2f} cost={a_cost} vp_gain={a_vp:.2f} orig_index={idx}")
        return chosen_action

    # --- Iteration-2 stubs ---
    def spot_value(self, action, game):
        try:
            pos = getattr(action, "target_pos", None) or getattr(action, "location", None) or getattr(action, "node_id", None)
            board = getattr(game, "board", None) or (getattr(game, "get_board", None)() if callable(getattr(game, "get_board", None)) else None)
            if not board or pos is None:
                return 0.0
            tiles = None
            if hasattr(board, "get_node_tiles"):
                tiles = board.get_node_tiles(pos)
            elif hasattr(board, "nodes") and pos in getattr(board, "nodes", {}):
                node = board.nodes[pos]
                tiles = getattr(node, "tiles", None)
            if not tiles:
                return 0.0
            pip_weight = 0.0
            for t in tiles:
                pip = getattr(t, "pip", None) or getattr(t, "value", None) or getattr(t, "number", None)
                if pip in (6, 8):
                    pip_weight += 2.0
                elif pip in (5, 9):
                    pip_weight += 1.5
                elif pip in (4, 10):
                    pip_weight += 1.0
                elif pip in (3, 11):
                    pip_weight += 0.7
                elif pip in (2, 12):
                    pip_weight += 0.3
            return pip_weight
        except Exception:
            return 0.0

    def road_extension_value(self, action, game):
        try:
            return 1.0 if getattr(action, "extends_own_road", False) else 0.0
        except Exception:
            return 0.0

    def blocking_value(self, action, game):
        try:
            return 1.0 if getattr(action, "blocks_player", False) else 0.0
        except Exception:
            return 0.0

    def hand_count_after(self, action, game):
        try:
            player_res = {}
            get_res = getattr(game, "get_player_resources", None)
            if callable(get_res):
                try:
                    player_res = get_res(self) or {}
                except Exception:
                    player_res = {}
            player_res = player_res or getattr(self, "resources", None) or (getattr(game, "resources", None) or {}).get(self, {})
            hand = sum(int(v) for v in player_res.values()) if isinstance(player_res, dict) else 0
            cost = self.resource_spend_cost(action)
            return max(0, hand - cost)
        except Exception:
            return 0
