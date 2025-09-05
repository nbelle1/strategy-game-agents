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
        if at is None:
            return "UNKNOWN_ACTION"
        name = getattr(at, "name", None) or getattr(at, "value", None) or str(at)
        return str(name)

    def can_afford(self, action, game) -> bool:
        """Return True if this player appears to have resources to pay for action's cost.

        Uses fallbacks (exact):
        Cost: getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
        Player resources: try callable game.get_player_resources(self) first, else getattr(self, "resources", None), else game.resources.get(self, {})
        If formats are unexpected, permissive True is returned so we do not crash during decision.
        """
        # Exact cost fallback chain
        cost = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None)
        if not cost:
            # permissive default if cost not available (e.g., ROLL, END_TURN)
            return True

        # Try multiple player-resource lookup variants and be verbose when failing
        player_res = {}
        get_res = getattr(game, "get_player_resources", None)
        try:
            if callable(get_res):
                # Try calling with several common player identifiers (self, player_id, index, color)
                for arg in (self, getattr(self, "player_id", None), getattr(self, "index", None), getattr(self, "color", None)):
                    try:
                        if arg is None:
                            continue
                        pr = get_res(arg)
                        if pr:
                            player_res = pr
                            break
                    except Exception:
                        # ignore and try next possibility
                        continue
        except Exception:
            # If game.get_player_resources itself throws, swallow and fallbacks below will try
            pass

        # additional fallbacks (exact fallback order as requested)
        if not player_res:
            player_res = getattr(self, "resources", None) or {}
        if not player_res:
            gr = getattr(game, "resources", None)
            try:
                if isinstance(gr, dict):
                    player_res = gr.get(self, {}) or {}
            except Exception:
                player_res = {}

        # Debugging trace when resources lookups are empty or small
        try:
            if not player_res:
                print(f"[DEBUG][can_afford] no player_res found for player={getattr(self,'player_id',None)}; action={self._extract_action_type_name(action)} cost={cost}")
        except Exception:
            print(f"[DEBUG][can_afford] resource lookup debug failed for action")

        # Check cost defensively
        try:
            items = cost.items() if hasattr(cost, "items") else []
            for r, amt in items:
                try:
                    need = int(amt)
                except Exception:
                    need = 0
                if player_res.get(r, 0) < need:
                    return False
        except Exception as e:
            # log unexpected format and be permissive
            print(f"[ERROR][can_afford] unexpected cost format for action={self._extract_action_type_name(action)}; err={e}")
            traceback.print_exc()
            return True
        return True

    def resource_spend_cost(self, action) -> int:
        """Return integer total of resource quantities required by action cost using safe fallbacks.

        Cost fallback (exact):
        Cost: getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
        """
        cost = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
        total = 0
        try:
            # If cost is a dict-like mapping
            if hasattr(cost, "items"):
                for r, amt in cost.items():
                    try:
                        total += int(amt)
                    except Exception:
                        total += 0
            elif isinstance(cost, (list, tuple)):
                # if cost provided as list of (res,amount)
                for entry in cost:
                    try:
                        amt = entry[1]
                        total += int(amt)
                    except Exception:
                        pass
            else:
                # unknown format: log and return 0
                print(f"[DEBUG][resource_spend_cost] unexpected cost format: {cost}")
        except Exception as e:
            print(f"[ERROR][resource_spend_cost] exception: {e}")
            traceback.print_exc()
        return total

    def estimate_vp_gain(self, action, game) -> float:
        """Conservative immediate VP estimate for the action.

        Uses ActionType comparisons and safe name fallbacks. Treat BUILD_SETTLEMENT as +1 VP,
        BUILD_CITY as +1 (conservative) and BUY_DEV_CARD as expected dev VP defined by dev_expected_vp.
        If a city is being built only as an upgrade when an existing settlement is present, the
        optional attribute 'upgrades_settlement' is honored if provided by the environment.
        """
        at = getattr(action, "action_type", None) or getattr(action, "type", None)
        name = getattr(at, "name", None) if at is not None else None
        try:
            # Treat settlement build as +1 VP
            if at == ActionType.BUILD_SETTLEMENT or (name == "BUILD_SETTLEMENT"):
                return 1.0
            # City upgrade: conservative +1 if it upgrades a settlement
            if at == ActionType.BUILD_CITY or (name == "BUILD_CITY"):
                # optional improvement: ensure it's an upgrade if action supplies that info
                if getattr(action, "upgrades_settlement", None) is not None:
                    return 1.0 if getattr(action, "upgrades_settlement", False) else 0.0
                return 1.0
            if at == ActionType.BUY_DEV_CARD or (name == "BUY_DEV_CARD"):
                return float(self.dev_expected_vp)
        except Exception as e:
            print(f"[ERROR] estimate_vp_gain exception for action={self._extract_action_type_name(action)} err={e}")
            traceback.print_exc()
        # Unknown action types: return 0.0
        return 0.0

    def score_action(self, action, game) -> float:
        """Score an action using exact formula for iteration 1.

        Exact constants used:
        weight_vp = 100.0
        weight_dev = 30.0
        dev_expected_vp = 0.2
        resource_penalty_scale = 5.0

        score = weight_vp * f_vp + weight_dev * f_dev + (-1.0) * (f_res_cost / resource_penalty_scale)
        """
        try:
            f_vp = float(self.estimate_vp_gain(action, game))
            at = getattr(action, "action_type", None) or getattr(action, "type", None)
            at_name = getattr(at, "name", None) if at is not None else None
            f_dev = 1.0 if (at == ActionType.BUY_DEV_CARD or str(at_name) == "BUY_DEV_CARD") else 0.0
            f_res_cost = int(self.resource_spend_cost(action))
            score = float(self.weight_vp * f_vp + self.weight_dev * f_dev + (-1.0) * (f_res_cost / self.resource_penalty_scale))
            return score
        except Exception as e:
            print(f"[ERROR] score_action exception for action={self._extract_action_type_name(action)} err={e}")
            traceback.print_exc()
            # return a large negative numeric instead of -inf to keep sorting numeric
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
        # Exact debug prints to replace previous line
        print(f"[DEBUG] ChosenAction: type={chosen_name} score={chosen_score:.2f} cost={chosen_cost} vp_gain={chosen_vp:.2f}")
        print("[DEBUG] Top3 candidates:")
        for rank, (s, idx, a, a_name, a_cost) in enumerate(scored_sorted[:3], start=1):
            a_vp = self.estimate_vp_gain(a, game)
            print(f"[DEBUG]  {rank}) type={a_name} score={s:.2f} cost={a_cost} vp_gain={a_vp:.2f} orig_index={idx}")
        return chosen_action
