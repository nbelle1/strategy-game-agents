import os
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
        cost = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None)
        if not cost:
            # permissive default if cost not available
            return True
        # get player resources with fallbacks
        player_res = {}
        get_res = getattr(game, "get_player_resources", None)
        try:
            if callable(get_res):
                player_res = get_res(self) or {}
        except Exception:
            player_res = {}
        # additional fallbacks
        if not player_res:
            player_res = getattr(self, "resources", None) or {}
        if not player_res:
            gr = getattr(game, "resources", None)
            try:
                if isinstance(gr, dict):
                    player_res = gr.get(self, {}) or {}
            except Exception:
                player_res = {}
        # Check cost (defensive)
        try:
            for r, amt in (cost.items() if hasattr(cost, "items") else []):
                try:
                    need = int(amt)
                except Exception:
                    need = 0
                if player_res.get(r, 0) < need:
                    return False
        except Exception:
            # unexpected format -> permissive
            return True
        return True

    def resource_spend_cost(self, action) -> int:
        """Return integer total of resource quantities required by action cost using safe fallbacks.

        Cost fallback (exact):
        Cost: getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
        """
        cost = getattr(action, "cost", None) or getattr(action, "required_resources", None) or getattr(action, "resources_needed", None) or {}
        if not hasattr(cost, "items"):
            return 0
        total = 0
        try:
            for r, amt in cost.items():
                try:
                    total += int(amt)
                except Exception:
                    total += 0
        except Exception:
            return 0
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
                if getattr(action, "upgrades_settlement", True):
                    return 1.0
                # otherwise still return 1.0 conservatively
                return 1.0
            if at == ActionType.BUY_DEV_CARD or (name == "BUY_DEV_CARD"):
                return float(self.dev_expected_vp)
        except Exception:
            pass
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
        f_vp = float(self.estimate_vp_gain(action, game))
        at = getattr(action, "action_type", None) or getattr(action, "type", None)
        at_name = getattr(at, "name", None) if at is not None else None
        f_dev = 1.0 if (at == ActionType.BUY_DEV_CARD or str(at_name) == "BUY_DEV_CARD") else 0.0
        f_res_cost = int(self.resource_spend_cost(action))
        score = float(self.weight_vp * f_vp + self.weight_dev * f_dev + (-1.0) * (f_res_cost / self.resource_penalty_scale))
        return score

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
            except Exception:
                affordable = True
            if not affordable:
                s = float("-inf")
            else:
                try:
                    s = self.score_action(action, game)
                except Exception:
                    s = float("-inf")
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
