# Robust import that works both as a package and when run locally
try:
    from .adapters import (
        Game, Player, Color, Action, ActionType,
        playable_actions, pruned_actions, chance_children,
        make_value_fn, DEFAULT_WEIGHTS, value_production,
        production_features_sampler, winning_color, copy_game
    )
except ImportError:
    # Fallback if executed as a script from within this folder
    from adapters import (
        Game, Player, Color, Action, ActionType,
        playable_actions, pruned_actions, chance_children,
        make_value_fn, DEFAULT_WEIGHTS, value_production,
        production_features_sampler, winning_color, copy_game
    )

class FooPlayer(Player):
    def __init__(self, color, value_builder="base_fn", params=DEFAULT_WEIGHTS):
        super().__init__(color)
        self.V = make_value_fn(value_builder, params)  # callable(game, pov_color)

    def decide(self, game, _playable):
        acts = pruned_actions(game)
        if not acts:
            acts = playable_actions(game)
        if len(acts) == 1:
            return acts[0]

        # Expectation over chance outcomes, mirroring AlphaBeta’s spectrum
        exp = {}
        for a, outs in chance_children(game, acts).items():
            exp[a] = sum(p * self.V(gp, self.color) for gp, p in outs)

        # choose max-EV action
        return max(exp, key=exp.get)

    def pip_weight(number):
        if number in (6, 8):
            return 5.0
        if number in (5, 9):
            return 4.0
        if number in (4, 10):
            return 3.0
        if number in (3, 11):
            return 2.0
        if number in (2, 12):
            return 1.0
        return 0.0

    def get_adjacent_tiles_for_action(game, action):
        # Try multiple attribute names that may be present on action
        # and on game.board representation.
        # Return a list of tile objects or simple dicts with .resource and .number
        try:
            location = getattr(action, "location", None)
            if location is None:
                location = getattr(action, "vertex", None)
            if location is None:
                location = getattr(action, "point", None)
            # If action includes a list of hex ids directly
            hexes = getattr(action, "hexes", None) or getattr(action, "adjacent_hexes", None)
            if hexes:
                tiles = []
                for hx in hexes:
                    # try resolving hex object from game.board
                    try:
                        tiles.append(game.board.hexes[hx])
                    except Exception:
                        # maybe hex is already a tile object
                        tiles.append(hx)
                return tiles
            # Otherwise, try to derive adjacent tiles from board using location
            if location is not None:
                # This is adapter-dependent. Try common patterns, gracefully degrade.
                if hasattr(game.board, "adjacent_tiles_to_vertex"):
                    return game.board.adjacent_tiles_to_vertex(location)
                if hasattr(game.board, "tiles_adjacent_to_point"):
                    return game.board.tiles_adjacent_to_point(location)
                # Last-resort: look for an attribute on location itself
                tiles = getattr(location, "adjacent_tiles", None)
                if tiles:
                    return tiles
        except Exception as e:
            print("get_adjacent_tiles_for_action fallback error:", e)
        # If we can't find anything, return empty list
        return []

    def score_opening_placement(game, action):
        try:
            tiles = get_adjacent_tiles_for_action(game, action)
            if not tiles:
                return -9999.0  # strongly discourage unknown placements
            resources = set()
            pip_sum = 0.0
            for t in tiles:
                # tile may be a dict or object
                resource = None
                number = None
                if isinstance(t, dict):
                    resource = t.get("resource") or t.get("type") or t.get("res")
                    number = t.get("number") or t.get("pip") or t.get("roll")
                else:
                    resource = getattr(t, "resource", None) or getattr(t, "type", None)
                    number = getattr(t, "number", None) or getattr(t, "pip", None)
                if resource and str(resource).lower() not in ("desert", "none", "null"):
                    resources.add(resource)
                pip_sum += pip_weight(number)
            # Score formula:
            # diversity bonus + pip sum (weighted)
            diversity_score = len(resources) * 2.0
            pip_score = pip_sum * 1.5  # 1.5 scaling for pip weight
            return diversity_score + pip_score
        except Exception as e:
            print("Error scoring opening placement:", e)
            return -9999.0

    def is_settlement_placement_action(a):
        try:
            # detect placement actions conservatively
            at = getattr(a, "action_type", None) or getattr(a, "type", None) or str(a)
            if at is None:
                return False
            at_str = str(at).upper()
            if "SETTLEMENT" in at_str or "PLACE_SETTLEMENT" in at_str or "BUILD_SETTLEMENT" in at_str:
                return True
        except Exception:
            return False
        return False

    def decide(self, game, _playable):
        try:
            # opening detection
            my_settlements = len([s for s in game.settlements if s.owner == self.color]) if hasattr(game, "settlements") else 0
            opening_condition = (getattr(game, "turn", 0) < 2) or (my_settlements < 2)
            if opening_condition:
                candidate_actions = []
                for a in available_actions:
                    try:
                        if is_settlement_placement_action(a):
                            candidate_actions.append(a)
                    except Exception:
                        continue
                if candidate_actions:
                    best = None
                    best_score = -1e9
                    for a in candidate_actions:
                        s = score_opening_placement(game, a)
                        print(f"[OPENING] action={a} score={s}")
                        if s > best_score:
                            best_score = s
                            best = a
                    if best is not None:
                        print(f"[OPENING] chosen {best} score {best_score}")
                        return best
        except Exception as e:
            print("[OPENING] placement logic error:", e)
            # continue with normal decision-making

        acts = pruned_actions(game)
        if not acts:
            acts = playable_actions(game)
        if len(acts) == 1:
            return acts[0]

        # Expectation over chance outcomes, mirroring AlphaBeta’s spectrum
        exp = {}
        for a, outs in chance_children(game, acts).items():
            exp[a] = sum(p * self.V(gp, self.color) for gp, p in outs)

        # choose max-EV action
        return max(exp, key=exp.get)
