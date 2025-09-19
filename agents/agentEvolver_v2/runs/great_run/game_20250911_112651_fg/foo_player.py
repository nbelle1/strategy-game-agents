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
