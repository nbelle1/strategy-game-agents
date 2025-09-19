# Robust import that works both as a package and when run locally
from .adapters import (
    Game, Player, Color, Action, ActionType,
    playable_actions, pruned_actions, chance_children,
    make_value_fn, DEFAULT_WEIGHTS, value_production,
    production_features_sampler, winning_color, copy_game
)
import random

class FooPlayer(Player):
    def __init__(self, color, *_args, **_kwargs):
        super().__init__(color)

    def decide(self, game, _playable):
        acts = pruned_actions(game) or playable_actions(game)
        if not acts:
            return None  # no legal moves
        if len(acts) == 1:
            return acts[0]
        return random.choice(acts)