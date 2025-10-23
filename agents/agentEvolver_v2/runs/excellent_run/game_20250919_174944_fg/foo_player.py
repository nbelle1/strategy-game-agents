import os
from .adapters import (
    Player, Color, Action, ActionType,
    copy_game, execute, make_value_fn, DEFAULT_WEIGHTS,
)


class FooPlayer(Player):
    """
    FooPlayer: an iterative improvement over the trivial "first-action" bot.

    Strategy implemented here (1-ply lookahead):
    - Filter out END_TURN when there are available build/play actions.
    - Cap number of actions evaluated (default 20) for performance.
    - For each candidate action: copy the game, execute the action on the copy,
      and score the resulting state with make_value_fn("base_fn").
    - Choose the action with the highest score; tie-break by a prioritized
      action ordering that prefers building and development actions.

    Notes / debugging:
    - We only use the adapter surface (copy_game, execute, make_value_fn).
    - Print statements are provided to help trace decision-making during
      experiments. Remove or tone down if logs become too noisy.
    """

    def __init__(self, name: str | None = None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        """
        Decide which action to take from playable_actions.

        Args:
            game: full Game object (read-only for this method).
            playable_actions: iterable of Action objects available this turn.
        Returns:
            One Action chosen from playable_actions.
        """
        # Configuration: cap how many actions we simulate to bound runtime
        max_actions = 20

        # Helper: try multiple common enum member name variants and return
        # the first matching ActionType. This avoids brittle AttributeError
        # when different forks name members differently.
        def resolve_action_type(*names):
            for n in names:
                if hasattr(ActionType, n):
                    return getattr(ActionType, n)
            return None

        # Resolve commonly-used action types robustly (fallback-safe)
        AT_BUILD_CITY = resolve_action_type('BUILD_CITY')
        AT_BUILD_SETTLEMENT = resolve_action_type('BUILD_SETTLEMENT', 'BUILD_SETTLES')
        AT_BUILD_ROAD = resolve_action_type('BUILD_ROAD', 'BUILD_ROADS')
        AT_BUY_DEV = resolve_action_type('BUY_DEVELOPMENT_CARD', 'BUY_DEV_CARD', 'BUY_DEVELOPMENT')
        AT_PLAY_DEV = resolve_action_type('PLAY_DEVELOPMENT_CARD', 'PLAY_DEV_CARD', 'PLAY_DEVELOPMENT')
        AT_TRADE = resolve_action_type('TRADE')
        AT_MOVE_ROBBER = resolve_action_type('MOVE_ROBBER')
        AT_END_TURN = resolve_action_type('END_TURN')

        # Priority mapping for tie-breaking. Higher -> more preferred.
        _priority_map = {
            AT_BUILD_CITY: 100,
            AT_BUILD_SETTLEMENT: 90,
            AT_BUILD_ROAD: 80,
            AT_BUY_DEV: 70,
            AT_PLAY_DEV: 60,
            AT_TRADE: 50,
            AT_MOVE_ROBBER: 40,
            AT_END_TURN: 0,
        }
        # Remove any None keys (in case some ActionType variants don't exist)
        PRIORITY = {k: v for k, v in _priority_map.items() if k is not None}

        def action_priority(a):
            # Guard if a is None
            if a is None:
                return -1
            return PRIORITY.get(getattr(a, 'action_type', None), 10)

        # Convert to list to allow multiple passes
        actions = list(playable_actions)
        if not actions:
            print('[foo_player] No playable actions available; returning None')
            return None

        # If there exist build-like actions, avoid selecting END_TURN immediately
        build_like_types = {t for t in (AT_BUILD_CITY, AT_BUILD_SETTLEMENT, AT_BUILD_ROAD, AT_BUY_DEV, AT_PLAY_DEV) if t is not None}
        has_build_like = any(getattr(a, 'action_type', None) in build_like_types for a in actions)

        # Filter out END_TURN if we can do something productive
        if AT_END_TURN is not None:
            filtered = [a for a in actions if not (getattr(a, 'action_type', None) == AT_END_TURN and has_build_like)]
        else:
            filtered = list(actions)

        # If filtering removed everything (rare), fall back to original actions
        if not filtered:
            filtered = actions

        # Cap the number of actions to evaluate. Keep highest priority actions first.
        if len(filtered) > max_actions:
            filtered = sorted(filtered, key=lambda x: action_priority(x), reverse=True)[:max_actions]

        # Build the value function. Some adapter variants accept weights, some not.
        try:
            value_fn = make_value_fn("base_fn", DEFAULT_WEIGHTS)
        except TypeError:
            # Fallback: older/newer signatures might only take the builder name
            value_fn = make_value_fn("base_fn")

        best_action = None
        best_score = -float('inf')
        evaluated = 0

        # Simulate each candidate (1-ply) and score the resulting state
        for action in filtered:
            try:
                gcopy = copy_game(game)
                # Try the faster validate=False first; if adapter/engine doesn't accept it, fallback.
                try:
                    execute(gcopy, action, validate=False)
                except TypeError:
                    execute(gcopy, action)

                # Score from our POV
                try:
                    score = value_fn(gcopy, self.color)
                except TypeError:
                    # Some value_fn implementations may only accept game and return global score;
                    # in that case we assume higher is better and proceed.
                    score = value_fn(gcopy)

            except Exception as e:
                # If a simulated action fails for any reason, skip it and keep going.
                print(f"[foo_player] simulation failed for action={action} error={e}")
                continue

            evaluated += 1
            # Tie-break by score then action priority
            if (score > best_score) or (score == best_score and action_priority(action) > action_priority(best_action)):
                best_score = score
                best_action = action

        # If no action survived simulation (very unlikely), return the first playable action
        if best_action is None:
            print('[foo_player] No simulated action succeeded; falling back to first playable action')
            return actions[0]

        # Debug/logging to help with experiments
        print(f"[foo_player] Chosen action: {best_action}, score: {best_score:.3f}, evaluated: {evaluated}/{len(filtered)}")

        return best_action
