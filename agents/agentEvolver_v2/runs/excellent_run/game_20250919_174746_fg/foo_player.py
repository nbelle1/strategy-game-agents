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

        # Priority mapping for tie-breaking. Higher -> more preferred.
        PRIORITY = {
            ActionType.BUILD_CITY: 100,
            ActionType.BUILD_SETTLEMENT: 90,
            ActionType.BUILD_ROAD: 80,
            ActionType.BUY_DEVELOPMENT_CARD: 70,
            ActionType.PLAY_DEVELOPMENT_CARD: 60,
            ActionType.TRADE: 50,
            ActionType.MOVE_ROBBER: 40,
            ActionType.END_TURN: 0,
        }

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
        build_like_types = {
            ActionType.BUILD_CITY,
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_ROAD,
            ActionType.BUY_DEVELOPMENT_CARD,
            ActionType.PLAY_DEVELOPMENT_CARD,
        }
        has_build_like = any(getattr(a, 'action_type', None) in build_like_types for a in actions)

        # Filter out END_TURN if we can do something productive
        filtered = [a for a in actions if not (getattr(a, 'action_type', None) == ActionType.END_TURN and has_build_like)]

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
            print('Choosing First Action on Default')
            return actions[0]

        # Debug/logging to help with experiments
        print(f"[foo_player] Chosen action: {best_action}, score: {best_score:.3f}, evaluated: {evaluated}/{len(filtered)}")

        return best_action
