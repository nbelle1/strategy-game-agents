import random
import statistics
from .adapters import (
    Game, Player, Color
)

# Hyperparameters for this player. Tweak across evolutions.
K_ROLLOUTS = 0  # rollouts disabled in this adapter-limited implementation
MAX_ROLLOUT_DEPTH = 10  # not used currently; kept for future use
MAX_ACTIONS_TO_EVALUATE = 12
DEBUG = True


class FooPlayer(Player):
    """A stronger FooPlayer that performs a 1-ply lookahead and evaluates
    the immediate successor state using a robust, defensive static evaluator.

    Notes on integration with adapters.py:
    - We only use the thin adapter surface exported above (Game, Player, Color).
    - We call game.copy() to create hypothetical states and game.execute(action)
      to apply actions to those copies. We avoid calling any non-exported
      adapter helpers so this file remains compatible with the framework.

    Limitations and rationale:
    - The adapters surface available in this environment does not explicitly
      expose helper functions for enumerating playable actions from an
      arbitrary game object (those are provided to decide() by the harness).
      Because of this we cannot reliably perform multi-step random rollouts
      (we cannot ask the engine for "playable_actions" inside the player for
      subsequent turns). Attempting to call hypothetical internal APIs would
      risk using non-portable / unsupported functions.
    - To still fix the key flaw (always pick the first action) we implement a
      1-ply lookahead over a sampled set of candidate actions and evaluate the
      successor state with a robust static value function that inspects the
      game.state. This is a significant upgrade over the previous behavior
      and provides a solid foundation for future rollout-based evolution.
    """

    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        """Choose an action from playable_actions.

        Strategy implemented:
        - If there are many playable actions, randomly sample up to
          MAX_ACTIONS_TO_EVALUATE actions to limit computation.
        - For each candidate action, copy the game, execute the action on the
          copy, and evaluate the resulting state with _evaluate_state().
        - Choose the action with the highest evaluation. Break ties randomly.

        The evaluation is defensive: it attempts multiple common access
        patterns to extract victory points and common counts (settlements,
        cities, roads). If extraction fails, the evaluator falls back to 0.

        Args:
            game (Game): complete game state. read-only. Use game.copy() to
                         create hypothetical states.
            playable_actions (Iterable[Action]): legal options for this turn.
        Returns:
            action: chosen element of playable_actions, or None if no options.
        """
        # Defensive: if no actions available, return None
        if not playable_actions:
            if DEBUG:
                print('FooPlayer.decide: no playable_actions -> returning None')
            return None

        # Convert playable_actions to a list so we can sample and index
        try:
            actions = list(playable_actions)
        except Exception:
            # If iterable cannot be converted, fall back to returning first
            if DEBUG:
                print('FooPlayer.decide: playable_actions not list-like; defaulting to first')
            try:
                return playable_actions[0]
            except Exception:
                return None

        # Sample candidate actions if there are too many
        if len(actions) > MAX_ACTIONS_TO_EVALUATE:
            candidates = random.sample(actions, MAX_ACTIONS_TO_EVALUATE)
            if DEBUG:
                print(f'FooPlayer.decide: sampled {len(candidates)} of {len(actions)} actions to evaluate')
        else:
            candidates = actions
            if DEBUG:
                print(f'FooPlayer.decide: evaluating all {len(candidates)} actions')

        # Evaluate each candidate action by applying it to a copy of the game
        scores = []  # list of (action, score)
        for i, action in enumerate(candidates):
            try:
                # Copy the game to avoid mutating the original
                new_game = game.copy()

                # Apply the candidate action on the copied game.
                # The standard Game API exposes execute(action) to apply an action.
                # We try both .execute and .apply for defensive compatibility.
                executed = False
                try:
                    new_game.execute(action)
                    executed = True
                except Exception:
                    # Some versions may expose a differently named method.
                    try:
                        new_game.apply(action)
                        executed = True
                    except Exception:
                        executed = False

                if not executed:
                    # If we couldn't apply the action on the copy, mark it as
                    # very poor and continue.
                    if DEBUG:
                        print(f'FooPlayer.decide: failed to execute candidate action {i}; marking score -inf')
                    scores.append((action, float('-inf')))
                    continue

                # Evaluate the successor state
                score = self._evaluate_state(new_game)
                scores.append((action, score))
                if DEBUG:
                    print(f'FooPlayer.decide: action #{i} -> score {score}')

            except Exception as e:
                # Catch-all: do not let the player crash the harness. Penalize
                # the action and continue evaluating others.
                if DEBUG:
                    print(f'FooPlayer.decide: exception while evaluating action #{i}: {e}! Marking -inf')
                scores.append((action, float('-inf')))

        # Choose the best action. If all are -inf or evaluation failed, fall back
        # to the original first-action policy.
        if not scores:
            if DEBUG:
                print('FooPlayer.decide: no scores produced -> defaulting to first action')
            return actions[0]

        # Compute the maximum score
        max_score = max(score for (_, score) in scores)
        # Filter all actions that have the max score (handle ties)
        best_candidates = [a for (a, s) in scores if s == max_score]

        if not best_candidates or max_score == float('-inf'):
            # All evaluations failed; fallback
            if DEBUG:
                print('FooPlayer.decide: all evaluations failed -> defaulting to first action')
            return actions[0]

        chosen = random.choice(best_candidates)
        if DEBUG:
            try:
                # Try to pretty-print a small summary for debugging
                print(f'FooPlayer.decide: selected action -> {repr(chosen)} with score {max_score}')
            except Exception:
                print('FooPlayer.decide: selected an action (repr failed)')

        return chosen

    def _evaluate_state(self, game):
        """Static evaluation of a game state from this player's perspective.

        The evaluator attempts multiple common access patterns to extract
        victory points and simple progress indicators (settlements, cities,
        roads). The returned score is primarily the victory points (higher is
        better). Secondary counts are used as small tiebreakers.

        This function is defensive to avoid attribute errors across different
        engine versions.

        Returns:
            float: heuristic score for the state (larger is better)
        """
        color = self.color
        vp = None
        settlements = None
        cities = None
        roads = None

        # Try a number of plausible attribute access patterns. Use try/except
        # blocks liberally because different engine versions expose different
        # structures.
        try:
            players = game.state.players
        except Exception:
            players = None

        # Attempt to access player state by Color key
        player_state = None
        if players is not None:
            try:
                player_state = players[color]
            except Exception:
                # Maybe players is a list keyed by integer colors
                try:
                    idx = int(color)
                    player_state = players[idx]
                except Exception:
                    player_state = None

        # Extract victory points with common attribute names
        if player_state is not None:
            for attr in ('victory_points', 'victoryPoints', 'vp', 'points'):
                try:
                    val = getattr(player_state, attr)
                    # If it's a callable (method), call it
                    if callable(val):
                        val = val()
                    vp = int(val)
                    break
                except Exception:
                    vp = None

            # Try dictionary-style if attributes failed
            if vp is None:
                try:
                    if isinstance(player_state, dict):
                        for key in ('victory_points', 'vp', 'points'):
                            if key in player_state:
                                vp = int(player_state[key])
                                break
                except Exception:
                    vp = None

            # Extract simple asset counts to break ties
            for attr in ('settlements', 'settle_count', 'settlement_count', 'settles'):
                try:
                    val = getattr(player_state, attr)
                    if callable(val):
                        val = val()
                    settlements = int(val)
                    break
                except Exception:
                    settlements = None

            for attr in ('cities', 'city_count'):
                try:
                    val = getattr(player_state, attr)
                    if callable(val):
                        val = val()
                    cities = int(val)
                    break
                except Exception:
                    cities = None

            for attr in ('roads', 'road_count'):
                try:
                    val = getattr(player_state, attr)
                    if callable(val):
                        val = val()
                    roads = int(val)
                    break
                except Exception:
                    roads = None

        # Fallbacks if extraction failed: try to compute from visible board pieces
        # (e.g., lengths of lists). This is optional and best-effort.
        if vp is None and players is not None:
            try:
                # If player_state contains lists of pieces, inspect lengths
                if isinstance(player_state, dict):
                    # Look for settlement/city lists
                    s = None
                    for key in ('settlements', 'settle_list'):
                        if key in player_state and isinstance(player_state[key], (list, tuple)):
                            s = len(player_state[key])
                            break
                    if s is not None:
                        settlements = settlements or s
                # We intentionally do not try to derive vp from the board in a
                # brittle way; leave vp as None and fall back to 0.
            except Exception:
                pass

        # Final fallback: if we couldn't determine vp, set to 0
        if vp is None:
            vp = 0

        # Build a composite score. Main contributor is victory points. Add
        # small weighted bonuses for settlements/cities/roads if available.
        score = float(vp)
        if settlements is not None:
            score += 0.01 * float(settlements)
        if cities is not None:
            score += 0.02 * float(cities)
        if roads is not None:
            score += 0.005 * float(roads)

        return score
