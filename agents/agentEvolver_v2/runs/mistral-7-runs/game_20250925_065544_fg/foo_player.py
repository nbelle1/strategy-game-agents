from .adapters import (
    Game,
    Player,
    Color,
    Action,
    ActionType,
    copy_game,
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    list_prunned_actions,
    prune_robber_actions,
    base_fn,
    value_production,
    get_value_fn,
)

import math
import time
import random
from typing import List, Optional


class FooPlayer(Player):
    """
    FooPlayer implementing a Monte Carlo Tree Search (MCTS) player.

    This replaces the earlier Expectimax approach with an adaptive MCTS that:
    - Uses UCB1 for selection (exploration vs exploitation)
    - Samples probabilistic outcomes via execute_spectrum during expansion/simulation
    - Uses base_fn to evaluate leaf/terminal states
    - Uses adapters' pruning helpers (list_prunned_actions, prune_robber_actions)

    Notes:
    - All interactions with the game use the adapters surface as required.
    - Plenty of defensive try/except blocks and print() calls are included to help
      trace decisions and debug runtime behavior while evolving the player.
    """

    def __init__(self, name=None, iterations: int = 1000, node_budget: int = 15000, exploration_const: float = math.sqrt(2)):
        # Initialize player color and name
        super().__init__(Color.BLUE, name)
        # How many MCTS iterations to run per decision (subject to node_budget)
        self.iterations = iterations
        # Hard cap on number of node expansions / evaluations
        self.node_budget = max(100, int(node_budget))
        # Exploration constant for UCB1
        self.exploration_const = float(exploration_const)

    def _get_game_phase(self, game: Game) -> str:
        """Robust game phase detection (early/mid/late) using available turn counters.
        This mirrors previous logic to allow phase-adaptive behaviors in simulations
        if desired later. For now MCTS uses base_fn for evaluation.
        """
        try:
            turn_count = getattr(game, 'turn_count', None)
            if turn_count is None:
                st = getattr(game, 'state', None)
                if isinstance(st, dict):
                    turn_count = st.get('turn_count') or st.get('turn') or st.get('tick')
                else:
                    turn_count = getattr(st, 'turn_count', None) if st is not None else None

            if turn_count is None:
                turn_count = 0
            turn_count = int(turn_count)
        except Exception:
            turn_count = 0

        if turn_count < 20:
            return 'early'
        elif turn_count < 40:
            return 'mid'
        else:
            return 'late'

    class MCTSNode:
        """Node in the MCTS tree.

        Attributes:
            game: Game state at this node
            parent: parent node or None
            action: Action that led from parent -> this node (None for root)
            children: list of child nodes
            visits: number of times node visited
            total_reward: cumulative reward (for our player) accumulated
            untried_actions: actions available at this node that are not yet expanded
        """

        def __init__(self, game: Game, parent: Optional['FooPlayer.MCTSNode'] = None, action: Optional[Action] = None):
            self.game = game
            self.parent = parent
            self.action = action
            self.children: List['FooPlayer.MCTSNode'] = []
            self.visits: int = 0
            self.total_reward: float = 0.0
            self.untried_actions: Optional[List[Action]] = None

        def is_fully_expanded(self) -> bool:
            return self.untried_actions is not None and len(self.untried_actions) == 0

        def best_child_by_ucb(self, exploration_const: float) -> Optional['FooPlayer.MCTSNode']:
            """Select child with highest UCB1 score."""
            if not self.children:
                return None
            log_parent = math.log(max(1, self.visits))
            best = None
            best_score = -math.inf
            for c in self.children:
                if c.visits == 0:
                    # Encourage unvisited children
                    score = math.inf
                else:
                    exploitation = c.total_reward / c.visits
                    exploration = exploration_const * math.sqrt(log_parent / c.visits)
                    score = exploitation + exploration
                if score > best_score:
                    best_score = score
                    best = c
            return best

    def decide(self, game: Game, playable_actions):
        """Run MCTS and select the best action.

        High-level flow:
          - Create root node for current game
          - For up to self.iterations (bounded by node_budget):
              - Selection: traverse tree via UCB1 until a node with untried actions or terminal is found
              - Expansion: expand one untried action (sample an outcome) and add child
              - Simulation: simulate a random playout from child (sampling chance outcomes) until terminal or depth limit
              - Backpropagation: propagate reward (base_fn relative to self.color) up the tree
          - Choose the root child with max visits (robust) or max average reward as final action

        The implementation samples probabilistic outcomes using execute_spectrum and falls back to execute_deterministic.
        """
        actions = list(playable_actions) if playable_actions is not None else []

        if not actions:
            print('FooPlayer.decide: No playable actions; returning None')
            return None

        # Setup
        iterations = max(1, int(self.iterations))
        node_budget = max(100, int(self.node_budget))
        exploration_const = float(self.exploration_const)
        base_value_fn = base_fn()

        node_count = 0  # counts expansions / simulations roughly
        start_time = time.time()

        # Create root node
        root_game = copy_game(game)
        root = FooPlayer.MCTSNode(root_game)

        # Initialize root untried actions defensively using pruned list helper
        try:
            root.untried_actions = list_prunned_actions(root_game) or []
        except Exception:
            root.untried_actions = list(actions)

        # Helper: sample an outcome from execute_spectrum's outcomes list
        def _sample_outcome(outcomes):
            # outcomes: list of (game, prob)
            if not outcomes:
                return None
            if len(outcomes) == 1:
                return outcomes[0][0]
            # sample by probability
            r = random.random()
            cum = 0.0
            for (g, p) in outcomes:
                cum += float(p)
                if r <= cum:
                    return g
            # Fallback to last
            return outcomes[-1][0]

        # Helper: get legal/pruned actions at a node (defensive)
        def _legal_actions_for(g: Game):
            try:
                acts = list_prunned_actions(g) or []
            except Exception:
                # Exhaustive fallback: no pruning available, try expand_spectrum or empty
                try:
                    acts = []
                except Exception:
                    acts = []
            return list(acts)

        # Helper: select an action for simulation playouts (avoid END_TURN/ROLL when possible)
        def _simulation_policy(g: Game):
            acts = _legal_actions_for(g)
            if not acts:
                return None
            # try to filter out passive actions if there are alternatives
            non_passive = [a for a in acts if getattr(a, 'action_type', None) not in (ActionType.END_TURN, ActionType.ROLL)]
            if non_passive:
                return random.choice(non_passive)
            return random.choice(acts)

        # Terminal detection using winning_color if available
        def _is_terminal(g: Game) -> bool:
            try:
                w = g.winning_color()
                return w is not None
            except Exception:
                # No winning_color API? Fallback heuristics could be added; assume not terminal
                return False

        # Simulation: play random (but slightly biased) moves until terminal or depth limit
        def _simulate_from(g: Game, max_sim_depth: int = 50) -> float:
            nonlocal node_count
            sim_game = copy_game(g)
            depth = 0
            while depth < max_sim_depth and not _is_terminal(sim_game):
                act = _simulation_policy(sim_game)
                if act is None:
                    break
                # Execute (sample) an outcome for this action
                try:
                    outcomes = execute_spectrum(sim_game, act)
                except Exception:
                    try:
                        outcomes = execute_deterministic(sim_game, act)
                    except Exception:
                        outcomes = []
                if not outcomes:
                    break
                chosen_after = _sample_outcome(outcomes)
                if chosen_after is None:
                    break
                sim_game = chosen_after
                depth += 1
                node_count += 1
                if node_count > node_budget:
                    # stop simulation early if we reached budget
                    break
            # Evaluate final state for our player
            try:
                val = base_value_fn(sim_game, self.color)
            except Exception as e:
                print(f'FooPlayer._simulate_from: base_fn evaluation failed: {e}')
                val = -1e9
            return float(val)

        # Backpropagation updates node statistics with reward
        def _backpropagate(node: FooPlayer.MCTSNode, reward: float):
            while node is not None:
                node.visits += 1
                node.total_reward += reward
                node = node.parent

        # Expand one action from node: pick an untried action, sample outcome, create child
        def _expand(node: FooPlayer.MCTSNode) -> Optional[FooPlayer.MCTSNode]:
            nonlocal node_count
            if node.untried_actions is None:
                node.untried_actions = _legal_actions_for(node.game)
            if not node.untried_actions:
                return None
            # Pop one action to expand
            try:
                action = node.untried_actions.pop()
            except Exception:
                return None
            # Execute and sample an outcome to create a deterministic child state
            try:
                outcomes = execute_spectrum(node.game, action)
            except Exception:
                try:
                    outcomes = execute_deterministic(node.game, action)
                except Exception:
                    outcomes = []

            if not outcomes:
                return None

            g_after = _sample_outcome(outcomes)
            if g_after is None:
                return None

            child = FooPlayer.MCTSNode(copy_game(g_after), parent=node, action=action)
            # Initialize child's untried actions lazily
            child.untried_actions = None
            node.children.append(child)
            node_count += 1
            return child

        # Selection: traverse from root using UCB1 until a node with untried actions or terminal
        def _select(node: FooPlayer.MCTSNode) -> FooPlayer.MCTSNode:
            current = node
            while True:
                if _is_terminal(current.game):
                    return current
                # initialize untried_actions if needed
                if current.untried_actions is None:
                    current.untried_actions = _legal_actions_for(current.game)
                if current.untried_actions:
                    # node has untried actions -> stop at current (expandable)
                    return current
                # otherwise fully expanded: move to best child by UCB
                best = current.best_child_by_ucb(exploration_const)
                if best is None:
                    return current
                current = best

        # Main MCTS loop
        print(f'FooPlayer.decide: Starting MCTS with iterations={iterations}, node_budget={node_budget}')
        iters = 0
        try:
            for it in range(iterations):
                if node_count > node_budget:
                    print('FooPlayer.decide: node_budget reached; stopping iterations')
                    break
                iters += 1
                # 1. Selection
                leaf = _select(root)

                # 2. Expansion
                if not _is_terminal(leaf.game):
                    child = _expand(leaf)
                    if child is None:
                        # Could not expand (no outcomes); treat leaf as child for simulation
                        node_to_simulate = leaf
                    else:
                        node_to_simulate = child
                else:
                    node_to_simulate = leaf

                # 3. Simulation
                reward = _simulate_from(node_to_simulate.game)

                # 4. Backpropagation
                _backpropagate(node_to_simulate, reward)

            # Completed iterations or budget
        except Exception as e:
            print(f'FooPlayer.decide: Exception during MCTS main loop: {e}')

        # Choose the best action: child of root with highest visit count (robust) or highest avg reward
        best_child = None
        best_visits = -1
        best_avg = -math.inf
        for c in root.children:
            avg = (c.total_reward / c.visits) if c.visits > 0 else -math.inf
            # prefer visits first
            if c.visits > best_visits or (c.visits == best_visits and avg > best_avg):
                best_child = c
                best_visits = c.visits
                best_avg = avg

        chosen_action = None
        if best_child is not None:
            chosen_action = best_child.action
        else:
            # Fallback: choose highest-priority playable action
            try:
                actions_sorted = sorted(actions, key=lambda a: 0 if getattr(a, 'action_type', None) not in (ActionType.END_TURN, ActionType.ROLL) else -1)
                chosen_action = actions_sorted[0]
            except Exception:
                chosen_action = actions[0]

        elapsed = time.time() - start_time
        print(f'FooPlayer.decide: MCTS finished iterations={iters}, node_count={node_count}, time={elapsed:.3f}s')
        print(f'FooPlayer.decide: Chosen action = {chosen_action} (visits={best_visits}, avg={best_avg:.3f})')

        return chosen_action
