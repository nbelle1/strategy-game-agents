from collections import namedtuple

from rich.table import Table

from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

# from catanatron_experimental.mcts_score_collector import (
#     MCTSScoreCollector,
#     MCTSPredictor,
# )
# from catanatron_experimental.machine_learning.players.reinforcement import (
#     QRLPlayer,
#     TensorRLPlayer,
#     VRLPlayer,
#     PRLPlayer,
# )
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    SameTurnAlphaBetaPlayer,
)
from catanatron.players.search import VictoryPointPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.machine_learning.players.playouts import (
    GreedyPlayoutsPlayer,
)

from agents.structuredAgent.llm_player import StructuredAgentPlayer
from agents.baseAgent.baseAgentPlayer import BaseAgentPlayer
from agents.promptEvolver.promptEvolverPlayer import PromptEvolverPlayer
from agents.agentEvolver.foo_player import FooPlayer as AgentEvolverFooPlayer
from agents.llmAgentEvolver.foo_player import FooPlayer as LLMAgentEvolverFooPlayer
from agents.agentEvolver_v2.foo_player import FooPlayer as AgentEvolverFooPlayerV2
from agents.best_player.foo_player import FooPlayer as BestPlayer

# from agents.fromScratchLLM_player_v2.runs.creator_20250508_112135_hitl.foo_player import FooPlayer as FooLLMPlayerV2_1
# from catanatron_experimental.machine_learning.players.online_mcts_dqn import (
#     OnlineMCTSDQNPlayer,
# )

# PLAYER_CLASSES = {
#     "O": OnlineMCTSDQNPlayer,
#     "S": ScikitPlayer,
#     "Y": MyPlayer,
#     # Used like: --players=V:path/to/model.model,T:path/to.model
#     "C": ForcePlayer,
#     "VRL": VRLPlayer,
#     "Q": QRLPlayer,
#     "P": PRLPlayer,
#     "T": TensorRLPlayer,
#     "D": DQNPlayer,
#     "CO": MCTSScoreCollector,
#     "COP": MCTSPredictor,
# }

# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])
CLI_PLAYERS = [
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
    # NEW PLAYERS BELOW
    CliPlayer(
        "BA",
        "BaseAgentPlayer",
        "Initial Base Player utilizing LLM with no special prompt",
        BaseAgentPlayer,
    ),
    CliPlayer(
        "SA",
        "StructuredAgentPlayer",
        "LLM with adjusted prompt and code to fix bugs with vanilla llm",
        StructuredAgentPlayer,
    ),
    CliPlayer(
        "PE",
        "PromptEvolverPlayer",
        "Multi-agent system with prompt evolvers that improve the prompt of the Player PE",
        PromptEvolverPlayer,
    ),
    CliPlayer(
        "AE",
        "AgentEvolverFooPlayer",
        "Multi-agent system with agent evolvers that improve the player code of the Player AE",
        AgentEvolverFooPlayer,
    ),
    CliPlayer(
        "AE2",
        "AgentEvolverFooPlayerV2",
        "Multi-agent system with agent evolvers that improve the player code of the Player AE2",
        AgentEvolverFooPlayerV2,
    ),
    CliPlayer(
        "LAE",
        "LLMAgentEvolverFooPlayer",
        "Multi-agent system with agent evolvers that improve the LLM player code of the Player LAE",
        LLMAgentEvolverFooPlayer,
    ),
        CliPlayer(
        "BP",
        "BestPlayer",
        "Multi-agent system with agent evolvers that improve the LLM player code of the Player LAE",
        BestPlayer,
    ),
]


def register_player(code):
    def decorator(player_class):
        CLI_PLAYERS.append(
            CliPlayer(
                code,
                player_class.__name__,
                player_class.__doc__,
                player_class,
            ),
        )

    return decorator


CUSTOM_ACCUMULATORS = []


def register_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table
