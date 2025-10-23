import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)


from catanatron import Game, RandomPlayer, Color

from agents.promptEvolver.creator_agent import CreatorAgent as promptEvolver
from agents.agentEvolver.creator_agent import CreatorAgent as agentEvolver
from agents.agentEvolver_v2.creator_agent import CreatorAgent as agentEvolverV2
from agents.llmAgentEvolver.creator_agent import CreatorAgent as llmAgentEvolver

from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

# Set this variable to select the evolver: "prompt", "agent", or "llm"
EVOLVER_TYPE = "agentEvolver2"

def main():
    # Choose evolver based on EVOLVER_TYPE
    if EVOLVER_TYPE == "promptEvolver":
        evolver = promptEvolver()
    elif EVOLVER_TYPE == "agentEvolver":
        evolver = agentEvolver()
    elif EVOLVER_TYPE == "agentEvolver2":
        evolver = agentEvolverV2()
    elif EVOLVER_TYPE == "llmAgentEvolver":
        evolver = llmAgentEvolver()
    else:
        raise ValueError(f"Unknown EVOLVER_TYPE: {EVOLVER_TYPE}")

    # Run The Evolver
    evolver.run_react_graph()



if __name__ == "__main__":
    main()