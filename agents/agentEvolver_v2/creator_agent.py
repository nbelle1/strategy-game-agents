import os
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from typing import List, Dict, Tuple, Any, Optional
import json
from datetime import datetime
import shutil
from pathlib import Path
import subprocess, shlex


from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_aws import ChatBedrockConverse
from langgraph.errors import GraphRecursionError
from typing_extensions import TypedDict




###################################################################################################
#  CONFIG / CONSTANTS / VARIABLES
###################################################################################################
# CONFIG
LANGCHAIN_TRACING_VAR = "false"

# Coder LLM
CODER_LLM_BACKEND = "mistral"
CODER_LLM_MODEL = "codestral-latest"

# CODER_LLM_BACKEND = "openai"
# CODER_LLM_MODEL = "gpt-5-mini"

# Analyzer LLM
ANALYZER_LLM_BACKEND = "mistral"
ANALYZER_LLM_MODEL = "mistral-large-latest"

# Researcher LLM
RESEARCHER_LLM_BACKEND = "mistral"
RESEARCHER_LLM_MODEL = "devstral-medium-latest"

# Strategizer LLM
STRATEGIZER_LLM_BACKEND = "mistral"
STRATEGIZER_LLM_MODEL = "mistral-large-latest"

# Meta LLM
META_LLM_BACKEND = "openai"
META_LLM_MODEL = "gpt-5-mini"

FOO_MAX_BYTES   = 64_000      # context-friendly cap
CREATOR_LANGRAPH_RECURSION_LIMIT = 200  # max depth of graph recursion
CREATOR_NUM_EVOLUTIONS = 20

MAX_MESSAGES_TOOL_CALLING = 4
MAX_META_MESSAGES_GIVEN_TO_CODER = 6
MAX_MESSAGES_IN_AGENT = 20

# Catanatron
FOO_RUN_COMMAND = "catanatron-play --players=AB,AE2 --num=30 --config-map=MINI --config-vps-to-win=10"
# caffeinate -i catanatron-play --players=AB,BP --num=1 --config-vps-to-win=5
# catanatron-play --players=F,BP --num=30 --config-map=MINI --config-vps-to-win=10

# Phase control: "auto" | "discovery" | "improvement"
START_PHASE = "improvement"

# CONSTANTS
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path

MULTI_AGENT_PROMPT = f"""You are apart of a multi-agent system that is working to evolve the code in {FOO_TARGET_FILENAME} to become the best player in the Catanatron Minigame.\n\tYour specific role is the:"""
ANALYZER_NAME = "ANALYZER"
STRATEGIZER_NAME = "STRATEGIZER"
RESEARCHER_NAME = "RESEARCHER"
CODER_NAME = "CODER"
ADAPTER_ANALYZER_NAME = "ADAPTER_ANALYZER"
ADAPTER_RESEARCHER_NAME = "ADAPTER_RESEARCHER"
ADAPTER_CODER_NAME = "ADAPTER_CODER"

AGENT_KEYS = [ANALYZER_NAME, STRATEGIZER_NAME, RESEARCHER_NAME, CODER_NAME,
              ADAPTER_ANALYZER_NAME, ADAPTER_RESEARCHER_NAME, ADAPTER_CODER_NAME]


LLM_POLICY = f"""
<LLM USAGE POLICY (read carefully)>
- Purpose: The adapters.py file exposes a stable function **call_llm(...)** that routes a prompt to an external LLM.
  You may and should use this function **as a strategic “oracle”** when it can overcome blind spots in the current heuristic/search.
- What to ask it for (examples):
  1) **Action tie-breaks**: When multiple legal actions have near-identical scores, ask the LLM to choose based on long-horizon board impact.
  2) **Heuristic shaping**: Ask for a small, explicit, numeric weighting tweak for the state (e.g., priority for ore/wheat vs. brick/wood given dev/road plans).
  3) **Contingency rules**: Ask for a short, explicit rule (“If X holds, prefer Y”) to break cycles or avoid traps observed in analysis logs.
  4) **Opening books**: Ask for a crisp opening principle for first N turns, conditioned on visible starting placements/ports/tiles.
  5) **Trade & dev card timing**: Ask for a concise policy (e.g., when to buy vs. build roads/cities given bankroll and board).
- What NOT to ask:
  - Don’t ask for raw code; that is the CODER’s job.
  - Don’t ask for hidden game info; only query using adapter-visible state.
  - Don’t rely on LLM hallucinations about Catanatron APIs—**bind outputs back to adapter-verified actions**.
- Inputs to call_llm:
  - You must **read adapters.py to confirm the exact signature**. If uncertain, first call read_adapter.
  - Provide a compact, **serializable state sketch** (e.g., adapter.get_state_summary/game.to_dict) and the set of **legal actions** (adapter.list_actions) in plain text or JSON.
  - Keep prompts short and structured. Ask for **one** of: best_action_id, ranked_actions, or numeric weights per action_id.
- Output handling:
  - Parse defensively. If the model suggests an **illegal or unknown** action_id, **ignore and fall back** to local heuristics (and log it).
  - Prefer **deterministic mapping**: if the LLM returns weights, normalize and combine with local score = α*local + β*llm; use small β initially (e.g., 0.15–0.35).
- Cost/latency hygiene:
  - Only call once per decision point **when necessary**. Add a **memo/cache** keyed by a compact state hash + sorted legal actions.
  - Add a global **max_calls_per_game** (e.g., 5–15) and stop using LLM if exceeded.
- Telemetry & safety:
  - Log every LLM call with: state_hash, truncated prompt, returned object, chosen action, and fallback reason if used.
  - Wrap the call in try/except; on any error, **no-crash fallback** to local heuristic/search.
- Schema-first contract:
  - During improvement, you may ONLY read state via names exported by adapters.py (e.g., FULL_SCHEMA, getp, state_core, legal_actions, actions_id_map, my_hand, my_vps).
  - Do NOT traverse engine objects (no game.state.*, no board.*) and do NOT invent keys not present in FULL_SCHEMA.
  - Before using any field, confirm it exists by referencing FULL_SCHEMA and (optionally) gating reads through getp(...).
  - If a required field is missing, fall back gracefully or ask the Strategizer to extend the adapter in the next iteration.
</LLM USAGE POLICY>
"""

# NEW: Adapter artifact targets
ADAPTER_TARGET_FILENAME = "adapters.py"
ADAPTER_TARGET_FILE = Path(__file__).parent / ADAPTER_TARGET_FILENAME

# Optional: tiny starter template the ADAPTER_CODER can write on first pass
ADAPTER_TEMPLATE = """
# adapters.py – created by Discovery Phase
# Contract: see Adapter interface (environment-agnostic).
from typing import Any, Iterable, Optional

class CatanatronAdapter:
    # ---- core transition API ----
    def new_game(self, seed: Optional[int] = None) -> Any:
        from catanatron.game import Game
        # Seed is ignored by base Game; Strategizer may later expose if available.
        return Game()

    def clone(self, g: Any) -> Any:
        return g.copy()

    def legal_actions(self, g: Any) -> Iterable[Any]:
        return list(g.state.playable_actions)

    def apply(self, g: Any, a: Any) -> Any:
        g2 = g.copy()
        # Try both common entry points
        if hasattr(g2, "apply"):
            g2.apply(a)
        elif hasattr(g2.state, "apply"):
            g2.state.apply(a)
        else:
            raise AttributeError("No apply() method found on Game or Game.state")
        return g2

    def current_player(self, g: Any) -> Any:
        return g.state.current_color()

    def is_terminal(self, g: Any) -> bool:
        return g.winning_color() is not None

    def winner(self, g: Any) -> Optional[Any]:
        return g.winning_color()

    # ---- optional helpers ----
    def serialize_action(self, a: Any) -> str:
        return str(a)
"""


# VARIABLES


###################################################################################################
#  LANGGRAPH
###################################################################################################
class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None
    current_evolution = 0

    def _create_llm(self, backend, model):
        if backend == "openai":
            return ChatOpenAI(
                model=model,
                max_retries=10,
            )
        elif backend == "mistral":
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=1,    # Adjust based on your API tier
                check_every_n_seconds=0.1,
            )
            return ChatMistralAI(
                model=model,
                temperature=0,
                max_retries=10,
                rate_limiter=rate_limiter,
            )
        elif backend == "claude":
            return ChatBedrockConverse(
                aws_access_key_id = os.environ["AWS_ACESS_KEY"],
                aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
                region_name = "us-east-2",
                provider = "anthropic",
                model_id="# TODO: ADD MODEL ID"
            )
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {backend}")
    
    def log_config_settings(self):
        config_path = os.path.join(CreatorAgent.run_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"LANGCHAIN_TRACING_VAR = {LANGCHAIN_TRACING_VAR}\n")
            f.write(f"CODER_LLM_BACKEND = {CODER_LLM_BACKEND}\n")
            f.write(f"CODER_LLM_MODEL = {CODER_LLM_MODEL}\n")
            f.write(f"ANALYZER_LLM_BACKEND = {ANALYZER_LLM_BACKEND}\n")
            f.write(f"ANALYZER_LLM_MODEL = {ANALYZER_LLM_MODEL}\n")
            f.write(f"RESEARCHER_LLM_BACKEND = {RESEARCHER_LLM_BACKEND}\n")
            f.write(f"RESEARCHER_LLM_MODEL = {RESEARCHER_LLM_MODEL}\n")
            f.write(f"STRATEGIZER_LLM_BACKEND = {STRATEGIZER_LLM_BACKEND}\n")
            f.write(f"STRATEGIZER_LLM_MODEL = {STRATEGIZER_LLM_MODEL}\n")
            f.write(f"META_LLM_BACKEND = {META_LLM_BACKEND}\n")
            f.write(f"META_LLM_MODEL = {META_LLM_MODEL}\n")
            f.write(f"FOO_MAX_BYTES = {FOO_MAX_BYTES}\n")
            f.write(f"CREATOR_LANGRAPH_RECURSION_LIMIT = {CREATOR_LANGRAPH_RECURSION_LIMIT}\n")
            f.write(f"CREATOR_NUM_EVOLUTIONS = {CREATOR_NUM_EVOLUTIONS}\n")
            f.write(f"MAX_MESSAGES_TOOL_CALLING = {MAX_MESSAGES_TOOL_CALLING}\n")
            f.write(f"MAX_META_MESSAGES_GIVEN_TO_CODER = {MAX_META_MESSAGES_GIVEN_TO_CODER}\n")
            f.write(f"MAX_MESSAGES_IN_AGENT = {MAX_MESSAGES_IN_AGENT}\n")
            f.write(f"FOO_RUN_COMMAND = {FOO_RUN_COMMAND}\n")
            f.write(f"START_PHASE = {START_PHASE}\n")

    def __init__(self):
        # Get API key from environment variable
        self.coder_llm = self._create_llm(CODER_LLM_BACKEND, CODER_LLM_MODEL)
        self.analyzer_llm = self._create_llm(ANALYZER_LLM_BACKEND, ANALYZER_LLM_MODEL)
        self.researcher_llm = self._create_llm(RESEARCHER_LLM_BACKEND, RESEARCHER_LLM_MODEL)
        self.strategizer_llm = self._create_llm(STRATEGIZER_LLM_BACKEND, STRATEGIZER_LLM_MODEL)
        self.meta_llm = self._create_llm(META_LLM_BACKEND, META_LLM_MODEL)

        # Optionally set tracing
        os.environ["LANGCHAIN_TRACING"] = LANGCHAIN_TRACING_VAR

        # Create run directory if it doesn't exist
        if CreatorAgent.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("creator_%Y%m%d_%H%M%S")
            CreatorAgent.run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(CreatorAgent.run_dir, exist_ok=True)

        #Copy the Blank FooPlayer to the run directory
        shutil.copy2(                           # ↩ copy with metadata
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),  # ../foo_player.py
            FOO_TARGET_FILE.resolve()          # ./foo_player.py
        )

        self.config = {
            "recursion_limit": CREATOR_LANGRAPH_RECURSION_LIMIT, # set recursion limit for graph
        }

        self.log_config_settings()

        self.react_graph = self.create_langchain_react_graph()

    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        

        class CreatorGraphState(TypedDict):
            meta_messages: list[AnyMessage] # Messages from the meta node (used for debugging)
            analyzer_messages: list[AnyMessage] # Messages from the analyzer node (used for debugging)
            strategizer_messages: list[AnyMessage] # Messages from the strategizer node (used for debugging)
            researcher_messages: list[AnyMessage] # Messages from the researcher node (used for debugging)
            coder_messages: list[AnyMessage] # Messages from the coder node (used for debugging)

            recent_meta_message: HumanMessage # Recent Message from the meta node (used for debugging)
            recent_helper_response: HumanMessage # Recent Message from the helper node (used for debugging)
            game_results: HumanMessage # Last results of running the game
            tool_calling_messages: list[AnyMessage] # Messages from the tool calling state graph
            phase: str                           # "discovery" | "improvement"
            adapter_test_report: HumanMessage     # last adapter test JSON / summary

        def tool_calling_state_graph(llm, agent_name, sys_msg: SystemMessage, msgs: list[AnyMessage], tools):
            # Bind Tools to the LLM
            #llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
            llm_with_tools = llm.bind_tools(tools)

            def assistant(sub_state: MessagesState):
                return {"messages": [llm_with_tools.invoke([sys_msg] + sub_state["messages"])]}

            # Graph
            builder = StateGraph(MessagesState)

            # Define nodes: these do the work
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))
            #builder.add_node("final_assistant", final_assistant)

            # Define edges: these determine how the control flow moves
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition,
            )
            #builder.add_conditional_edges("tools", check_num_messages, "assistant", "final_assistant")
            builder.add_edge("tools", "assistant")
            react_graph = builder.compile()
            
            # Run Graph
            for event in react_graph.stream({"messages": msgs}, stream_mode="values"):
                msg = event['messages'][-1]
                msg.pretty_print()
                print("\n")
                last_event = event

            # Save tools to continuouslog file
            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_tools.txt")
            with open(log_path, "a") as log_file:
                for m in last_event['messages']:
                    log_file.write(m.pretty_repr() + "\n")

            # Create messages_tools directory if needed
            os.makedirs(os.path.join(CreatorAgent.run_dir, "messages_tools"), exist_ok=True)

            # Save tools to log file for individual agents
            log_path = os.path.join(CreatorAgent.run_dir, "messages_tools", f"llm_log_{agent_name}_tools.txt")
            with open(log_path, "a") as log_file:
                for m in last_event['messages']:
                    log_file.write(m.pretty_repr() + "\n")

            return last_event

        def init_node(state: CreatorGraphState):
            print("In Init Node")

            sp = (str(START_PHASE).lower() if START_PHASE is not None else "auto")
            if sp in ("discovery", "improvement"):
                phase = sp
            else:
                phase = "improvement" if ADAPTER_TARGET_FILE.exists() else "discovery"

            return {
                "meta_messages": [],
                "analyzer_messages": [],
                "strategizer_messages": [],
                "researcher_messages": [],
                "coder_messages": [],
                "recent_meta_message": HumanMessage(content=""),
                "recent_helper_response": HumanMessage(content=""),
                "game_results": HumanMessage(content=""),
                "tool_calling_messages": [],
                "phase": phase,
                "adapter_test_report": HumanMessage(content=""),
            }


        def run_player_node(state: CreatorGraphState):
            """
            If in Discovery: run adapter tests and prime the ADAPTER_ANALYZER.
            If in Improvement: run the Catan match and prime the ANALYZER.
            """
            
            # FIX ME: Discovery phase
            if state.get("phase") == "discovery":
                report = run_adapter_tests()
                msg = HumanMessage(content=f"ADAPTER TEST RESULTS:\n\n{report}")

                # Default objective for the adapter-analyzer
                default_adapter_analyze_msg = HumanMessage(content=f"""
                    ADAPTER_ANALYZER OBJECTIVE:

                    - Parse the JSON in the latest adapter test report (above).
                    - If any tests failed: list them with tight, line-precise guidance on what to change in adapters.py.
                    - If all tests passed: say so, and recommend switching to IMPROVEMENT phase.

                    Keep it concise. Start with "After Running the Adapter Tests, here are my findings:".
                """)
                return {
                    "adapter_test_report": msg,
                    "recent_meta_message": default_adapter_analyze_msg,
                    "meta_messages": state["meta_messages"] + [msg],
                }

            # === Improvement phase: original behavior ===
            game_results = run_testfoo(short_game=False)
            game_msg = HumanMessage(content=f"GAME RESULTS:\n\n{game_results}")

            defualt_analyze_msg = HumanMessage(content=f"""
ANALYZER OBJECTIVE:

Start with: "After Running The New {FOO_TARGET_FILENAME} Player, Here is my analysis and findings:"

If the game failed to compile/run (no game_results JSON or score==0):
- ERROR SUMMARY:
  - First error line (verbatim), exception type, file, exact line number, and the exact code line (from game_output.txt).
- LIKELY CAUSE (1–2 bullets): short hypothesis based on the error/log text (e.g., unknown ActionType, bad import, attribute missing).
- QUICK FIX FOCUS: 1–2 bullets pointing to the specific function/line in {FOO_TARGET_FILENAME} (or adapters.py) to inspect.

If the game ran (game_results JSON present):
1) PERFORMANCE SUMMARY:
   - Outcome (Win/Loss), our VP vs opponent VP, VP diff.
   - Key counts: cities, settlements, roads, dev cards (if available), total turns.
2) VERDICT:
   - Good if Win OR VP diff ≥ +0.5
   - Borderline if −0.5 < VP diff < +0.5
   - Poor if Loss OR VP diff ≤ −0.5
3) IF BORDERLINE/POOR — LIKELY REASONS:
   - Briefly scan {FOO_TARGET_FILENAME} and list 2–4 concrete issues with short citations (line numbers/snippets), prioritizing:
     - Missing 1-ply value lookahead (no `copy_game` + `make_value_fn` usage).
     - No chance handling (dice/dev/robber), or robber/knight policy absent.
     - Placement helpers stubbed/always False (roads/settlements).
     - No end-turn policy or repeated random selection.
     - Illegal/unknown actions (e.g., trying to play `VICTORY_POINT`).
   - Pull 2–4 corroborating log lines from game_output.txt (e.g., "Unrecognized action type", "Defaulting to Random Action", stack traces).
4) NEXT STEP (one line):
   - Clear route like: "Send to Coder to add 1-ply value lookahead", or "Send to Strategizer to specify robber/placement policy", etc.

End with: "Let me know if you need anything else".

            """)
            return {
                "game_results": game_msg,
                "recent_meta_message": defualt_analyze_msg,
                "meta_messages": state["meta_messages"] + [game_msg],
            }


        def meta_node(state: CreatorGraphState):

            sys_msg = SystemMessage(
                content=f"""
{MULTI_AGENT_PROMPT} META SUPERVISOR

### Task 
You are the **Lead Scientist** of an AI research team. Your primary role is to guide your team of specialized AI agents through a rigorous cycle of experimentation. Your thinking must be critical, logical, and focused on the scientific method.

### META HIGH LEVEL GOAL 
Systematically improve the `foo_player.py` code until it can consistently win against the AlphaBeta opponent in Catan.

<Performance History>
Here is your Current Performance History for Evolving the {FOO_TARGET_FILENAME} player:
{read_full_performance_history()}
</Performance History>

### The Experimental Workflow 
Your team operates in a strict cycle. You must guide them through these steps in order: 
1. **Analyze:** After a game is played, you MUST first call the **ANALYZER** to diagnose *why* the player won or lost. Your objective for the Analyzer must be to find the root cause. 
2. **Strategize:** Once the **ANALYZER** identifies a strategic flaw, you will call the **STRATEGIZER** to propose a solution to that specific flaw. 
3. **Code:** Once the **STRATEGIZER** provides a clear, actionable plan, you will call the **CODER** to implement the new strategy and run the next experiment. 
4. **Repeat:** You will repeat this cycle, using the performance history to track progress and avoid repeating failed strategies.

<Available Tools>
You have access to the following tool:
1. **think_tool**: For reflection and strategic planning during research. Note that your thoughts will not be saved in your message history.

**CRITICAL: Use think_tool to plan your approach if you feel like you need to think deeper. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
1st Step: Look at the previous messages and take note of your previous goals, and the newest information provided to you
    - Be sure to carefully consider what the analyzer is saying regarding the game output
    - If needed, use think_tool to reflect on your current situation and plan your next steps
    - Note: The think_tool messages will only be visible to you for your current turn, so ensure to summarize your thoughts in META THOUGHTS

2nd Step: Output your current META THOUGHTS, and META GOAL at the top of your message
    - If you used think_tool, include a brief summary of your thoughts from the tool call in META THOUGHTS

3rd Step: Determine the sub-agent that you wish to consult, and prepare an OBJECTIVE message for them
    - If your performance history has not improved in the last three evolutions or stays at 0, consult the strategizer

Rules:
- You MUST choose one of: ANALYZER, STRATEGIZER, CODER.
</Instructions>

### Your Agents 
You have a team of specialists. You must delegate the correct task to the correct agent. 
- **{ANALYZER_NAME}: The Diagnostician.** 
- **When to Call:** Call this agent **first** after a game is played. 
- **Purpose:** Its job is to perform a **Root Cause Analysis**. It must connect the logic in `foo_player.py` to the behavior in the game logs and the scores in the results. It tells you **WHY** the player is failing. 
- **CRITICAL:** When you task the Analyzer, you must instruct it to find the **strategic flaw in the code**. Do not just ask for a summary of the results. Use the template below. 
- **{STRATEGIZER_NAME}: The Idea Generator.** 
- **When to Call:** Call this agent **after** the Analyzer has identified a clear strategic flaw. 
- **Purpose:** Its job is to propose a new, concrete strategy to fix the flaw. It provides the "what to do next." 
- **{CODER_NAME}: The Implementer.** 
- **When to Call:** Call this agent **after** the Strategist has provided a clear, actionable plan. 
- **Purpose:** Its job is to write the code for the new strategy and run the next experiment.

<Guidelines>
    - Make sure to be clear and concise in your message
    - Do not include vague messages to your agents, 
    - Always keep your GOAL in mind and try to achieve them
    - Only include one agent key (the output is parsed to determine which agent to send it to)
</Guidelines>

<Output Format>
    - META THOUGHTS: <insert here>
    - META GOAL: <insert here>
    - CHOSEN AGENT: {ANALYZER_NAME} / {STRATEGIZER_NAME} / {RESEARCHER_NAME} / {CODER_NAME} (choose one)
    - AGENT OBJECTIVE: <insert your objective message for the agent here>
</Output Format>

                """
            )
            
            msgs = state["meta_messages"][-MAX_MESSAGES_IN_AGENT:]
            tools = [think_tool]
            output = tool_calling_state_graph(self.meta_llm, "META",sys_msg, msgs, tools)

            #new_meta_message = HumanMessage(content=f"Temporary Meta Message ")
            
            # Place AI Message in the meta history
            meta_messages = state["meta_messages"] + [output["messages"][-1]]

            # Save the new_meta_message as a human message
            new_meta_message = HumanMessage(content=output["messages"][-1].content)

            return {"recent_meta_message": new_meta_message,"meta_messages": meta_messages}
        
        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")
            
            sys_msg = SystemMessage(
                content=f"""
{MULTI_AGENT_PROMPT} ANALYZER
                    
<Your Inputs>
    - The previous messages between the Coordinator agent and you
    - The most up to date performance history, with the scores and game results of the {FOO_TARGET_FILENAME} player across evolutions
    - The most recent foo_player.py file (note previous messages might be referring to an older version)
    - The adapters.py file which is used to interact with the Catantron API.
    - The most recent game_output.txt file which contains the output from run game command
    - The most recent game_results json file which contains the breakdown of the {FOO_TARGET_FILENAME} player vs. the opponent
        - Note: The game_results json file will not be included if the game failed to run due to a syntax error
    - Your OBJECTIVE: The most recent message includes the task that you are responding to... starts with {ANALYZER_NAME}
</Your Inputs>

<Your Role>
- You are the **Chief Diagnostician** for the team evolving the foo_player.py. - Your primary purpose is to form a hypothesis, grounded in the `foo_player.py` code, that explains *why* the player is winning or losing. - You must connect the logic in the code to the behavior in the logs and the performance in the results.
    - You are the Game ANALYZER Expert for Evolving the {FOO_TARGET_FILENAME} player
    - Assume adapters.py is stable and sufficient for API usage.
    - Do NOT browse the Catanatron core or other local files in improvement phase
    - You will be given the current foo_player.py, adapters.py, game_output.txt, game_results JSON, and performance history inline
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the analyzer, you are the forefront for the game output for the foo_player.py
    - You are aware of the nuances of the game output, and how to interpret the results
    - You are in charge of storing all the knowledge that you have learned
    - You can open any file from the performace history using the read_local_file tool
    - Ensure output from the game_output.txt matches the {FOO_TARGET_FILENAME} player
</Your Role>

<Your Task>
    1. **Start with the code.** Read the `foo_player.py` file first to understand its intended logic and strategy. 
    2. **Synthesize all inputs.** Digest your past inquiries, the performance history, the game output, the game results, and your OBJECTIVE to form a complete picture. 
    3. **Form a hypothesis.** Connect the code's strategy (or lack thereof) to the game's outcome. 
    4. **Respond to your OBJECTIVE** following your guidelines.
</Your Task>]

<Your Guidelines>
    - Prepare an organized, clear, and concise report with your answer to the most recent message
    - Do not make up information. If you do not know the answer, say you do not know and where you looked
    - Cite the sources that you used in your report at the bottom (so you know where to find it in the future)
    - Anytime when asked about the game output, log, or game_output.txt file, be sure to return debugging information
    - Ensure to include log messages like this in your response
            "Error: Syntax Error"
            "Unrecognized action type: UNKNOWN" - could be problem with action type
            "Defaulting to Random Action" - could be problem with action selection
            "Choose action with score: 0" - could be problem with action scoring 
   - **CRITICAL:** Your final report must include a section titled "**Strategic Flaw**" where you state, in one or two clear sentences, the fundamental weakness of the player's logic. 
   - End your response with 'Let me know if you need anything else.'
</Your Guidelines>

<Your Tools>              
    - think_tool: Reflect on your current situation and plan your next steps
        Input: String reflection -Your detailed reflection on research progress, findings, gaps, and next steps
        Output: String - Confirmation that reflection was recorded for decision-making
    - read_adapter: Read the current adapters.py file that provides all available functions to interact with the Catantron API.
    - read_local_file: Read the content of a file that is in the performance history
        Input: String rel_path - path of the file to read
        Output: String - content of the file
</Your Tools>

YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
Make sure to start your output with '{ANALYZER_NAME}' and end with 'END {ANALYZER_NAME}'.
Respond with No Commentary, just the Analysis.
                """
            )
            
            tools = [read_adapter, think_tool, read_local_file]

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            game_output_msg = HumanMessage(content=f"This is the current game_output.txt file\n\n{read_game_output_file()}")
            game_results_msg = HumanMessage(content=f"This is the current game_results json file\n\n{read_game_results_file()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")


            # Call the LLM with the provided tools and msgs
            #base_len = len(state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:])
            msgs = state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, game_output_msg, game_results_msg, current_foo_msg, adapter_msg, state["recent_meta_message"]]
            # msgs = state["meta_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, game_output_msg, game_results_msg, current_foo_msg, adapter_msg]
            output = tool_calling_state_graph(self.analyzer_llm, ANALYZER_NAME, sys_msg, msgs, tools)

            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_path = os.path.join(CreatorAgent.run_dir, f"analzyer_prompt_{ts}.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write("=== SYSTEM MESSAGE ===\n")
                    f.write(getattr(sys_msg, "pretty_repr", lambda: str(sys_msg))())
                    f.write("\n\n=== MESSAGES ===\n")
                    for i, m in enumerate(msgs, start=1):
                        f.write(f"\n--- Message {i} ---\n")
                        pretty = getattr(m, "pretty_repr", None)
                        f.write(pretty() if callable(pretty) else str(m))
                        f.write("\n")
            except Exception:
                # Logging should never crash the run
                pass
            
            # Add to Meta Messages
            response = HumanMessage(content=output["messages"][-1].content)
            meta_messages = state["meta_messages"] + [response]

            # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
            # Only summarize new messages
            #tool_call_summary = summarize_messages(output["messages"][base_len:])
            # analyzer_messages = state["analyzer_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            analyzer_messages = state["analyzer_messages"] + [AIMessage(content=response.content)]

            return {
                "recent_helper_response": response, 
                "tool_calling_messages": output["messages"], 
                "meta_messages": meta_messages, 
                "analyzer_messages": analyzer_messages,
            }
        
        def strategizer_node(state: CreatorGraphState):
            
            #print("In Strategizer Node")
            # Add custom tools for strategizer

            sys_msg = SystemMessage(
                content=f"""
{MULTI_AGENT_PROMPT} {STRATEGIZER_NAME}

<Your Inputs>
    - The previous messages between the Coordinator agent and you
    - The most up to date performance history, with the scores and game results of the {FOO_TARGET_FILENAME} player accross evolutions.
        - If a score is 0 for a Evolution and json_game_results_path is None, it means that the game failed to run due to a syntax error
        - Sometimes you might need to look at the most recent running {FOO_TARGET_FILENAME} player to see if the game ran, which will be a nonzero score for Evolution
    - The most recent foo_player.py file (note previous messages might be referring to an older version)
    - The adapters.py file which is used to interact with the Catantron API.
    - Your OBJECTIVE: The most recent message includes the task that you are responding to... starts with {STRATEGIZER_NAME}
</Your Inputs>

<Your Role>
    - You are the Strategy Expert for Evolving the {FOO_TARGET_FILENAME} player
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the strategizer, you are the forefront for improvement the foo_player.py
    - Propose strategies that leverage the available adapter functions. For example, if the adapter.py file has `get_state_representation` and `get_reward`, you could suggest reinforcement learning. If it has `get_all_possible_outcomes`, you could suggest a Monte Carlo Tree Search (MCTS) approach.
    - You are **Creative**, and are always looking for new strategies to implement
    - If you feel like the current strategy is not working, feel free to include it in your response
    - You are in charge of storing all the different attempts at strategies, and the results of each strategy
    - Avoid basic brute force approaches and be thoughtful about creating a foo_player with a clear strategy.
</Your Role>

<Your Task>
    1. Digest the current performance history, the current foo_player.py, the adapters.py file, the past messages, and your OBJECTIVE
    2. Use any additional tools required to get the information you need
    3. Respond to your OBJECTIVE message following your guidelines
</Your Task>

<Your Guidelines>
    - Prepare an organized, clear, and concise report with your answer to the most recent message
    - Do not make up information. If you do not know the answer, say you do not know
    - Cite any sources that you use in your report at the bottom
</Your Guidelines>

<Scenarios>
    Within the first 5 Evolutions, if The performance history shows that the player consistentaly does not compile (score stays at 0) or cannot get a score better than default (score stays at 2), Repond With
        Try the following code snippet to get the player to compile and get simple results:
        for action in playable_actions:
            "if action.action_type == ActionType.BUILD_SETTLEMENT:
                return action"

    If The performance history contains a previous version of {FOO_TARGET_FILENAME} that is more successful then the recent iterations, 
        Call read_older_foo_file tool to get the code of the previous {FOO_TARGET_FILENAME}
        Either return the entire contents of the file, or just your analysis of the differences

    If the performance history shows no signs of player improving over the last 3 successful evolutions (game ran successfully)
        Recommend that the player should try a new strategy to optimize the {FOO_TARGET_FILENAME} player (This means starting from scratch)
</Scenarios>
    
<Your Tools>
    - read_adapter: Read the current adapters.py to see what functions are available to interact with the Catantron API.
    - read_local_file: Read the content of a file that is in the performance history
        Input: String rel_path - path of the file to read
        Output: String - content of the file
    - read_game_results_file: Read the content of the game_results*.json file
        Input: Int num - the evolution number you want to read (default is -1 for most recent), 0 will return the default template
        Output: String - contents of the file (Includes Player Summary With Wins, Victory Points, Cities, Settles, Road, Army, and Game Summary with number of Ticks, Turns))
    - read_older_foo_file: Read the content of an older vesrion {FOO_TARGET_FILENAME} file
        Input: Int num - the evolution number you want to read (default is -1 for most recent), 0 will return the default template
        Output: String - contents of the python file for the older player as a string
    - web_search_tool_call: Perform a web search using the Tavily API.
        Input: String query - the search query
        Output: TavilySearchResults - the search results
    - think_tool: Reflect on your current situation and plan your next steps
        Input: String reflection - Your detailed reflection on strategy options, tradeoffs, and next steps
        Output: String - Confirmation that reflection was recorded for decision-making
</Your Tools>

YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
Make sure to start your output with '{STRATEGIZER_NAME}' and end with 'END {STRATEGIZER_NAME}'.
Respond with No Commentary, just the Strategy.

                """
            )

            tools = [read_local_file, read_game_results_file, read_older_foo_file, web_search_tool_call, read_adapter, think_tool]
            
            # Call the LLM with the provided tools
            #base_len = len(state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:])

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")

            msgs = state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, current_foo_msg, adapter_msg, state["recent_meta_message"]]
            # msgs = state["meta_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, current_foo_msg, adapter_msg]
            output = tool_calling_state_graph(self.strategizer_llm, STRATEGIZER_NAME, sys_msg, msgs, tools)

            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_path = os.path.join(CreatorAgent.run_dir, f"strategizer_prompt_{ts}.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write("=== SYSTEM MESSAGE ===\n")
                    f.write(getattr(sys_msg, "pretty_repr", lambda: str(sys_msg))())
                    f.write("\n\n=== MESSAGES ===\n")
                    for i, m in enumerate(msgs, start=1):
                        f.write(f"\n--- Message {i} ---\n")
                        pretty = getattr(m, "pretty_repr", None)
                        f.write(pretty() if callable(pretty) else str(m))
                        f.write("\n")
            except Exception:
                # Logging should never crash the run
                pass

            # Add to Meta Messages
            response = HumanMessage(content=output["messages"][-1].content)
            meta_messages = state["meta_messages"] + [response]

            # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
            # Only summarize new messages
            #tool_call_summary = summarize_messages(output["messages"][base_len:])
            # strategizer_messages = state["strategizer_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            strategizer_messages = state["strategizer_messages"] + [AIMessage(content=response.content)]

            return {
                "recent_helper_response": response, 
                "tool_calling_messages": output["messages"], 
                "meta_messages": meta_messages, 
                "strategizer_messages": strategizer_messages,
            }
    
        def researcher_node(state: CreatorGraphState):
            
            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content=f"""                     
{MULTI_AGENT_PROMPT} {RESEARCHER_NAME}

<Your Inputs>
    - The previous messages between the Coordinator agent and you
    - A list of all of the files in the catanatron directory that you have access to
    - Your OBJECTIVE: The most recent message includes the task that you are responding to... starts with {RESEARCHER_NAME}
</Your Inputs>

<Your Role>
    - You are the Research Expert for Evolving the {FOO_TARGET_FILENAME} player
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the researcher, you are the forefront for knowledge for the foo_player.py
    - You are aware of the nuances of the Catanatron game, and the Catanatron codebase
    - You are in charge of storing all the knowledge that you have learned
</Your Role>

<Your Task>
    1. Digest the catanatron directory, your past inquiries, and your current OBJEECTIVE
    2. Use any additional tools required to get the information you need
    3. Respond to your OBJECTIVE message following your guidelines
</Your Task>

<Your Guidelines>
    - Prepare an organized, clear, and concise report with your answer to the most recent message
    - For questions on syntax, ensure to provide relevant code that you found
    - Do not make up information. If you do not know the answer, say you do not know and where you looked
    - Cite the sources that you used in your report at the bottom, with a note on the information they included (so you know where to find it in the future)
        Ex. 1. catanatron_core/catanatron/models/enums.py - includes enums for Development Cards, NodeRef, EdgeRef, ActionPrompt, and ActionType
</Your Guidelines>

<Your Tools>
    - read_local_file: Read the content of a file that is in the catanatron files. (look at previous sources cited at the bottom of your messages for file information)
        Input: String rel_path - path of the file to read from catanatron files or {FOO_TARGET_FILENAME}
        Output: String - content of the file
    - web_search_tool_call: Perform a web search using the Tavily API.
        Input: String query - the search query
        Output: TavilySearchResults - the search results
    - think_tool: Reflect on your current situation and plan your next steps
        Input: String reflection - Your detailed reflection on research progress, findings, gaps, and next steps
        Output: String - Confirmation that reflection was recorded for decision-making
</Your Tools>

YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
Make sure to start your output with '{RESEARCHER_NAME}' and end with 'END {RESEARCHER_NAME}'.
Respond with No Commentary, just the Research.


                """
            )

            tools = [read_local_file, web_search_tool_call, read_adapter, think_tool]
            
            catanatron_files_msg = HumanMessage(content=f"This is the list of catanatron files\n\n{list_catanatron_files()}")
            # Call the LLM with the provided tools (Add 1 because no need to summarize catanatron files)
            #base_len = len(state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:]) + 1
            msgs = state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:] + [catanatron_files_msg, state["recent_meta_message"]]
            output = tool_calling_state_graph(self.researcher_llm, "RESEARCHER", sys_msg, msgs, tools)

            # Add to Meta Messages
            response = HumanMessage(content=output["messages"][-1].content)
            meta_messages = state["meta_messages"] + [response]

            # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
            # Only summarize new messages
            #tool_call_summary = summarize_messages(output["messages"][base_len:])
            researcher_messages = state["researcher_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]

            return {
                "recent_helper_response": response, 
                "tool_calling_messages": output["messages"], 
                "meta_messages": meta_messages, 
                "researcher_messages": researcher_messages,
            }

        def coder_node(state: CreatorGraphState):

            sys_msg = SystemMessage(
                content=f"""                    
{MULTI_AGENT_PROMPT} {CODER_NAME}

<Your Inputs>
    - The previous messages between the Coordinator agent and you
    - The most last {MAX_META_MESSAGES_GIVEN_TO_CODER} before the {FOO_TARGET_FILENAME} include the most recent META messages
    - Your OBJECTIVE: The most last META message that includes the task that you are responding to... starts with {CODER_NAME}
    - The most recent foo_player.py file (note previous messages might be referring to an older version)
    - The adapter.py file which you are to use to interact with the Catanatron API
</Your Inputs>

<Your Role>
    - You are the Coding Expert for Evolving the {FOO_TARGET_FILENAME} player
    - HARD REQUIREMENT: You MUST use adapters.py to interact with the API
      and call ONLY the adapter surface (e.g., forward, list_actions, value_probe). Never remove
      The imports from .adapters.
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the coder, you are the forefront for implementation for the foo_player.py based on strategy recommendations
       provided to you.
    - You are in charge of storing all the coding nuances that you have learned
</Your Role>

<Your Task>
    1. Digest your past inquiries, the meta messages, your current OBJEECTIVE, and the current {FOO_TARGET_FILENAME}
    2. Call the write_foo tool call to write the new code to the {FOO_TARGET_FILENAME} file
    3. Create a report with the changes you made to the code
</Your Task>

<Coding Guidelines>
    - Lint rule: The code MUST contain `from .adapters import`
       and MUST NOT contain `from catanatron` or `import catanatron` in foo_player.py.
       If you see those, rewrite to use the adapter.
    - Focus on making sure the code implements the solution in the most correct way possible
    - Make Sure to not add backslashes to comments, ONLY OUTPUT VALID PYTHON CODE
        WRONG:        print(\\'Choosing First Action on Default\\')
        CORRECT:      print('Choosing First Action on Default')
    - Give plenty of comments in the code to explain what you are doing, and what you have learned (along with syntax help)
    - Use print statement to usefully debug the output of the code
    - DO NOT MAKE UP VARIABLES OR FUNCTIONS RELATING TO THE GAME
    - Note: You will have multiple of iterations to evolve, so make sure the syntax is correct
    - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
    - Make sure to follow **python 3.11** syntax!!
    - Your code will go straight to the {FOO_TARGET_FILENAME} file, to be run in the game, so make sure to be aware of the syntax
</Coding Guidelines>

<Report Guidelines>
    - Return bullet points of the changes you made to the code
    - Make sure to report if you did any of the following
        - Created new functions
        - Added functions/enums from the game
        - Are not sure if the syntax is correct for specific lines of code
        - Added print statements to debug the code
        - Want information on imports, or the game
    - Include any comments that can be included in next OBJECTIVE to help you write better code 
</Report Guidelines>

Your Tools:
    - write_foo: Write the entire content of {FOO_TARGET_FILENAME}. Use this when you need to make significant changes or rewrite the file.
        Input: String new_text - python code that will be written to {FOO_TARGET_FILENAME}
    - replace_code_in_foo: Replace a specific block of code in {FOO_TARGET_FILENAME}. Use this for smaller, targeted changes.
        Input: String search - the exact code block to search for.
        Input: String replace - the new code block to replace the search block with.
    - think_tool: Reflect on your current situation and plan your next steps before writing or after errors
        Input: String reflection - Your detailed reflection on implementation approach, risks, and next steps
        Output: String - Confirmation that reflection was recorded for decision-making
</Your Tools>

Make sure to start your report with '{CODER_NAME}' and end with 'END {CODER_NAME}'.
                """
            )
           
            tools = [write_foo, replace_code_in_foo, think_tool]
            
            # Give Coder The Last Number of Meta Messages
            if len(state["meta_messages"]) > MAX_META_MESSAGES_GIVEN_TO_CODER:
                meta_msgs = state["meta_messages"][-MAX_META_MESSAGES_GIVEN_TO_CODER:]
            else:
                meta_msgs = state["meta_messages"]

            
            # Call the LLM with the provided tools
            current_foo_msg = HumanMessage(content=f"This is the old foo_player.py file\nNow It is your turn to update it with the new recommendations from META\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file that you must use to interact with the Catanatron API\n\n{read_adapter()}")

            #base_len = len(state["coder_messages"][-MAX_MESSAGES_IN_AGENT:])
            msgs = state["coder_messages"][-MAX_MESSAGES_IN_AGENT:] + meta_msgs + [current_foo_msg, adapter_msg]
            #msgs = state["meta_messages"][-MAX_META_MESSAGES_GIVEN_TO_CODER:] + [current_foo_msg, adapter_msg]
            output = tool_calling_state_graph(self.coder_llm, "CODER", sys_msg, msgs, tools)

            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_path = os.path.join(CreatorAgent.run_dir, f"coder_prompt_{ts}.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write("=== SYSTEM MESSAGE ===\n")
                    f.write(getattr(sys_msg, "pretty_repr", lambda: str(sys_msg))())
                    f.write("\n\n=== MESSAGES ===\n")
                    for i, m in enumerate(msgs, start=1):
                        f.write(f"\n--- Message {i} ---\n")
                        pretty = getattr(m, "pretty_repr", None)
                        f.write(pretty() if callable(pretty) else str(m))
                        f.write("\n")
            except Exception:
                # Logging should never crash the run
                pass

            # Add to Meta Messages
            response = HumanMessage(content=output["messages"][-1].content)
            meta_messages = state["meta_messages"] + [response]
            coder_messages = state["coder_messages"] + [AIMessage(content=response.content)]
            # meta_messages = state["meta_messages"]
            # coder_messages = state["coder_messages"] + [AIMessage(content=response)]
            


            #Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
            #Only summarize new messages
            #tool_call_summary = summarize_messages(output["messages"][base_len:])
            # coder_messages = state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            
            # Add to Coder Messages
            #coder_messages = state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            
            return {
                "recent_helper_response": response, 
                "tool_calling_messages": output["messages"], 
                "meta_messages": meta_messages, 
                "coder_messages": coder_messages,
            }
        
        def adapter_researcher_node(state: CreatorGraphState):
            sys_msg = SystemMessage(content=f"""
        {MULTI_AGENT_PROMPT} ADAPTER_RESEARCHER

        Goal: Implement the following methods in the CatanatronAdapter class in adapters.py:
        - new_game, clone, legal_actions, apply, is_terminal, winner, current_player, serialize_action
        - get_state_representation: A function that returns a numpy array representing the game state.
        - get_action_representation: A function that returns a numpy array representing an action.
        - get_reward: A function that returns a float representing the reward for the current state.
        - get_all_possible_outcomes: A function that returns a list of (next_state, probability) tuples for a given action.
        
        Your tools: read_local_file (scan repo), think_tool.
        Output: a concise plan + a copy-paste ready adapters.py (or the diff) that follows the Adapter contract.

        Keep it short, precise, and actionable.
        """)
            tools = [read_local_file, read_adapter, run_adapter_tests, think_tool]
            msgs = state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:] + [
                HumanMessage(content=f"Project files:\n\n{list_catanatron_files()}"),
                state["recent_meta_message"]
            ]
            output = tool_calling_state_graph(self.researcher_llm, "ADAPTER_RESEARCHER", sys_msg, msgs, tools)
            response = HumanMessage(content=output["messages"][-1].content)
            return {
                "recent_helper_response": response,
                "tool_calling_messages": output["messages"],
                "meta_messages": state["meta_messages"] + [response],
                "researcher_messages": state["researcher_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)],
            }

        def adapter_coder_node(state: CreatorGraphState):
            sys_msg = SystemMessage(content=f"""
        {MULTI_AGENT_PROMPT} ADAPTER_CODER

        Task: write or fix {ADAPTER_TARGET_FILENAME} per the plan/diagnostics.
        Use write_adapter / replace_in_adapter. If file missing, start from ADAPTER_TEMPLATE (provided below).

        Goal: Implement the following methods in the CatanatronAdapter class in adapters.py:
        - new_game, clone, legal_actions, apply, is_terminal, winner, current_player, serialize_action
        - get_state_representation: A function that returns a numpy array representing the game state.
        - get_action_representation: A function that returns a numpy array representing an action.
        - get_reward: A function that returns a float representing the reward for the current state.
        - get_all_possible_outcomes: A function that returns a list of (next_state, probability) tuples for a given action.
        
        Return a bullet summary of changes.

        ADAPTER_TEMPLATE BEGINS
        {ADAPTER_TEMPLATE}
        ADAPTER_TEMPLATE ENDS
        """)
            tools = [write_adapter, replace_in_adapter, read_adapter, think_tool]
            current_adapter = HumanMessage(content=f"Current {ADAPTER_TARGET_FILENAME}:\n\n{read_adapter()}")
            msgs = state["coder_messages"][-MAX_MESSAGES_IN_AGENT:] + [current_adapter, state["recent_meta_message"]]
            output = tool_calling_state_graph(self.coder_llm, "ADAPTER_CODER", sys_msg, msgs, tools)
            response = HumanMessage(content=output["messages"][-1].content)
            return {
                "recent_helper_response": response,
                "tool_calling_messages": output["messages"],
                "meta_messages": state["meta_messages"] + [response],
                "coder_messages": state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)],
            }

        def adapter_analyzer_node(state: CreatorGraphState):
            sys_msg = SystemMessage(content=f"""
        {MULTI_AGENT_PROMPT} ADAPTER_ANALYZER

        Inputs:
        - The latest adapter test report (JSON in previous message)
        - The current adapters.py

        Your job:
        - Parse the test report, list failures with precise fixes (line numbers if possible).
        - If all tests passed: recommend switching phase to IMPROVEMENT.
        """)
            tools = [run_adapter_tests, read_local_file, think_tool]
            msgs = state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:] + [
                HumanMessage(content=f"Latest adapter test report:\n\n{state['adapter_test_report'].content}"),
                HumanMessage(content=f"Current {ADAPTER_TARGET_FILENAME}:\n\n{read_adapter()}"),
                state["recent_meta_message"]
            ]
            output = tool_calling_state_graph(self.analyzer_llm, "ADAPTER_ANALYZER", sys_msg, msgs, tools)
            response = HumanMessage(content=output["messages"][-1].content)

            # Phase switch heuristic: if tests now pass, flip to improvement
            try:
                rpt = json.loads(run_adapter_tests())
                all_good = len(rpt.get("failed", [])) == 0
            except Exception:
                all_good = False

            new_phase = "improvement" if all_good else state.get("phase", "discovery")

            return {
                "recent_helper_response": response,
                "tool_calling_messages": output["messages"],
                "meta_messages": state["meta_messages"] + [response],
                "analyzer_messages": state["analyzer_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)],
                "phase": new_phase,
            }

 
        def meta_choice(state: CreatorGraphState):
            """
            Conditional edge for Meta
            """
            print("In Conditional Edge Meta")
        
            # Create current_messages directory if needed
            os.makedirs(os.path.join(CreatorAgent.run_dir, "messages_current"), exist_ok=True)

            # Save all messages to log files
            lists = ["meta_messages", "analyzer_messages", "strategizer_messages", "researcher_messages", "coder_messages"]
            for msg_list in lists:
                log_path = os.path.join(CreatorAgent.run_dir, "messages_current", f"llm_log_{msg_list}.txt")
                with open(log_path, "w") as log_file:
                    for m in state[msg_list]:
                        log_file.write(m.pretty_repr() + "\n")

            # End evolution if we exceed max evolutions
            if (CreatorAgent.current_evolution > CREATOR_NUM_EVOLUTIONS):
                print(f"Reached Max Evolutions of {CREATOR_NUM_EVOLUTIONS}, going to END")
                return END

            meta_message = state["meta_messages"][-1].content

            # First, try to find the chosen agent using the specific format
            match = re.search(r"CHOSEN AGENT:\s*(\w+)", meta_message)
            if match:
                agent_name = match.group(1)
                if agent_name == RESEARCHER_NAME:
                    print(f"Meta Message: RESEARCHER blocked in improvement; routing to {STRATEGIZER_NAME}")
                    return STRATEGIZER_NAME
                if agent_name in AGENT_KEYS:
                    print(f"Meta Message: Found agent {agent_name} via specific format - going to {agent_name}")
                    return agent_name

            # If not found, fall back to just searching the test
            for key in AGENT_KEYS:
                if key in meta_message:
                    if key == RESEARCHER_NAME:
                        print(f"Meta Message: RESEARCHER blocked in improvement; routing to {STRATEGIZER_NAME}")
                        return STRATEGIZER_NAME
                if key in meta_message:
                    print(f"Meta Message: {key} - going to {key}")
                    return key
                
                # Default case if neither string is found
            print(f"Warning: Could not determine desired agent in recent meta message. Defaulting to {ANALYZER_NAME}")
            return ADAPTER_ANALYZER_NAME if state.get("phase") == "discovery" else ANALYZER_NAME


        def post_run_choice(state: dict) -> str:
            # After "run_player": route to the right analyzer based on phase
            return "ADAPTER_ANALYZER" if state.get("phase") == "discovery" else ANALYZER_NAME

                
        def construct_graph():
            graph = StateGraph(CreatorGraphState)
            graph.add_node("init", init_node)

            # Existing nodes
            graph.add_node(ANALYZER_NAME, analyzer_node)
            graph.add_node(STRATEGIZER_NAME, strategizer_node)
            graph.add_node(RESEARCHER_NAME, researcher_node)
            graph.add_node(CODER_NAME, coder_node)
            graph.add_node("run_player", run_player_node)
            graph.add_node("meta", meta_node)

            # NEW adapter nodes
            graph.add_node(ADAPTER_RESEARCHER_NAME, adapter_researcher_node)
            graph.add_node(ADAPTER_CODER_NAME, adapter_coder_node)
            graph.add_node(ADAPTER_ANALYZER_NAME, adapter_analyzer_node)

            graph.add_edge(START, "init")

            # From init → run
            graph.add_edge("init", "run_player")

            # After run, route based on phase to the proper analyzer
            graph.add_conditional_edges(
                "run_player",
                post_run_choice,
                {ADAPTER_ANALYZER_NAME: ADAPTER_ANALYZER_NAME, ANALYZER_NAME: ANALYZER_NAME}
            )

            # Meta routing
            graph.add_conditional_edges(
                "meta", 
                meta_choice,
                {
                    ANALYZER_NAME: ANALYZER_NAME,
                    STRATEGIZER_NAME: STRATEGIZER_NAME,
                    RESEARCHER_NAME: RESEARCHER_NAME,
                    CODER_NAME: CODER_NAME,
                    ADAPTER_ANALYZER_NAME: ADAPTER_ANALYZER_NAME,
                    ADAPTER_RESEARCHER_NAME: ADAPTER_RESEARCHER_NAME,
                    ADAPTER_CODER_NAME: ADAPTER_CODER_NAME,
                    END: END
                }
            )

            # Improvement loop
            graph.add_edge(ANALYZER_NAME, "meta")
            graph.add_edge(STRATEGIZER_NAME, "meta")
            graph.add_edge(RESEARCHER_NAME, "meta")
            graph.add_edge(CODER_NAME, "run_player")

            # Discovery loop
            graph.add_edge(ADAPTER_ANALYZER_NAME, "meta")
            graph.add_edge(ADAPTER_RESEARCHER_NAME, "meta")
            graph.add_edge(ADAPTER_CODER_NAME, "run_player")

            return graph.compile()

    
        return construct_graph()

    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        
        try:

            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log.txt")

            def append_log_file(content: str):
                with open(log_path, "a") as log_file:
                    log_file.write(content + "\n")


            for step in self.react_graph.stream({}, self.config, stream_mode="updates"):
                #print(step)
                #log_file.write(f"Step: {step.}\n")
                for node, update in step.items():
                    
                    print(f"In Node: {node}")
                    append_log_file(f"In Node: {node}")
                    # Simplified Messages code
                    key_types = ["recent_meta_message", "recent_helper_response", "game_results"]
                    for key in key_types:
                        if key in update:
                            msg = update[key]
                            #msg.pretty_print()
                            append_log_file(msg.pretty_repr() + "\n")

            print("✅  graph finished")

            # Copy Result File to the new directory
            dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")

            shutil.copy2(                           
                (FOO_TARGET_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / ("final" + dt + FOO_TARGET_FILENAME))
            )

            if ADAPTER_TARGET_FILE.exists():
                shutil.copy2(
                    ADAPTER_TARGET_FILE.resolve(),
                    (Path(CreatorAgent.run_dir) / ("final" + dt + ADAPTER_TARGET_FILENAME))
                )

        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
        return None


###################################################################################################
#  TOOL CALLS AND UTILS
###################################################################################################
def list_catanatron_files(_: str = "") -> str:
    """Return all files beneath BASE_DIR, one per line."""
    return "\n".join(
        str(p.relative_to(LOCAL_CATANATRON_BASE_DIR))
        for p in LOCAL_CATANATRON_BASE_DIR.glob("**/*")
        if p.is_file() and p.suffix in {".py", ".txt", ".md"}
    )

def read_local_file(rel_path: str) -> str:
    """
    Return the text content of rel_path if it's inside BASE_DIR.
    Args:
        rel_path: Relative path to the file to read.
    """
    # Path Requested is from Agent File
    if rel_path == FOO_TARGET_FILENAME:
        return read_foo()
    
    if rel_path == ADAPTER_TARGET_FILENAME:
        return read_adapter()
    
    # Path is from Catanatron base directory
    if rel_path.startswith("catanatron/"):
        candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path.replace("catanatron/", "")).resolve()
        if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
            raise ValueError("Access denied or not a file")
        if candidate.stat().st_size > 64_000:
            raise ValueError("File too large")
        return candidate.read_text(encoding="utf-8", errors="ignore")
    
    # Handle paths relative to run_dir (used in performance history)
    # This includes both paths starting with "runs/" and paths that don't start with "/"
    run_path = Path(CreatorAgent.run_dir) / rel_path
    if run_path.exists() and run_path.is_file():
        if run_path.stat().st_size > 64_000:
            raise ValueError("File too large")
        return run_path.read_text(encoding="utf-8", errors="ignore")
    
    # Check if path is relative to Catanatron directory
    candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path).resolve()
    if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
        raise ValueError(f"Access denied or file not found: {rel_path}")
    if candidate.stat().st_size > 64_000:
        raise ValueError("File too large")
    return candidate.read_text(encoding="utf-8", errors="ignore")

def read_foo(_: str = "") -> str:
    """
    Return the UTF-8 content of Agent File (≤64 kB).
    """
    if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")  # pathlib API :contentReference[oaicite:2]{index=2}

def write_foo(new_text: str) -> str:
    """
    Overwrite Agent File with new_text (UTF-8).
    """
    if len(new_text.encode()) > FOO_MAX_BYTES:
        raise ValueError("Refusing to write >64 kB")
    FOO_TARGET_FILE.write_text(new_text, encoding="utf-8")                 # pathlib write_text :contentReference[oaicite:3]{index=3}

    return f"{FOO_TARGET_FILENAME} updated successfully"

def replace_code_in_foo(search: str, replace: str) -> str:
    """
    Replace a block of code in the Agent File.
    """
    # First, read the file
    try:
        content = read_foo()
    except Exception as e:
        return f"Error reading file: {e}"

    # Then, perform the replacement
    new_content = content.replace(search, replace)

    if new_content == content:
        return "Search string not found in file. No changes made."

    # Finally, write the file back
    try:
        write_foo(new_content)
        return f"Successfully replaced code in {FOO_TARGET_FILENAME}"
    except Exception as e:
        return f"Error writing file: {e}"

def run_testfoo(short_game: bool = False) -> str:
    """
    Run one Catanatron match and update performance history.
    The catanatron CLI writes JSON results to the --results-path we supply.
    """
    run_id = datetime.now().strftime(f"game_%Y%m%d_%H%M%S_{'vg' if short_game else 'fg'}")
    game_run_dir = Path(CreatorAgent.run_dir) / run_id
    game_run_dir.mkdir(exist_ok=True)

    # Explicit results file path (named by run_id)
    results_json_path = game_run_dir / f"{run_id}.json"
    print(f"[DEBUG] Writing results to: {results_json_path}")

    # Build command (strip any prior --results-path)
    base_cmd = re.sub(r"--results-path=\S+", "", FOO_RUN_COMMAND).strip()
    dynamic_command = f'{base_cmd} --results-path="{results_json_path}"'

    # Snapshot current foo_player.py
    cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
    shutil.copy2(FOO_TARGET_FILE.resolve(), cur_foo_path)

    adapter_src = ADAPTER_TARGET_FILE
    if adapter_src.exists():
        shutil.copy2(adapter_src.resolve(), game_run_dir / ADAPTER_TARGET_FILENAME)

    print("RUNNING GAME")
    
    # Play the game through the API with a timeout
    MAX_CHARS = 20_000
    try:
        result = subprocess.run(
            shlex.split(dynamic_command),
            capture_output=True,
            text=True,
            timeout=30 if short_game else 14400,
            check=False,
        )
        stdout_limited = result.stdout[-MAX_CHARS:]
        stderr_limited = result.stderr[-MAX_CHARS:]
        combined_output = (stdout_limited + stderr_limited).strip()
    except subprocess.TimeoutExpired as e:
        so = (e.stdout or "")
        se = (e.stderr or "")
        stdout_limited = so[-MAX_CHARS:]
        stderr_limited = se[-MAX_CHARS:]
        combined_output = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()

    # Save raw CLI output
    output_file_path = game_run_dir / "game_output.txt"
    output_file_path.write_text(combined_output, encoding="utf-8")

    # Read JSON results (single attempt—CLI should have written it)
    try:
        json_content = json.loads(results_json_path.read_text(encoding="utf-8")) if results_json_path.exists() else {}
    except json.JSONDecodeError:
        json_content = {"error": "Failed to parse JSON file"}

    # Extract stats
    wins = 0
    avg_score = 0
    avg_turns = 0
    try:
        if "Player Summary" in json_content:
            for player_key, stats in json_content["Player Summary"].items():
                if "fooplayer" in player_key.lower():
                    wins = stats.get("WINS", 0)
                    avg_score = stats.get("AVG VP", 0)
                    break
        if "Game Summary" in json_content:
            avg_turns = json_content["Game Summary"].get("AVG TURNS", 0)
    except Exception as e:
        print(f"[DEBUG] Stat extraction error: {e}")

    # Update performance history only for full (non-short) runs
    if not short_game:
        perf_path = Path(CreatorAgent.run_dir) / "performance_history.json"
        try:
            performance_history = json.loads(perf_path.read_text(encoding="utf-8"))
        except Exception:
            performance_history = {}

        evolution_key = CreatorAgent.current_evolution
        CreatorAgent.current_evolution += 1

        performance_history[f"Evolution {evolution_key}"] = {
            "wins": wins,
            "avg_score": avg_score,
            "avg_turns": avg_turns,
            "full_game_log_path": str(output_file_path.relative_to(CreatorAgent.run_dir)),
            "json_game_results_path": str(results_json_path.relative_to(CreatorAgent.run_dir)) if results_json_path.exists() else "None",
            "cur_foo_player_path": str(cur_foo_path.relative_to(CreatorAgent.run_dir)),
            "cli_run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        perf_path.write_text(json.dumps(performance_history, indent=2), encoding="utf-8")

    return json.dumps(json_content, indent=2) if json_content else combined_output

def web_search_tool_call(query: str) -> str:
    """Perform a web search using the Tavily API.

    Args:
        query: The search query string.

    Returns:
        The search result as a string.
    """
    # Simulate a web search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs

def read_full_performance_history(_: str = "") -> str:
    """Return the content of performance_history.json as a string (≤16 kB)."""
    performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"

    if not performance_history_path.exists():
        return "Performance history file does not exist."
    
    if performance_history_path.stat().st_size > 64_000:
        return "Performance history file is too large (>16 KB). Consider truncating or summarizing it."
    
    with open(performance_history_path, 'r') as f:
        performance_history = json.load(f)
        return json.dumps(performance_history, indent=2)
    
def read_game_output_file(num: int = -1) -> str:
    """Return the contents of the *.txt* game-log for the chosen num Evolution."""
    entry, err = _get_evolution_entry(num)
    if err:
        return err

    path = entry.get("full_game_log_path")
    if not path or path == "None":
        return f"No game-output file recorded for Evolution {num}."

    try:
        return read_local_file(path)
    except Exception as exc:            # pragma: no cover
        return f"Error reading '{path}': {exc}"
    
def read_game_results_file(num: int = -1) -> str:
    """Return the contents of the *.json* game-results file for the chosen num Evolution."""
    entry, err = _get_evolution_entry(num)
    if err:
        return err

    path = entry.get("json_game_results_path")
    if not path or path == "None":
        return f"No game-results file recorded for Evolution {num}."

    try:
        return read_local_file(path)
    except Exception as exc:            # pragma: no cover
        return f"Error reading '{path}': {exc}"
    
def read_older_foo_file(num: int = -1) -> str:
    """
    Return the contents of the *foo_player.py* file saved for the
    chosen num evolution
    """
    entry, err = _get_evolution_entry(num)
    if err:
        return err

    path = entry.get("cur_foo_player_path")
    if not path or path == "None":
        return f"No foo-player file recorded for Evolution {num}."

    try:
        return read_local_file(path)
    except Exception as exc:          # pragma: no cover
        return f"Error reading '{path}': {exc}"

# Think tool from langchain-ai:open_deep_research
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
# Helper to parse performance history
def _get_evolution_entry(num: int) -> Tuple[Dict[str, Any], str]:
    """
    Return (entry, "") on success or (None, error_msg) on failure.
    """
    perf_str = read_full_performance_history()
    try:
        perf = json.loads(perf_str)
    except json.JSONDecodeError:
        return None, f"Could not parse performance history JSON:\n{perf_str}"

    if not perf:
        return None, "Performance history is empty."

    # Choose evolution index
    if num == -1:
        # latest (largest) evolution number
        nums = [int(k.split()[1]) for k in perf if k.startswith("Evolution ")]
        if not nums:
            return None, "No Evolution entries found."
        num = max(nums)

    key = f"Evolution {num}"
    if key not in perf:
        return None, f"{key} not found in performance history."

    return perf[key], ""


# ---------- Adapter artifact tools ----------


def read_adapter(_: str = "") -> str:
    """Return the UTF-8 content of adapters.py (≤64 kB), or a sentinel string if missing."""
    if not ADAPTER_TARGET_FILE.exists():
        return "(adapters.py not found)"
    if ADAPTER_TARGET_FILE.stat().st_size > 64_000:
        raise ValueError("adapters.py too large")
    return ADAPTER_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")


def write_adapter(new_text: str) -> str:
    """Overwrite adapters.py with `new_text` (UTF-8). Enforces a 64 kB limit."""
    if len(new_text.encode()) > 64_000:
        raise ValueError("Refusing to write >64 kB")
    ADAPTER_TARGET_FILE.write_text(new_text, encoding="utf-8")
    return f"{ADAPTER_TARGET_FILENAME} updated successfully"


def replace_in_adapter(search: str, replace: str) -> str:
    """Find/replace exact `search` substring in adapters.py; writes back if changed."""
    if not ADAPTER_TARGET_FILE.exists():
        return f"{ADAPTER_TARGET_FILENAME} not found"
    content = read_adapter()
    new_content = content.replace(search, replace)
    if new_content == content:
        return "Search string not found in adapters.py. No changes made."
    ADAPTER_TARGET_FILE.write_text(new_content, encoding="utf-8")
    return f"Successfully replaced code in {ADAPTER_TARGET_FILENAME}"

def run_adapter_tests(_: str = "") -> str:
    """
    Load adapters.py dynamically, locate *Adapter class, instantiate, and run property tests.
    Returns a JSON string {passed:[], failed:[{name,trace}], meta:{…}}.
    """
    import importlib.util, json, traceback, random

    out = {"passed": [], "failed": [], "meta": {}}

    if not ADAPTER_TARGET_FILE.exists():
        out["failed"].append({"name": "file_exists", "trace": "adapters.py not found"})
        return json.dumps(out, indent=2)

    try:
        spec = importlib.util.spec_from_file_location("adapters", str(ADAPTER_TARGET_FILE))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    except Exception:
        out["failed"].append({"name": "import_adapters", "trace": traceback.format_exc()})
        return json.dumps(out, indent=2)

    # find a candidate class
    adapter_cls = None
    for name in dir(mod):
        if name.endswith("Adapter"):
            adapter_cls = getattr(mod, name)
            if callable(adapter_cls):
                break
    if adapter_cls is None:
        out["failed"].append({"name": "find_adapter_class", "trace": "No *Adapter class in adapters.py"})
        return json.dumps(out, indent=2)

    try:
        adapter = adapter_cls()
    except Exception:
        out["failed"].append({"name": "instantiate_adapter", "trace": traceback.format_exc()})
        return json.dumps(out, indent=2)

    # property tests
    def record(test_name, fn):
        try:
            fn()
            out["passed"].append(test_name)
        except Exception:
            out["failed"].append({"name": test_name, "trace": traceback.format_exc()})

    def test_clone_purity():
        g = adapter.new_game()
        g2 = adapter.clone(g)
        assert id(g) != id(g2)
        assert list(adapter.legal_actions(g)) == list(adapter.legal_actions(g2))

    def test_step_is_pure():
        g = adapter.new_game()
        acts = list(adapter.legal_actions(g))
        if not acts:
            return
        a = random.choice(acts)
        g2 = adapter.apply(adapter.clone(g), a)
        # original unchanged in spirit
        _ = list(adapter.legal_actions(g))
        # apply again shouldn't crash
        _ = adapter.apply(adapter.clone(g2), a)

    def test_terminal_contract():
        g = adapter.new_game()
        # bounded random playout to ensure no crashes and (ideally) eventual termination
        for _ in range(800):
            if adapter.is_terminal(g):
                assert adapter.winner(g) is not None
                return
            acts = list(adapter.legal_actions(g))
            if not acts:
                break
            g = adapter.apply(g, random.choice(acts))
        # Not strictly failing if not terminal; only ensure no exceptions

    for name, fn in [
        ("test_clone_purity", test_clone_purity),
        ("test_step_is_pure", test_step_is_pure),
        ("test_terminal_contract", test_terminal_contract),
    ]:
        record(name, fn)

    out["meta"]["adapter_class"] = adapter_cls.__name__
    
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpt_path = Path(CreatorAgent.run_dir) / f"discovery_report_{ts}.json"
        rpt_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        out["meta"]["report_path"] = str(rpt_path.relative_to(CreatorAgent.run_dir))
    except Exception:
        pass

    return json.dumps(out, indent=2)
# ---------- end adapter tools ----------
