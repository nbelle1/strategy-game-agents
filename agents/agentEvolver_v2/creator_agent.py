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
import traceback

from agents.agentEvolver_v2.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    DEFAULT_ANALYZE_MSG,
    META_SYSTEM_PROMPT,
    MULTI_AGENT_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    STRATEGIZER_SYSTEM_PROMPT,
    DISCOVERY_MULTI_AGENT_PROMPT,
    DISCOVERY_META_SYSTEM_PROMPT,
    DISCOVERY_ANALYZER_SYSTEM_PROMPT,
    DISCOVERY_STRATEGIZER_SYSTEM_PROMPT,
    DISCOVERY_CODER_SYSTEM_PROMPT,
    DISCOVERY_RESEARCHER_SYSTEM_PROMPT,

)
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
PRINT_DEBUG = True
PRINT_LLM = "LOG" # "NONE", "LOG", "LOG_FULL"

# Starting phase for the run: "improvement" (default) or "discovery"
START_PHASE = "discovery"  # "discovery" | "improvement"
CURRENT_PHASE = START_PHASE

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
RESEARCHER_LLM_MODEL = "mistral-large-latest"

# Strategizer LLM
STRATEGIZER_LLM_BACKEND = "mistral"
STRATEGIZER_LLM_MODEL = "mistral-large-latest"

# Meta LLM
META_LLM_BACKEND = "mistral"
META_LLM_MODEL = "mistral-large-latest"
# META_LLM_BACKEND = "openai"
# META_LLM_MODEL = "gpt-5-mini"

FOO_MAX_BYTES   = 64_000      # context-friendly cap
CREATOR_LANGRAPH_RECURSION_LIMIT = 200  # max depth of graph recursion
CREATOR_NUM_EVOLUTIONS = 20

MAX_MESSAGES_TOOL_CALLING = 4
MAX_META_MESSAGES_GIVEN_TO_CODER = 6
MAX_MESSAGES_IN_AGENT = 20

# Catanatron
FOO_RUN_COMMAND = "catanatron-play --players=AB,AE2 --num=30 --config-map=MINI --config-vps-to-win=10"
# catanatron-play --players=AB,BP --num=100 --config-vps-to-win=10

# Adapter
ADAPTER_TARGET_FILENAME = "adapters.py"
ADAPTER_TARGET_FILE = Path(__file__).parent / ADAPTER_TARGET_FILENAME
ADAPTER_TEMPLATE_FILENAME = "__TEMPLATE__adapters.py"
ADAPTER_TEMPLATE_FILE = Path(__file__).parent / ADAPTER_TEMPLATE_FILENAME

# CONSTANTS
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path

ANALYZER_NAME = "ANALYZER"
STRATEGIZER_NAME = "STRATEGIZER"
RESEARCHER_NAME = "RESEARCHER"
CODER_NAME = "CODER"
AGENT_KEYS = [ANALYZER_NAME, STRATEGIZER_NAME, RESEARCHER_NAME, CODER_NAME]

# VARIABLES

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
            f.write(f"PRINT_DEBUG = {PRINT_DEBUG}\n")
            f.write(f"PRINT_LLM = {PRINT_LLM}\n")
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
            os.makedirs(os.path.join(CreatorAgent.run_dir, "log"), exist_ok=True)

        #Copy the Blank FooPlayer to the run directory
        shutil.copy2(                           # ↩ copy with metadata
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),  # ../foo_player.py
            FOO_TARGET_FILE.resolve()          # ./foo_player.py
        )

        self.config = {
            "recursion_limit": CREATOR_LANGRAPH_RECURSION_LIMIT, # set recursion limit for graph
        }

        self.log_config_settings()

    def _run_phase(self, phase_name: str):
        """
        Compiles and runs the graph for a specific phase.
        """
        global CURRENT_PHASE
        CURRENT_PHASE = phase_name
        self.debug_log(f"PHASE START: Compiling and running the {phase_name.upper()} phase.")

        if phase_name == "discovery":
            try:
                src = ADAPTER_TEMPLATE_FILE.resolve()
                dst = ADAPTER_TARGET_FILE.resolve()
                shutil.copy2(src, dst)
                self.debug_log(f"Initialized {ADAPTER_TARGET_FILENAME} from template: {src.name}")
            except Exception as e:
                self.debug_log(f"Failed to initialize {ADAPTER_TARGET_FILENAME} from template: {e}")

        if phase_name == "discovery":
            graph = self.create_discovery_graph()
        elif phase_name == "improvement":
            graph = self.create_improvement_graph()
        else:
            self.debug_log(f"Error: Unknown phase '{phase_name}'")
            return

        # Each phase starts with a fresh state.
        for step in graph.stream({}, self.config, stream_mode="updates"):
            pass  # The work is done in the nodes

        # Copy adapters.py into the run directory at the end of the discovery phase
        if phase_name == "discovery":
            try:
                if ADAPTER_TARGET_FILE.exists():
                    out_path = Path(CreatorAgent.run_dir) / "adapters_after_discovery.py"
                    shutil.copy2(ADAPTER_TARGET_FILE.resolve(), out_path)
                    rel = Path(out_path).relative_to(Path(CreatorAgent.run_dir))
                    self.debug_log(f"Copied adapters.py to run dir after discovery: {rel}")
                else:
                    self.debug_log("adapters.py not found at end of discovery; nothing to copy.")
            except Exception as e:
                self.debug_log(f"Failed to copy adapters.py at end of discovery: {e}")

        self.debug_log(f"✅ PHASE COMPLETE: {phase_name.upper()} phase finished.")

    def debug_log(self, message: str):
        """
        Logs debug messages to both console and a debug log file.
        Use instead of print statements
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        if PRINT_DEBUG:
            print(log_entry)

        log_path = os.path.join(CreatorAgent.run_dir, "log", "debug_log.txt")
        with open(log_path, "a") as f:
            f.write(log_entry + "\n")

    def agent_log_input(self, agent_name: str, messages: list[AnyMessage]):
        """
        Logs input messages for a specific agent to their own log file
        """
        log_dir = os.path.join(CreatorAgent.run_dir, "log", agent_name)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{agent_name}_e{CreatorAgent.current_evolution}_{timestamp}.txt")

        with open(log_path, "w") as f:
            f.write(f"--- Input for {agent_name} at {timestamp} ---\n")
            for message in messages:
                f.write(message.pretty_repr() + "\n")
            f.write("\n")

        # Print only the relative path from run_dir
        relative_path = Path(log_path).relative_to(Path(CreatorAgent.run_dir))
        self.debug_log(f"{agent_name} logged to {relative_path}")

        return log_path

    def agent_log_output(self, agent_name: str, messages: list[AnyMessage], agent_log_path: str):
        """
        Logs output messages for a specific agent to their own log file, as well
        as to the full and short LLM logs if requested. 
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(agent_log_path, "a") as f:
            f.write(f"--- Output from {agent_name} at {timestamp} ---\n")
            for message in messages:
                f.write(message.pretty_repr() + "\n")
            f.write("\n")

        # Log to llm_log_full.txt
        full_log_path = os.path.join(CreatorAgent.run_dir, "log", "llm_log_full.txt")
        with open(full_log_path, "a") as f:
            f.write(f"--- Output from {agent_name} at {timestamp} ---\n")
            for message in messages:
                f.write(message.pretty_repr() + "\n")
            f.write("\n")

        # Log to llm_log.txt
        log_path = os.path.join(CreatorAgent.run_dir, "log", "llm_log.txt")
        with open(log_path, "a") as f:
            f.write(f"--- Output from {agent_name} at {timestamp} ---\n")
            if messages:
                f.write(messages[-1].pretty_repr() + "\n")
            f.write("\n")

        if PRINT_LLM == "LOG_FULL":
            print(f"--- Output from {agent_name} at {timestamp} ---")
            for message in messages:
                print(message.pretty_repr())
        elif PRINT_LLM == "LOG":
            print(f"--- Output from {agent_name} at {timestamp} ---")
            if messages:
                print(messages[-1].pretty_repr())

    def _tool_calling_state_graph(self, llm, sys_msg: SystemMessage, msgs: list[AnyMessage], tools):
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
        last_event = None
        for event in react_graph.stream({"messages": msgs}, stream_mode="values"):
            last_event = event

        return last_event

    def _init_node(self, state: CreatorGraphState):
        """
        Initialize the state of the graph
        """
        self.debug_log("INIT NODE")

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
        }

    # Used to test foo_player.py in IMPROVEMENT phase
    def _run_player_node(self, state: CreatorGraphState):
        """
        Runs Catanatron with the current Code
        """
        self.debug_log("RUN PLAYER NODE")

        # Generate a test results (later will be running the game)
        game_results = self._run_testfoo(short_game=False)
        game_msg = HumanMessage(content=f"GAME RESULTS:\n\n{game_results}")

        meta_messages = state["meta_messages"] + [game_msg]

        # Create a dummy meta message to automatically generate a summary of the last run_pla
        defualt_analyze_msg = HumanMessage(content=DEFAULT_ANALYZE_MSG.format(
            FOO_TARGET_FILENAME=FOO_TARGET_FILENAME
        ))
        # Clear all past messages
        return {
            "game_results": game_msg,
            "recent_meta_message": defualt_analyze_msg,
            "meta_messages": meta_messages,
        }

    def _validate_adapter_node(self, state: CreatorGraphState):
        """
        Validates adapters.py with a two-stage check:
        1. Syntax Check: Compiles the code to ensure it's valid Python.
        2. Runtime Check: Executes a test harness that imports and calls the functions.
        """
        self.debug_log("VALIDATE ADAPTER NODE")
        
        if not ADAPTER_TARGET_FILE.exists():
            validation_msg = HumanMessage(content="VALIDATION RESULT: adapters.py does not exist. Please create it.")
            # Only update the specific state key, not the conversational history
            return {"recent_meta_message": validation_msg}

        # --- Stage 1: Syntax Check ---
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(ADAPTER_TARGET_FILE)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            if result.returncode != 0:
                error_details = (result.stdout + result.stderr).strip()
                validation_output = f"VALIDATION FAILED (Syntax Error):\n\n{error_details}"
                self.debug_log(f"Adapter validation failed at syntax stage: {error_details}")
                validation_msg = HumanMessage(content=validation_output)
                # Only update the specific state key
                return {"recent_meta_message": validation_msg}
        except Exception as e:
            validation_output = f"VALIDATION FAILED (Syntax Check Crashed):\n\n{e}"
            self.debug_log(f"Syntax check subprocess failed: {e}")
            validation_msg = HumanMessage(content=validation_output)
            # Only update the specific state key
            return {"recent_meta_message": validation_msg}
        
        self.debug_log("Syntax check passed. Proceeding to runtime check.")

        # --- Stage 2: Runtime Check ---
        try:
            test_harness_path = Path(__file__).parent / "run_adapter_test.py"
            runtime_result = subprocess.run(
                [sys.executable, str(test_harness_path)],
                capture_output=True, text=True, check=False, timeout=60,
            )
            
            if runtime_result.returncode == 0:
                validation_output = "SUCCESS: adapters.py passed both syntax and runtime checks."
                self.debug_log(validation_output)
            else:
                error_details = (runtime_result.stdout + runtime_result.stderr).strip()
                validation_output = f"VALIDATION FAILED (Runtime Error):\n\n{error_details}"
                self.debug_log(f"Adapter validation failed at runtime stage.")
        
        except Exception as e:
            validation_output = f"VALIDATION FAILED (Runtime Check Crashed):\n\n{e}"
            self.debug_log(f"Runtime check subprocess failed: {e}")

        validation_msg = HumanMessage(content=validation_output)
        # THE FIX: Only update the key that the next agent explicitly expects.
        # DO NOT add the validation_msg to the general 'meta_messages' history.
        return {"recent_meta_message": validation_msg}

    def _meta_node(self, state: CreatorGraphState):
        self.debug_log("META NODE")

        if CURRENT_PHASE == "discovery":
            sys_msg = SystemMessage(
                content=DISCOVERY_META_SYSTEM_PROMPT.format(
                    DISCOVERY_MULTI_AGENT_PROMPT=DISCOVERY_MULTI_AGENT_PROMPT,
                    ANALYZER_NAME=ANALYZER_NAME,
                    STRATEGIZER_NAME=STRATEGIZER_NAME,
                    RESEARCHER_NAME=RESEARCHER_NAME,
                    CODER_NAME=CODER_NAME,
                )
            )

        else: # IMPROVEMENT PHASE
            sys_msg = SystemMessage(
                content=META_SYSTEM_PROMPT.format(
                    MULTI_AGENT_PROMPT=MULTI_AGENT_PROMPT.format(FOO_TARGET_FILENAME=FOO_TARGET_FILENAME),
                    FOO_TARGET_FILENAME=FOO_TARGET_FILENAME,
                    read_full_performance_history=read_full_performance_history(),
                    ANALYZER_NAME=ANALYZER_NAME,
                    STRATEGIZER_NAME=STRATEGIZER_NAME,
                    RESEARCHER_NAME=RESEARCHER_NAME,
                    CODER_NAME=CODER_NAME,
                )
            )
            
        msgs = state["meta_messages"][-MAX_MESSAGES_IN_AGENT:]
        tools = [think_tool]
        log_path = self.agent_log_input("META", msgs)
        output = self._tool_calling_state_graph(self.meta_llm, sys_msg, msgs, tools)
        self.agent_log_output("META", output["messages"][len(msgs):], log_path)

        #new_meta_message = HumanMessage(content=f"Temporary Meta Message ")

        # Place AI Message in the meta history
        meta_messages = state["meta_messages"] + [output["messages"][-1]]

        # Save the new_meta_message as a human message
        new_meta_message = HumanMessage(content=output["messages"][-1].content)

        return {"recent_meta_message": new_meta_message,"meta_messages": meta_messages}

    def _analyzer_node(self, state: CreatorGraphState):
        self.debug_log("ANALYZER NODE")

        if CURRENT_PHASE == "discovery":
            sys_msg = SystemMessage(
                content=DISCOVERY_ANALYZER_SYSTEM_PROMPT.format(
                    DISCOVERY_MULTI_AGENT_PROMPT=DISCOVERY_MULTI_AGENT_PROMPT,
                    ANALYZER_NAME=ANALYZER_NAME,
                )
            )
            tools = [read_local_file, think_tool, read_adapter]

            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")

            # Call the LLM with the provided tools
            msgs = state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:] + [adapter_msg, state["recent_meta_message"]]

        else: # IMPROVEMENT PHASE
            sys_msg = SystemMessage(
                content=ANALYZER_SYSTEM_PROMPT.format(
                    MULTI_AGENT_PROMPT=MULTI_AGENT_PROMPT.format(FOO_TARGET_FILENAME=FOO_TARGET_FILENAME),
                    FOO_TARGET_FILENAME=FOO_TARGET_FILENAME,
                    ANALYZER_NAME=ANALYZER_NAME,
                    MAX_MESSAGES_TOOL_CALLING=MAX_MESSAGES_TOOL_CALLING,
                )
            )
            tools = [read_local_file, think_tool, read_adapter]

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            game_output_msg = HumanMessage(content=f"This is the current game_output.txt file\n\n{read_game_output_file()}")
            game_results_msg = HumanMessage(content=f"This is the current game_results json file\n\n{read_game_results_file()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")

            # Call the LLM with the provided tools
            msgs = state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, game_output_msg, game_results_msg, current_foo_msg, adapter_msg, state["recent_meta_message"]]
        
        log_path = self.agent_log_input(ANALYZER_NAME, msgs)
        output = self._tool_calling_state_graph(self.analyzer_llm, sys_msg, msgs, tools)
        self.agent_log_output(ANALYZER_NAME, output["messages"][len(msgs):], log_path)

        # Add to Meta Messages
        response = HumanMessage(content=output["messages"][-1].content)
        meta_messages = state["meta_messages"] + [response]

        # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
        # Only summarize new messages
        #tool_call_summary = summarize_messages(output["messages"][base_len:])
        analyzer_messages = state["analyzer_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]

        return {
            "recent_helper_response": response,
            "tool_calling_messages": output["messages"],
            "meta_messages": meta_messages,
            "analyzer_messages": analyzer_messages,
        }

    def _strategizer_node(self, state: CreatorGraphState):

        self.debug_log("STRATEGIZER NODE")

        if CURRENT_PHASE == "discovery":
            sys_msg = SystemMessage(
                content=DISCOVERY_STRATEGIZER_SYSTEM_PROMPT.format(
                    DISCOVERY_MULTI_AGENT_PROMPT=DISCOVERY_MULTI_AGENT_PROMPT,
                    STRATEGIZER_NAME=STRATEGIZER_NAME,
                )
            )

            tools = [read_local_file, web_search_tool_call, think_tool, read_adapter]

            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")

            msgs = state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:] + [adapter_msg, state["recent_meta_message"]]

        else: # IMPROVEMENT PHASE
            sys_msg = SystemMessage(
                content=STRATEGIZER_SYSTEM_PROMPT.format(
                    MULTI_AGENT_PROMPT=MULTI_AGENT_PROMPT.format(FOO_TARGET_FILENAME=FOO_TARGET_FILENAME),
                    STRATEGIZER_NAME=STRATEGIZER_NAME,
                    FOO_TARGET_FILENAME=FOO_TARGET_FILENAME,
                    MAX_MESSAGES_TOOL_CALLING=MAX_MESSAGES_TOOL_CALLING,
                )
            )

            tools = [read_local_file, read_game_results_file, read_older_foo_file, web_search_tool_call, think_tool, read_adapter]

            # Call the LLM with the provided tools
            #base_len = len(state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:])

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file\n\n{read_adapter()}")

            msgs = state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, current_foo_msg, adapter_msg, state["recent_meta_message"]]
        
        log_path = self.agent_log_input(STRATEGIZER_NAME, msgs)
        output = self._tool_calling_state_graph(self.strategizer_llm, sys_msg, msgs, tools)
        self.agent_log_output(STRATEGIZER_NAME, output["messages"][len(msgs):], log_path)

        # Add to Meta Messages
        response = HumanMessage(content=output["messages"][-1].content)
        meta_messages = state["meta_messages"] + [response]

        # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
        # Only summarize new messages
        #tool_call_summary = summarize_messages(output["messages"][base_len:])
        strategizer_messages = state["strategizer_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]

        return {
            "recent_helper_response": response,
            "tool_calling_messages": output["messages"],
            "meta_messages": meta_messages,
            "strategizer_messages": strategizer_messages,
        }

    def _researcher_node(self, state: CreatorGraphState):

        self.debug_log("RESEARCHER NODE")

        if CURRENT_PHASE == "discovery":
            sys_msg = SystemMessage(
                content=DISCOVERY_RESEARCHER_SYSTEM_PROMPT.format(
                    DISCOVERY_MULTI_AGENT_PROMPT=DISCOVERY_MULTI_AGENT_PROMPT,
                    RESEARCHER_NAME=RESEARCHER_NAME,
                )
            )

        else: # IMPROVEMENT PHASE
            sys_msg = SystemMessage(
                content=RESEARCHER_SYSTEM_PROMPT.format(
                    MULTI_AGENT_PROMPT=MULTI_AGENT_PROMPT.format(FOO_TARGET_FILENAME=FOO_TARGET_FILENAME),
                    RESEARCHER_NAME=RESEARCHER_NAME,
                    FOO_TARGET_FILENAME=FOO_TARGET_FILENAME,
                    MAX_MESSAGES_TOOL_CALLING=MAX_MESSAGES_TOOL_CALLING,
                )
            )

        tools = [read_local_file, web_search_tool_call, think_tool, read_adapter]

        catanatron_files_msg = HumanMessage(content=f"This is the list of catanatron files\n\n{list_catanatron_files()}")  
        msgs = state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:] + [catanatron_files_msg, state["recent_meta_message"]]  
        log_path = self.agent_log_input(RESEARCHER_NAME, msgs)
        output = self._tool_calling_state_graph(self.researcher_llm, sys_msg, msgs, tools)
        self.agent_log_output(RESEARCHER_NAME, output["messages"][len(msgs):], log_path)

        # Add to Meta Messages
        response = HumanMessage(content=output["messages"][-1].content)
        meta_messages = state["meta_messages"] + [response]

        # Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
        researcher_messages = state["researcher_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]

        return {
            "recent_helper_response": response,
            "tool_calling_messages": output["messages"],
            "meta_messages": meta_messages,
            "researcher_messages": researcher_messages,
        }

    def _coder_node(self, state: CreatorGraphState):
        
        self.debug_log("CODER NODE")

        # Give Coder The Last Number of Meta Messages
        if len(state["meta_messages"]) > MAX_META_MESSAGES_GIVEN_TO_CODER:
            meta_msgs = state["meta_messages"][-MAX_META_MESSAGES_GIVEN_TO_CODER:]
        else:
            meta_msgs = state["meta_messages"]

        if CURRENT_PHASE == "discovery":
            sys_msg = SystemMessage(
                content=DISCOVERY_CODER_SYSTEM_PROMPT.format(
                    DISCOVERY_MULTI_AGENT_PROMPT=DISCOVERY_MULTI_AGENT_PROMPT,
                    CODER_NAME=CODER_NAME,)
            )
            tools = [read_adapter, write_adapter, replace_code_in_adapter, think_tool]

            # Call the LLM with the provided tools
            adapter_msg = HumanMessage(content=f"This is the old adapters.py file\nNow It is your turn to update it with the new recommendations from META\n\n{read_adapter()}")
            msgs = state["coder_messages"][-MAX_MESSAGES_IN_AGENT:] + meta_msgs + [adapter_msg]
        
        else: # IMPROVEMENT PHASE
            sys_msg = SystemMessage(
                content=CODER_SYSTEM_PROMPT.format(
                    MULTI_AGENT_PROMPT=MULTI_AGENT_PROMPT.format(FOO_TARGET_FILENAME=FOO_TARGET_FILENAME),
                    CODER_NAME=CODER_NAME,
                    MAX_META_MESSAGES_GIVEN_TO_CODER=MAX_META_MESSAGES_GIVEN_TO_CODER,
                    FOO_TARGET_FILENAME=FOO_TARGET_FILENAME,
                )
            )
            tools = [write_foo, replace_code_in_foo, think_tool, read_adapter]

            # Call the LLM with the provided tools
            current_foo_msg = HumanMessage(content=f"This is the old foo_player.py file\nNow It is your turn to update it with the new recommendations from META\n\n{read_foo()}")
            adapter_msg = HumanMessage(content=f"This is the current adapters.py file that you must use to interact with the Catanatron API\n\n{read_adapter()}")
            msgs = state["coder_messages"][-MAX_MESSAGES_IN_AGENT:] + meta_msgs + [current_foo_msg, adapter_msg]
        
        log_path = self.agent_log_input(CODER_NAME, msgs)
        output = self._tool_calling_state_graph(self.coder_llm, sys_msg, msgs, tools)
        self.agent_log_output(CODER_NAME, output["messages"][len(msgs):], log_path)

        # Add to Meta Messages
        response = HumanMessage(content=output["messages"][-1].content)
        meta_messages = state["meta_messages"] + [response]

        #Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
        coder_messages = state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]

        return {
            "recent_helper_response": response,
            "tool_calling_messages": output["messages"],
            "meta_messages": meta_messages,
            "coder_messages": coder_messages,
        }


    def _meta_choice(self, state: CreatorGraphState):
        """
        Conditional edge for Meta
        """
        self.debug_log("Conditional Edge Meta")

        # End evolution if we exceed max evolutions
        if (CreatorAgent.current_evolution > CREATOR_NUM_EVOLUTIONS):
            self.debug_log(f"Reached Max Evolutions of {CREATOR_NUM_EVOLUTIONS}, going to END")
            return END

        meta_message = state["meta_messages"][-1].content

        # First, try to find the chosen agent using the specific format
        match = re.search(r"CHOSEN AGENT:\s*(\w+)", meta_message)
        if match:
            agent_name = match.group(1)

            # Explicitly check for the END keyword first.
            if agent_name == "END":
                self.debug_log("Meta Message: Found END signal. Terminating graph.")
                return END
            
            if agent_name in AGENT_KEYS:
                self.debug_log(f"Meta Message: Found agent {agent_name} via specific format - going to {agent_name}")
                return agent_name

        # If not found, fall back to just searching the test
        for key in AGENT_KEYS:
            if key in meta_message:
                self.debug_log(f"Meta Message: Found {key} somewhere in text - going to {key}")
                return key
            
            # Default case if neither string is found
        self.debug_log(f"Warning: Could not determine desired agent in recent meta message. Defaulting to {ANALYZER_NAME}")
        return ANALYZER_NAME

    def create_improvement_graph(self):
        """Create a react graph for the LLM to use."""
        graph = StateGraph(CreatorGraphState)
        graph.add_node("init", self._init_node)

        graph.add_node(ANALYZER_NAME, self._analyzer_node)
        graph.add_node(STRATEGIZER_NAME, self._strategizer_node)
        graph.add_node(RESEARCHER_NAME, self._researcher_node)
        graph.add_node(CODER_NAME, self._coder_node)
        graph.add_node("run_player", self._run_player_node)

        graph.add_node("meta", self._meta_node)

        graph.add_edge(START, "init")
        graph.add_edge("init", "run_player")
        graph.add_edge("run_player", ANALYZER_NAME)
        graph.add_conditional_edges(
            "meta",
            self._meta_choice,
            {
            ANALYZER_NAME: ANALYZER_NAME,
            STRATEGIZER_NAME: STRATEGIZER_NAME,
            RESEARCHER_NAME: RESEARCHER_NAME,
            CODER_NAME: CODER_NAME,
            END: END
            }
        )

        graph.add_edge(ANALYZER_NAME, "meta")
        graph.add_edge(STRATEGIZER_NAME, "meta")
        graph.add_edge(RESEARCHER_NAME, "meta")
        graph.add_edge(CODER_NAME, "run_player")


        return graph.compile()
    
    def create_discovery_graph(self):
        """Create a react graph for the DISCOVERY phase."""
        graph = StateGraph(CreatorGraphState)
        graph.add_node("init", self._init_node)
        
        # Use the same agent nodes, they will get new prompts
        graph.add_node(ANALYZER_NAME, self._analyzer_node)
        graph.add_node(STRATEGIZER_NAME, self._strategizer_node)
        graph.add_node(RESEARCHER_NAME, self._researcher_node)
        graph.add_node(CODER_NAME, self._coder_node)
        graph.add_node("meta", self._meta_node)
        
        # Add the new validation node
        graph.add_node("validate_adapter", self._validate_adapter_node)

        graph.add_edge(START, "init")
        graph.add_edge("init", "validate_adapter") # Start by checking the file
        
        # The validation result goes to META to decide the next step
        graph.add_edge("validate_adapter", "meta")
        
        # META routes to the appropriate agent
        graph.add_conditional_edges(
            "meta",
            self._meta_choice, # Can reuse the same chooser logic
            {
                ANALYZER_NAME: ANALYZER_NAME,
                STRATEGIZER_NAME: STRATEGIZER_NAME,
                RESEARCHER_NAME: RESEARCHER_NAME,
                CODER_NAME: CODER_NAME,
                END: END
            }
        )

        # After helper agents finish, they report back to META
        graph.add_edge(ANALYZER_NAME, "meta")
        graph.add_edge(STRATEGIZER_NAME, "meta")
        graph.add_edge(RESEARCHER_NAME, "meta")
        
        # After CODER writes the code, it goes back to validation
        graph.add_edge(CODER_NAME, "validate_adapter")

        return graph.compile()

    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        """
        Orchestrates the execution of the discovery and/or improvement phases.
        """
        try:
            if START_PHASE == "discovery":
                # 1. Run Discovery Phase to create adapters.py
                self._run_phase("discovery")
                
                # 2. Automatically transition to and run the Improvement Phase
                self.debug_log("Transitioning from Discovery to Improvement phase.")
                self._run_phase("improvement")

            elif START_PHASE == "improvement":
                # Run only the Improvement Phase
                self._run_phase("improvement")
            
            else:
                self.debug_log(f"Error: Invalid START_PHASE '{START_PHASE}'. Must be 'discovery' or 'improvement'.")
                return

            # --- Finalization after all phases are complete ---
            self.debug_log("All phases complete. Copying final artifacts.")
            
            # Copy the final adapters.py
            if ADAPTER_TARGET_FILE.exists():
                shutil.copy2(
                    ADAPTER_TARGET_FILE.resolve(),
                    (Path(CreatorAgent.run_dir) / f"final_{ADAPTER_TARGET_FILENAME}")
                )
            
            # Copy the final foo_player.py
            dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")
            shutil.copy2(                           
                (FOO_TARGET_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / ("final" + dt + FOO_TARGET_FILENAME))
            )

        except Exception as e:
            self.debug_log(f"FATAL ERROR during phase execution: {e}\n{traceback.format_exc()}")
        return None

    def _run_testfoo(self, short_game: bool = False) -> str:
        """
        Run one Catanatron match and return raw CLI output.
        """
        if short_game:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_vg")
        else:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_fg")

        # Build command, strip any stale --run-id
        base_cmd = re.sub(r"--run-id=\S+", "", FOO_RUN_COMMAND).strip()
        dynamic_command = f"{base_cmd} --run-id={run_id}"

        game_run_dir = Path(CreatorAgent.run_dir) / run_id
        game_run_dir.mkdir(exist_ok=True)

        cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
        shutil.copy2(FOO_TARGET_FILE.resolve(), cur_foo_path)

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
            game_results = (stdout_limited + stderr_limited).strip()
        except subprocess.TimeoutExpired as e:
            so = (e.stdout or "")
            se = (e.stderr or "")
            if not isinstance(so, str):
                so = so.decode("utf-8", errors="ignore")
            if not isinstance(se, str):
                se = se.decode("utf-8", errors="ignore")
            stdout_limited = so[-MAX_CHARS:]
            stderr_limited = se[-MAX_CHARS:]
            game_results = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()

        # Write combined output
        output_file_path = game_run_dir / "game_output.txt"
        output_file_path.write_text(game_results, encoding="utf-8")

        # ----- Locate results JSON via run_id pattern (no stdout parsing) -----
        json_content = {}
        json_copy_path = "None"

        # Candidate run_results directories
        candidate_dirs = [
            LOCAL_CATANATRON_BASE_DIR / "run_results",
            LOCAL_CATANATRON_BASE_DIR.parent / "run_results",
        ]
        existing_dirs = [d for d in candidate_dirs if d.exists()]
        target_file = None

        if existing_dirs:
            pattern = f"{run_id}.json"
            import time, glob
            # Poll up to 3 seconds (30 * 0.1s) for file creation
            for _ in range(30):
                matches = []
                for d in existing_dirs:
                    matches.extend(Path(d).glob(pattern))
                if matches:
                    # Choose newest
                    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    target_file = matches[0]
                    break
                time.sleep(0.1)

        if target_file and target_file.exists():
            json_copy_path = game_run_dir / target_file.name
            shutil.copy2(target_file, json_copy_path)
            try:
                json_content = json.loads(target_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                json_content = {"error": "Failed to parse JSON file"}
        else:
            self.debug_log(f"[DEBUG] No results JSON found for run_id={run_id}. Pattern search failed.")

        # ----- Update performance history -----
        performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"
        try:
            performance_history = json.loads(performance_history_path.read_text())  # type: ignore
        except Exception:
            performance_history = {}

        wins = 0
        avg_score = 0
        avg_turns = 0
        try:
            if "Player Summary" in json_content:
                for player_key, stats in json_content["Player Summary"].items():
                    if "fooplayer" in player_key.lower():
                        wins = stats.get("WINS", wins)
                        for vp_key in ("AVG VP", "AVG_VP", "AVG POINTS", "AVG_VPTS"):
                            if vp_key in stats:
                                avg_score = stats[vp_key]
                                break
                        break
            if "Game Summary" in json_content:
                avg_turns = json_content["Game Summary"].get("AVG TURNS", avg_turns)
        except Exception as e:
            self.debug_log(f"[DEBUG] Stat extraction error: {e}")

        if not short_game:
            evolution_key = CreatorAgent.current_evolution
            CreatorAgent.current_evolution += 1
            self.debug_log(f"Evolution {evolution_key}: wins={wins}, avg_score={avg_score}, avg_turns={avg_turns}")

            rel_output = output_file_path.relative_to(Path(CreatorAgent.run_dir))
            rel_cur_foo = cur_foo_path.relative_to(Path(CreatorAgent.run_dir))
            rel_json = "None"
            if isinstance(json_copy_path, Path):
                rel_json = json_copy_path.relative_to(Path(CreatorAgent.run_dir))

            performance_history[f"Evolution {evolution_key}"] = {
                "wins": wins,
                "avg_score": avg_score,
                "avg_turns": avg_turns,
                "full_game_log_path": str(rel_output),
                "json_game_results_path": str(rel_json),
                "cur_foo_player_path": str(rel_cur_foo),
                "cli_run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            performance_history_path.write_text(json.dumps(performance_history, indent=2))

        return json.dumps(json_content, indent=2) if json_content else game_results


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

def write_adapter(new_text: str) -> str:
    """
    Overwrite adapters.py with new_text (UTF-8).
    """
    if len(new_text.encode()) > FOO_MAX_BYTES:
        raise ValueError("Refusing to write >64 kB")
    ADAPTER_TARGET_FILE.write_text(new_text, encoding="utf-8")
    return f"{ADAPTER_TARGET_FILENAME} updated successfully."

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

def replace_code_in_adapter(search: str, replace: str) -> str:
    """
    Replace a block of code in the adapters.py File.
    """
    content = read_adapter()
    new_content = content.replace(search, replace)
    if new_content == content:
        return "Search string not found in file. No changes made."
    write_adapter(new_content)
    return f"Successfully replaced code in {ADAPTER_TARGET_FILENAME}."

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

# Adapter

def read_adapter(_: str = "") -> str:
    """Return the UTF-8 content of adapters.py (≤64 kB), or a sentinel string if missing."""
    if not ADAPTER_TARGET_FILE.exists():
        return "(adapters.py not found)"
    if ADAPTER_TARGET_FILE.stat().st_size > 64_000:
        raise ValueError("adapters.py too large")
    return ADAPTER_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")
