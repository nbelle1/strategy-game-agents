import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from base_llm import OpenAILLM, AzureOpenAILLM, MistralLLM, BaseLLM
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from enum import Enum
from io import StringIO
from datetime import datetime
import shutil
from pathlib import Path
import subprocess, shlex


from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrockConverse
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.rate_limiters import InMemoryRateLimiter
from typing_extensions import TypedDict
from typing_extensions import TypedDict

# import warnings
# from langchain.warnings import LangChainDeprecationWarning   # same class they raise
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)



# -------- tool call configuration ----------------------------------------------------
LOCAL_SEARCH_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
PROMPT_BASE_FILENAME = "base_prompt.txt"
PROMPT_NEW_FILENAME = "current_prompt.txt"
PROMPT_BASE_FILE = Path(__file__).parent / PROMPT_BASE_FILENAME
PROMPT_NEW_FILE = Path(__file__).parent / PROMPT_NEW_FILENAME
FOO_TARGET_FILENAME = "promptRefiningLLM_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap
OUR_PLAYER = "PE"

PLAYER_CODE_TO_NAME = {
    "R": "RandomPlayer",
    "W": "WeightedRandomPlayer",
    "VP": "VictoryPointPlayer",
    "G": "GreedyPlayoutsPlayer",
    "M": "MCTSPlayer",
    "F": "ValueFunctionPlayer",
    "AB": "AlphaBetaPlayer",
    "SAB": "SameTurnAlphaBetaPlayer",
    # New Players
    "BA": "BaseAgentPlayer",
    "SA": "StructuredAgentPlayer",
    "PE": "PromptEvolverPlayer",
    "AE": "AgentEvolverFooPlayer",
    "LAE": "LLMAgentEvolverFooPlayer",
}

EVOLVE_COUNTER_MAX = 10
# Set winning points to 5 for quicker game
FOO_RUN_COMMAND = f"catanatron-play --players=AB,{OUR_PLAYER} --num=5 --config-vps-to-win=10"
RUN_TEST_FOO_HAPPENED = False # Used to keep track of whether the testfoo tool has been called
# -------------------------------------------------------------------------------------

class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None
    current_evolution = 0

    def __init__(self):
        # Get API key from environment variable
        self.llm_name = "gpt-4o"
        self.llm = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint="# TODO: SET AZURE ENDPOINT",
            api_version = "2024-12-01-preview"
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


        # self.llm_name = "mistral-large-latest"
        # rate_limiter = InMemoryRateLimiter(
        #     requests_per_second=1,    # Adjust based on your API tier
        #     check_every_n_seconds=1,
        #     max_bucket_size=1        # Allows for burst handling
        # )
        # self.llm = ChatMistralAI(
        #     model="mistral-large-latest",
        #     temperature=0,
        #     max_retries=2,
        #     rate_limiter=rate_limiter,
        # )

        # self.llm_name = "claude-3.7"
        # self.llm = ChatBedrockConverse(
        #     aws_access_key_id = os.environ["AWS_ACESS_KEY"],
        #     aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
        #     region_name = "us-east-2",
        #     provider = "anthropic",
        #     model_id="# TODO: SET MODEL ID"
        # )

        # Create run directory if it doesn't exist
        if CreatorAgent.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            CreatorAgent.run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(CreatorAgent.run_dir, exist_ok=True)
            
            # Initialize performance_history.json
            performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"
            performance_history = {}
            with open(performance_history_path, 'w') as f:
                json.dump(performance_history, f, indent=2)

        # Create a marker file to allow the player to detect the current run directory
        with open(os.path.join(runs_dir, "current_run.txt"), "w") as f:
            f.write(CreatorAgent.run_dir)



        # Copy the base prompt to new prompt at the start of each run
        shutil.copy2(
            PROMPT_BASE_FILE.resolve(),
            PROMPT_NEW_FILE.resolve()
        )

        shutil.copy2(
            PROMPT_BASE_FILE.resolve(),
            Path(CreatorAgent.run_dir) / f"initial_{PROMPT_BASE_FILENAME}"
        )

        #Copy the Blank FooPlayer to the run directory
        # shutil.copy2(                           # ↩ copy with metadata
        #     (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),  # ../foo_player.py
        #     FOO_TARGET_FILE.resolve()          # ./foo_player.py
        # )

        self.memory_config = {
            "recursion_limit": 100, # set recursion limit for graph
            "configurable": {
                "thread_id": "1"
            }
        }
        #self.memory_config = {"configurable": {"thread_id": "1"}}
        self.num_memory_messages = 10        # Trim number of messages to keep in memory to limit API usage
        self.react_graph = self.create_langchain_react_graph()

    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        

        class CreatorGraphState(TypedDict):
            full_results: SystemMessage # Last results of running the game
            summary: HumanMessage # Summary of the conversation
            #analysis: AIMessage         # Output of Anlayzer, What Happend?
            #solution: AIMessage         # Ouput of Researcher, What should be done?
            #code_additions: AIMessage         # Output of Coder, What was added to the code?
            #test_results: SystemMessage # Running a test on code, to ensure correctness
            #validation: AIMessage       # Ouptut of Validator, Is the code correct?
            tool_calling_messages: list[AnyMessage]     # Messages from the tool calling state graph (used for debugging)

            evolve_counter: int         # Counter for the number of evolutions

        multi_agent_prompt = f"""You are apart of a multi-agent system that is working to create the best prompt for an LLM playing the game Catan.\n\tYour specific role is the:"""

        #tools = [add, multiply, divide]
        DEFAULT_EVOLVE_COUNTER = 3

        analyzer_continue_key = "CONTINUE_EVOLVING"
        analyzer_stop_key = "STOP_EVOLVING"


        val_ok_key = "PASSSED_VALIDATION"
        val_not_ok_key = "FAILED_VALIDATION"



        def tool_calling_state_graph(sys_msg: SystemMessage, msgs: list[AnyMessage], tools):
            # Node

            # Bind Tools to the LLM
            llm_with_tools = self.llm.bind_tools(tools, parallel_tool_calls=False)

            def assistant(state: MessagesState):
                return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
            
            # Graph
            builder = StateGraph(MessagesState)

            # Define nodes: these do the work
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))

            # Define edges: these determine how the control flow moves
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition,
            )
            builder.add_edge("tools", "assistant")
            react_graph = builder.compile()
            
            
            result = react_graph.invoke({"messages": msgs})
            
            for m in result['messages']:
                m.pretty_print()

            return {"messages": result['messages']}

        def run_player_node(state: CreatorGraphState):
            """
            Runs Catanatron with the current Code
            """
            #print("In Run Player Node")

            # Generate a test results (later will be running the game)
            game_results = run_gamefoo()
            #output = llm.invoke([HumanMessage(content="Create me a complicated algebra math problem, without a solution. Return only the problem...NO COMMENTARY!!")])
            
            # Clear all past messages
            return {
                "full_results": HumanMessage(content=f"GAME RESULTS:\n\n{game_results}"),
                "analysis": AIMessage(content=""),
                # "solution": AIMessage(content=""),
                # "code_additions": AIMessage(content=""),
                # "test_results": SystemMessage(content=""),
                # "validation": AIMessage(content=""),
                # "tool_calling_messages": []
            }

        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")

            # If evolve_counter isnt initialized, set it to 0. If it is, increment it
            if "evolve_counter" not in state:
                evolve_counter = DEFAULT_EVOLVE_COUNTER
            else:
                evolve_counter = state["evolve_counter"] - 1

            sys_msg = SystemMessage(content =
                (
                    f"""
                    You are in charge of creating the prompt for the Catan Player promptRefiningLLM_player in {FOO_TARGET_FILENAME}. 
                    
                    YOUR PRIMARY GOAL: Create a prompt that helps our promptRefiningLLM_player win against its opponent AlphaBetaPlayer.
                    
                    IMPROVEMENT PROCESS:
                    1. Carefully analyze game logs to identify key weaknesses in our player
                    3. Research specific Catan strategies online to address those weaknesses
                    4. Read and update the current prompt with targeted improvements
                    5. Test again and iterate until we reach the highest win rate possible
                    
                    STRATEGY AREAS TO FOCUS ON:
                    - Early game placement strategy
                    - Mid-game resource management
                    - Late-game victory point optimization
                    - Effective trading and negotiation
                    - Development card usage
                    - Robber placement strategy
                    
                    You Have the Following Tools at Your Disposal:
                    - list_local_files: List all files in the current directory.
                    - read_local_file: Read the content of a file in the current directory.
                    - read_output_file: Read the content of the game output file using the path retrieved from performance history.
                    - read_performance_history: Read the performance history from the previous evolutions with each of their prompts.
                    - read_prompt: Read the content of {PROMPT_NEW_FILENAME}.
                    - write_prompt: Write the content of {PROMPT_NEW_FILENAME}. (Any text in brackets MUST remain in brackets)
                    - web_search_tool_call: Research strategies for specific aspects of Catan gameplay.
                    
                    PROCESS GUIDELINES:
                    1. After each run, analyze the results to identify specific weaknesses. Each run includes the results of 5 games.
                    2. Review performance history to see if the prompt used is better or worse that other prompts.
                    3. Optionally read logging files listed in performance history for a better understanding of how an evolution's prompt impacted decisions made during the game and how that impacted results.
                    4. Use web_search_tool_call to research strategies addressing weaknesses that you recognize.
                    5. Hypothesis ways that you can improve the prompt with the things that you learned.
                    4. Read and modify the prompt with these improvements, being specific and detailed.
                    5. Once you have called enough tools and have written the new prompt, stop calling tools to allow the player to run a test game.
                    
                    Keep iterating and improving your prompt until the player consistently wins.
                    """
                )
            )
            
            # Check if state contains summary, if it does add it to msg
            if "summary" in state:
                summary = state["summary"].content
                # print("Summary exists")
                msg = [state["summary"], state["full_results"]]

            else:
                # print("Summary Does not exist")
                msg = [state["full_results"]]
                summary = ""

            # TODO: Add all tools Prompt Tools
            tools = [list_local_files, read_local_file, read_performance_history, read_prompt, write_prompt, read_output_file, web_search_tool_call]
            output = tool_calling_state_graph(sys_msg, msg, tools)


            # Create our summarization prompt 
            if summary != "":
                # A summary already exists
                summary_message = (
                    f"{multi_agent_prompt} Summarizer\n"
                    f"This is the current summary of all agent activity prior to this evolution: {summary}\n\n"
                    "YOUR GOAL:\n"
                    "Create an updated summary that incorporates both the existing summary and the most recent tool calls.\n\n"
                    "FOCUS ON:\n"
                    "1. What key insights have been learned throughout all evolutions\n" 
                    "2. How the prompt has evolved since the first version\n"
                    "3. What strategies have been implemented in the prompt\n"
                    "4. What performance improvements have been observed\n"
                    "5. What challenges still remain\n\n"
                    "6. What tools were called in this evolution and for what reason.\n"
                    "Keep track of the sequential evolution steps taken so far, and note this is evolution #{CreatorAgent.current_evolution}.\n"
                    "Use minimal tokens while preserving the key learnings and chronology.\n"
                    "RETURN THE FULL SUMMARY, and make sure to start your output with 'SUMMARY:' and end with 'SUMMARY'"
                )
                
            else:
                summary_message = (
                    "Create a summary of the conversation above that includes:\n"
                    "1. What the agent has learned from the game results\n"
                    "2. What strategies were implemented in the prompt\n"
                    "3. Performance metrics observed\n"
                    "Start your output with 'SUMMARY:' and end with 'SUMMARY'"
                )

            # print(f"\nCURRENT SUMMARY:\n {summary}")
            # state_msgs = state["full_results"] + output["messages"]
            # messages = state_msgs + [HumanMessage(content=summary_message)]
            # summarizer_response = self.llm.invoke(messages)

  
            state_msgs = output["messages"]
            # print("\nTOOL CALLS MADE:")
            # for i, message in enumerate(output["messages"]):
            #     print(f"Message {i+1}:")
            #     print(f"  Type: {message.type}")
            #     print(f"  Content: {message.content[:200]}...")  # Show first 200 chars
            messages = state_msgs + [HumanMessage(content=summary_message)]
            summarizer_response = HumanMessage(content=self.llm.invoke(messages).content)
            
            # print(f"\nRETURNED LLM SUMMARY:\n {summarizer_response.content}")


            #print(output)
            return {"evolve_counter": evolve_counter, "tool_calling_messages": output["messages"], "summary": summarizer_response}

        
        def continue_evolving_analyzer(state: CreatorGraphState):
            """
            Conditional edge for Analyzer
            """
            print(f"In Analyzer Condition Edge Counter: {state['evolve_counter']}")
            
            # Get the content of the validation message
            #analyzer_message = state["analysis"].content
            
            # Check for the presence of our defined result strings (Because analyzer node decrements the counter)
            if state["evolve_counter"] <= -1:
                print("Evolve counter is 0 - ending workflow")
                return END
            else:
                return "run_player"
            # if analyzer_stop_key in analyzer_message:
            #     print("Validation passed - ending workflow")
            #     return END
            # elif analyzer_continue_key in analyzer_message:  #
            #     print("Validation failed - rerunning player")
            #     return "researcher"
            # else:
            #     # Default case if neither string is found
            #     print("Warning: Could not determine validation result, defaulting to researcher")
            #     return END


        def construct_graph():
            graph = StateGraph(CreatorGraphState)

            graph.add_node("run_player", run_player_node)
            graph.add_node("analyzer", analyzer_node)
            #graph.add_node("tools", ToolNode(tools))
            # graph.add_node("researcher", researcher_node)
            # graph.add_node("coder", coder_node)
            # graph.add_node("test_player", test_player_node)
            # graph.add_node("validator", validator_node)

            graph.add_edge(START, "run_player")
            graph.add_edge("run_player", "analyzer")
            graph.add_conditional_edges(
                "analyzer",
                continue_evolving_analyzer,
                {END, "run_player"}
            )
            # #graph.add_edge("analyzer", "researcher")
            # graph.add_edge("researcher", "coder")
            # graph.add_edge("coder", "test_player")
            # graph.add_edge("test_player", "validator")
            # graph.add_conditional_edges(
            #     "validator",
            #     code_ok_validator,
            #     {"coder", "run_player"}
            # )

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

            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{self.llm_name}.txt")

            #initial_state: CreatorGraphState = {}

            # class CreatorGraphState(TypedDict):
            #     full_results: SystemMessage # Last results of running the game
            #     analysis: AIMessage         # Output of Anlayzer, What Happend?
            #     solution: AIMessage         # Ouput of Researcher, What should be done?
            #     code_additions: AIMessage         # Output of Coder, What was added to the code?
            #     test_results: SystemMessage # Running a test on code, to ensure correctness
            #     validation: AIMessage       # Ouptut of Validator, Is the code correct?
            #     tool_calling_messages: list[AnyMessage]     # Messages from the tool calling state graph (used for debugging)

            #     evolve_counter: int         # Counter for the number of evolutions

            CREATOR_STATE_KEYS = {
                "full_results",
                "analysis",
            #     "solution",
            #     "code_additions",
            #     "test_results",
            #     "validation",
                "tool_calling_messages",
                "evolve_counter",
            }

            # def _pretty_message(msg, indent="  "):
            #     """Return a human‑readable one‑liner for any BaseMessage instance."""
            #     # BaseMessage subclasses all expose .type (str) and .content (str | list)
            #     role = getattr(msg, "type", msg.__class__.__name__)
            #     return f"{indent}[{role}] {msg.content}"

            # # --- streaming ------------------------------------------------------------- #
            # with open(log_path, "a", encoding="utf‑8") as log_file:              # ❶
            #     for chunk in self.react_graph.stream({}, stream_mode="updates"):   # ❷
            #         # `chunk` looks like {"node_name": {"state_key": value, ...}}
            #         for node, update in chunk.items():                             # ❸
            #             for key, value in update.items():
            #                 if key not in CREATOR_STATE_KEYS:
            #                     continue                                           # skip everything else

            #                 # ---- normal state fields --------------------------------- #
            #                 if key != "tool_calling_messages":
            #                     line = f"{node}.{key}: {value}"
            #                     print(line)
            #                     log_file.write(line + "\n")
            #                     continue

            #                 # ---- tool_calling_messages (list[AnyMessage]) ------------ #
            #                 print(f"{node}.tool_calling_messages:")
            #                 log_file.write(f"{node}.tool_calling_messages:\n")
            #                 for i, msg in enumerate(value):
            #                     line = _pretty_message(msg, indent="    ")
            #                     print(line)
            #                     log_file.write(line + "\n")
            with open(log_path, "a") as log_file:                # Run the graph until the first interruption
                for step in self.react_graph.stream({"evolve_counter": EVOLVE_COUNTER_MAX}, self.memory_config, stream_mode="updates"):
                    #print(step)
                    #log_file.write(f"Step: {step.}\n")
                    for node, update in step.items():
                        if node == "run_player":
                            print("In run_player node")
                        elif node == "analyzer":
                            print("In analyzer node")

                        

                        if "tool_calling_messages" in update:
                            count = 0
                            for msg in update["tool_calling_messages"]:
                                #print(msg)
                                msg.pretty_print()
                                if isinstance(msg, ToolMessage):
                                    print("Tool Message: ", msg.name)
                                count += 1
                                log_file.write((msg).pretty_repr())
                            print(f"Number of Tool Calling Messages: {count}")

                        if "analysis" in update:
                            msg = update["analysis"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "evolve_counter" in update:
                            print("ENVOLVE COUNTER: ", update["evolve_counter"])
                            log_file.write(f"Evolve Counter: {update['evolve_counter']}\n")
                        if "full_results" in update:
                            print("Full Results:", update["full_results"])
                            log_file.write(f"Full Results: {update['full_results']}\n")

                        
        # if "run_player" in step:
            
        # if "tool_calling_messages" in step:
        #     for msg in step["tool_calling_messages"]:
        #         #print(msg)
        #         msg.pretty_print()
        #         log_file.write((msg).pretty_repr())
        # if "analysis" in step:
        #     #print(step["analysis"])
        #     msg = step["analysis"]
        #     msg.pretty_print()
        #     log_file.write((msg).pretty_repr())


            # with open(log_path, "a") as log_file:                # Run the graph until the first interruption
            #     for step in self.react_graph.stream({}, stream_mode="updates"):
            #         #print(step)
            #         #log_file.write(f"Step: {step.}\n")
            #         if "tool_calling_messages" in step:
            #             for msg in step["tool_calling_messages"]:
            #                 #print(msg)
            #                 msg.pretty_print()
            #                 log_file.write((msg).pretty_repr())
            #         if "analysis" in step:
            #             #print(step["analysis"])
            #             msg = step["analysis"]
            #             msg.pretty_print()
            #             log_file.write((msg).pretty_repr())


                    #msg = step['messages'][-1]
                    #msg.pretty_print()
                    #log_file.write((msg).pretty_repr())


            print("✅  graph finished")

            # Copy Result File to the new directory
            dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")

            # Copy Result File to the new directory
            shutil.copy2(                           
                (PROMPT_NEW_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / f"final_{PROMPT_NEW_FILENAME}")
            )

        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
        return None


def list_local_files(_: str = "") -> str:
    """Return all files beneath BASE_DIR, one per line."""
    return "\n".join(
        str(p.relative_to(LOCAL_SEARCH_BASE_DIR))
        for p in LOCAL_SEARCH_BASE_DIR.glob("**/*")
        if p.is_file() and p.suffix in {".py", ".txt", ".md"}
    )


def read_local_file(rel_path: str) -> str:
    """
    Return the text content of rel_path if it’s inside BASE_DIR.
    Args:
        rel_path: Relative path to the file to read.
    """
    if rel_path.endswith("game_output.txt") and "runs/" in rel_path:
        try:
            # First try as absolute path
            abs_path = Path(rel_path)
            if not abs_path.is_absolute():
                # If relative, try from project root (parent of parent of parent of file)
                project_root = Path(__file__).parent.parent.parent
                abs_path = project_root / rel_path
                
            if abs_path.exists() and abs_path.is_file():
                if abs_path.stat().st_size > 64_000:
                    return f"File {rel_path} is too large (showing first 64KB):\n\n" + abs_path.read_text(encoding="utf-8", errors="ignore")[:64_000]
                return abs_path.read_text(encoding="utf-8", errors="ignore")
            else:
                return f"Game output file not found at {abs_path}"
        except Exception as e:
            return f"Error reading game output file: {str(e)}"
    if rel_path == PROMPT_NEW_FILENAME:
        return read_prompt()
    candidate = (LOCAL_SEARCH_BASE_DIR / rel_path).resolve()
    if not str(candidate).startswith(str(LOCAL_SEARCH_BASE_DIR)) or not candidate.is_file():
        raise ValueError("Access denied or not a file")
    if candidate.stat().st_size > 64_000:
        raise ValueError("File too large")
    return candidate.read_text(encoding="utf-8", errors="ignore")


# def read_foo(_: str = "") -> str:
#     """
#     Return the UTF-8 content of Agent File (≤64 kB).
#     """
#     if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
#         raise ValueError("File too large for the agent")
#     return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")  # pathlib API :contentReference[oaicite:2]{index=2}


# def write_foo(new_text: str) -> str:
#     """
#     Overwrite Agent File with new_text (UTF-8).
#     """
#     if len(new_text.encode()) > FOO_MAX_BYTES:
#         raise ValueError("Refusing to write >64 kB")
#     FOO_TARGET_FILE.write_text(new_text, encoding="utf-8")                 # pathlib write_text :contentReference[oaicite:3]{index=3}
    
#     # Copy Result File to the new directory
#     dt = datetime.now().strftime("%Y%m%d_%H%M%S_")

#     shutil.copy2(                           
#         (FOO_TARGET_FILE).resolve(),
#         (Path(CreatorAgent.run_dir) / (dt + FOO_TARGET_FILENAME))
#     )

#     return f"{FOO_TARGET_FILENAME} updated successfully"

def read_prompt(_: str = "") -> str:
    """Return the UTF-8 content of base_prompt.txt (≤16 kB)."""
    if PROMPT_BASE_FILE.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return PROMPT_BASE_FILE.read_text(encoding="utf-8", errors="ignore")

def write_prompt(new_text: str) -> str:
    """Overwrite current_prompt.txt with new_text (UTF-8)."""
    if len(new_text.encode()) > FOO_MAX_BYTES:
        raise ValueError("Refusing to write >16 kB")
    PROMPT_NEW_FILE.write_text(new_text, encoding="utf-8")
    return f"{PROMPT_NEW_FILENAME} updated successfully"

def read_output_file(file_path: str) -> str:
    """
    Return the content of a game output file at the specified path.
    
    Args:
        file_path: Full path to the game output file. Found in performance history.
    
    Returns:
        The content of the file or an error message.
    """
    try:
        # Convert string path to Path object
        path = Path(file_path)
        
        # If it's a relative path, try to resolve it from project root
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path
        
        if not path.exists():
            return f"File not found at {file_path}"
            
        if not path.is_file():
            return f"Path exists but is not a file: {file_path}"
            
        # Check file size and limit if needed
        if path.stat().st_size > 64_000:
            return f"File {file_path} is too large (showing first 64KB):\n\n" + path.read_text(encoding="utf-8", errors="ignore")[:64_000]
            
        # Read and return file content
        return path.read_text(encoding="utf-8", errors="ignore")
        
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def read_performance_history(_: str = "") -> str:
    """Return the content of performance_history.json as a string (≤16 kB)."""
    performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"

    if not performance_history_path.exists():
        return "Performance history file does not exist."
    
    if performance_history_path.stat().st_size > 100_000:
        return "Performance history file is too large (>100 KB). Consider truncating or summarizing it."
    
    with open(performance_history_path, 'r') as f:
        performance_history = json.load(f)
        return json.dumps(performance_history, indent=2)
    
def run_gamefoo(_: str = "") -> str:
    """
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    """    

    run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S")
    game_run_dir = Path(CreatorAgent.run_dir) / run_id
    game_run_dir.mkdir(exist_ok=True)
    
    # Save the current prompt used for this game
    prompt_copy_path = game_run_dir / "prompt_used.txt"
    shutil.copy2(PROMPT_NEW_FILE, prompt_copy_path)
    prompt_content = PROMPT_NEW_FILE.read_text(encoding="utf-8")

    os.environ["CATAN_CURRENT_RUN_DIR"] = str(CreatorAgent.run_dir)


    result = subprocess.run(
        shlex.split(FOO_RUN_COMMAND),
        capture_output=True,          # capture stdout+stderr :contentReference[oaicite:1]{index=1}
        text=True,
        timeout=14400,                  # avoids infinite-loop hangs
        check=False                   # we’ll return non-zero output instead of raising
    )

    # Save the output to a log file in the game run directory
    output_file_path = game_run_dir / "game_output.txt"
    relative_output_path = os.path.relpath(output_file_path, Path(CreatorAgent.run_dir).parent.parent)
    
    game_results = (result.stdout + result.stderr).strip()

    # Update the run_test_foo flag to read game results
    global RUN_TEST_FOO_HAPPENED
    RUN_TEST_FOO_HAPPENED = True

    with open(output_file_path, "w") as output_file:
        output_file.write(game_results)

    # Search for the JSON results file path in the output
    json_path = None
    import re
    path_match = re.search(r'results_file_path:([^\s]+)', game_results)
    if path_match:
        # Extract the complete path
        json_path = path_match.group(1).strip()

    json_content = {}
    # If we found a JSON file path, copy it and load its contents
    if json_path and Path(json_path).exists():
        # Copy the JSON file to our game run directory
        json_filename = Path(json_path).name
        json_copy_path = game_run_dir / json_filename
        shutil.copy2(json_path, json_copy_path)
        
        # Load the JSON content
        try:
            with open(json_path, 'r') as f:
                json_content = json.load(f)
        except json.JSONDecodeError:
            json_content = {"error": "Failed to parse JSON file"}


        # Update performance_history.json
        performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"
        
        try:
            # Load existing performance history
            with open(performance_history_path, 'r') as f:
                performance_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            performance_history = {}
        
        # Extract relevant data from json_content
        wins = 0
        avg_score = 0
        avg_turns = 0
        
        try:
            # Extract data from the JSON structure
            if "Player Summary" in json_content:
                our_player = PLAYER_CODE_TO_NAME.get(OUR_PLAYER, OUR_PLAYER)
                for player, stats in json_content["Player Summary"].items():
                    if player.startswith(our_player):  # Check if player key starts with our player's name
                        if "WINS" in stats:
                            wins = stats["WINS"]
                        if "AVG VP" in stats:
                            avg_score = stats["AVG VP"]
                        
            if "Game Summary" in json_content:
                if "AVG TURNS" in json_content["Game Summary"]:
                    avg_turns = json_content["Game Summary"]["AVG TURNS"]
        except Exception as e:
            print(f"Error extracting stats from JSON: {e}")
            
        # Create or update the entry for this evolution
        evolution_key = CreatorAgent.current_evolution
        CreatorAgent.current_evolution += 1
        performance_history[f"Evolution {evolution_key}"] = {
            "prompt_used": prompt_content,
            "wins": wins,
            "avg_score": avg_score,
            "avg_turns": avg_turns,
            "full_game_log_path": str(output_file_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write updated performance history
        with open(performance_history_path, 'w') as f:
            json.dump(performance_history, f, indent=2)
        
    if json_content:
        return json.dumps(json_content, indent=2)
    else:
        # If we didn't find a JSON file, return a limited version of the game output
        MAX_CHARS = 5_000
        stdout_limited = result.stdout[-MAX_CHARS:]
        stderr_limited = result.stderr[-MAX_CHARS:]
        return (stdout_limited + stderr_limited).strip()


def run_testfoo(short_game: bool = False) -> str:
    """
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    Input: short_game (bool): If True, run a short game with a 30 second timeout.
    """    
    MAX_CHARS =20_000                      

    try:
        if short_game:
            result = subprocess.run(
                shlex.split(FOO_RUN_COMMAND),
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
        else:
            result = subprocess.run(
                shlex.split(FOO_RUN_COMMAND),
                capture_output=True,
                text=True,
                timeout=3600,
                check=False
            )
        stdout_limited  = result.stdout[-MAX_CHARS:]
        stderr_limited  = result.stderr[-MAX_CHARS:]
        game_results = (stdout_limited + stderr_limited).strip()
    except subprocess.TimeoutExpired as e:
        # Handle timeout case
        stdout_output = e.stdout or ""
        stderr_output = e.stderr or ""
        if stdout_output and not isinstance(stdout_output, str):
            stdout_output = stdout_output.decode('utf-8', errors='ignore')
        if stderr_output and not isinstance(stderr_output, str):
            stderr_output = stderr_output.decode('utf-8', errors='ignore')
        stdout_limited  = stdout_output[-MAX_CHARS:]
        stderr_limited  = stderr_output[-MAX_CHARS:]
        game_results = "TIMEOUT: Game exceeded time limit.\n\n" + (stdout_output + stderr_output).strip()
    
    # Update the run_test_foo flag to read game results
    global RUN_TEST_FOO_HAPPENED
    RUN_TEST_FOO_HAPPENED = True

    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]

    # Add a file with the stdout and stderr called catanatron_output.txt
    output_file_path = latest_run_folder / "catanatron_output.txt"
    with open(output_file_path, "w") as output_file:
        output_file.write(game_results)
        
    #print(game_results)
        # limit the output to a certain number of characters
    return game_results




def web_search_tool_call(query: str) -> str:
    """Perform a web search using the Tavily API.

    Args:
        query: The search query string.

    Returns:
        The search result as a string.
    """
    try:
        # Get the API key from environment variable
        api_key = os.environ.get("TAVILY_API_KEY")
        
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable is not set."
        
        # Create search instance with explicit API key
        tavily_search = TavilySearchResults(
            max_results=3,
            api_key=api_key
        )
        
        # Perform the search
        search_docs = tavily_search.invoke(query)
        
        # Format the search results
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return formatted_search_docs
    
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def view_last_game_llm_query(query_number: int = -1) -> str:
    """
    View the game results from a specific run.
    
    Args:
        query_number: The index of the file to view (0-based). 
                     If -1 (default), returns the most recent file.
    
    Returns:
        The content of the requested game results file or an error message.
    """

    if RUN_TEST_FOO_HAPPENED == False:
        return "No game run has been executed yet."
    
    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]
    
    # Get all files in the folder and sort them
    result_files = sorted(latest_run_folder.glob("*"))
    
    if not result_files:
        return f"No result files found in {latest_run_folder.name}."
    
    # Determine which file to read
    file_index = query_number if query_number >= 0 else len(result_files) - 1
    
    # Check if index is valid
    if file_index >= len(result_files):
        return f"Invalid file index. There are only {len(result_files)} files (0-{len(result_files)-1})."
    
    target_file = result_files[file_index]
    
    # Read and return the content of the file
    try:
        with open(target_file, "r") as file:
            return f"Content of {target_file.name}:\n\n{file.read()}"
    except Exception as e:
        return f"Error reading file {target_file.name}: {str(e)}"
    

def view_last_game_results(_: str = "") -> str:
    """
    View the game results from a specific run.
    
    Returns:
        The content of the requested game results file or an error message.
    """

    if RUN_TEST_FOO_HAPPENED == False:
        return "No game run has been executed yet."
    
    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]

    # Read a file with the stdout and stderr called catanatron_output.txt
    output_file_path = latest_run_folder / "catanatron_output.txt"
    
    # Read and return the content of the file
    try:
        with open(output_file_path, "w") as file:
            return f"Content of {output_file_path.name}:\n\n{file.read()}"
    except Exception as e:
        return f"Error reading file {output_file_path.name}: {str(e)}"
    




