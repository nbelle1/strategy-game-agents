import os
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
LANGCHAIN_TRACING_VAR = "true"

LLM_BACKEND = "mistral"  # "openai", "mistral", or "claude"
LLM_MODEL = "mistral-large-latest"
#LLM_MODEL = "devstral-medium-latest"

FOO_MAX_BYTES   = 64_000      # context-friendly cap
CREATOR_LANGRAPH_RECURSION_LIMIT = 200  # max depth of graph recursion
CREATOR_NUM_EVOLUTIONS = 20

MAX_MESSAGES_TOOL_CALLING = 4
MAX_META_MESSAGES_GIVEN_TO_CODER = 6
MAX_MESSAGES_IN_AGENT = 20

# CONSTANTS
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_RUN_COMMAND = "catanatron-play --players=AB,AE2  --num=10 --config-map=MINI  --config-vps-to-win=10"

MULTI_AGENT_PROMPT = f"""You are apart of a multi-agent system that is working to evolve the code in {FOO_TARGET_FILENAME} to become the best player in the Catanatron Minigame.\n\tYour specific role is the:"""
ANALYZER_NAME = "ANALYZER"
STRATEGIZER_NAME = "STRATEGIZER"
RESEARCHER_NAME = "RESEARCHER"
CODER_NAME = "CODER"
AGENT_KEYS = [ANALYZER_NAME, STRATEGIZER_NAME, RESEARCHER_NAME, CODER_NAME]

# VARIABLES


###################################################################################################
#  LANGGRAPH
###################################################################################################
class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None
    current_evolution = 0

    def __init__(self):
                # Get API key from environment variable

        self.llm_name = LLM_MODEL
        if LLM_BACKEND == "openai":
            self.llm = ChatOpenAI(
                model=self.llm_name,
                max_retries=10,
            )
        elif LLM_BACKEND == "mistral":
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=1,    # Adjust based on your API tier
                check_every_n_seconds=0.1,
            )
            self.llm = ChatMistralAI(
                model=self.llm_name,
                temperature=0,
                max_retries=10,
                rate_limiter=rate_limiter,
            )
        elif LLM_BACKEND == "claude":
            self.llm = ChatBedrockConverse(
                aws_access_key_id = os.environ["AWS_ACESS_KEY"],
                aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
                region_name = "us-east-2",
                provider = "anthropic",
                model_id="# TODO: ADD MODEL ID"
            )
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

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
            # "configurable": {
            #     "thread_id": "1"
            # }
        }
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

        def tool_calling_state_graph(sys_msg: SystemMessage, msgs: list[AnyMessage], tools):
            # Bind Tools to the LLM
            #llm_with_tools = self.llm.bind_tools(tools, parallel_tool_calls=False)
            llm_with_tools = self.llm.bind_tools(tools)

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

            # Save tools to log file
            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{self.llm_name}_tools.txt")
            with open(log_path, "a") as log_file:
                for m in last_event['messages']:
                    log_file.write(m.pretty_repr())

            return last_event

        def init_node(state: CreatorGraphState):
            """
            Initialize the state of the graph
            """
            print("In Init Node")
            
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

        def run_player_node(state: CreatorGraphState):
            """
            Runs Catanatron with the current Code
            """
            #print("In Run Player Node")

            # Generate a test results (later will be running the game)
            game_results = run_testfoo(short_game=False)
            game_msg = HumanMessage(content=f"GAME RESULTS:\n\n{game_results}")

            meta_messages = state["meta_messages"] + [game_msg]

            # Create a dummy meta message to automatically generate a summary of the last run_pla
            defualt_analyze_msg = HumanMessage(content=f"""
ANALYZER OBJECTIVE:

If there is no syntax errors, I want you to return
    - The Scores of the {FOO_TARGET_FILENAME} player from the game_results json file
    - Short analysis of the game output (return anything interseting that was printed)
    - EMPHASIZE any errors, warnings, or signs of player implementation error in the game_output.txt file 

If there is a syntax error, I want you to return
    - The error message from the game_output.txt file
    - The exact line number of the error if possible
    - The exact line of code that caused the error if possible

Keep the Response as concise as possible
Start your response with "After Running The New {FOO_TARGET_FILENAME} Player, Here is my analysis and findings:"
"""
            )
            # Clear all past messages
            return {
                "game_results": game_msg,
                "recent_meta_message": defualt_analyze_msg,
                "meta_messages": meta_messages,
            }

        def meta_node(state: CreatorGraphState):

            sys_msg = SystemMessage(
                content=f"""
{MULTI_AGENT_PROMPT} META SUPERVISOR

Task: You are the highest level of intelligence, and you must think critically about all your outputs.

META HIGH LEVEL GOAL: Learn how to create a Catanatron player in {FOO_TARGET_FILENAME} that can win games against the opponent

<Performance History>
Here is your Current Performance History for Evolving the {FOO_TARGET_FILENAME} player:
{read_full_performance_history()}
</Performance History>

<Available Tools>
You have access to three main tools:
1. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool to plan your approach if you feel like you need to think deeper. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
1st Step: Look at the previous messages and take note of your previous goals, and the newest information provided to you
    - Be sure to carefully consider what the analyzer is saying regarding the game output
    - If needed, use think_tool to reflect on your current situation and plan your next steps

2nd Step: Output your current META MEDIUM LEVEL GOAL, and META LOW LEVEL GOAL at the top of your message

3rd Step: Determine the sub-agent that you wish to consult, and prepare an OBJECTIVE message for them
    - If your performance history has not improved in the last three evolutions or stays at 0, consult the strategizer
</Instructions>

<AGENTS>
    {ANALYZER_NAME}: Analyer has access to the performance history, and the {FOO_TARGET_FILENAME}.py, game_output.txt, and game_results*.json for all the previous games/iterations
        Ex. - Can you give me the code for the best performing {FOO_TARGET_FILENAME} player?
        Ex. - Create a detailed report on all the game outputs
        Ex. - How many average wins, victory points, and cities did the most recent {FOO_TARGET_FILENAME} player obtain?
        Ex. - Can you give me the code for the last successful {FOO_TARGET_FILENAME} player?

    {STRATEGIZER_NAME}: Strategizer has knowledge of the strategies you have attempted, and can generate new strategies by searching the web
        Ex. - What was the strategy of the best {FOO_TARGET_FILENAME} player?
        Ex. - Can you search the web for a single new strategy to implement?
        Ex. - What are 5 new strategy options that could give the current {FOO_TARGET_FILENAME} player a boost?
        Ex. - What are the previous strategies that I have attempted, and what are the results of each strategy?

    {RESEARCHER_NAME}: Researcher has access to the game files, and can perform web searches to find information
        Ex. - Can you find for me the different ActionTypes, and what I need to import to include them?
        Ex. - Can you give me the strategy that the opponent player is using?
        Ex. - What are the state functions that I can call to get information about the game state?

    {CODER_NAME}: Coder will only write the {FOO_TARGET_FILENAME} file. Afterwards the game is automatically run and the results are returned
        - Make Sure to Give Very Explicit Instructions to the coder (including all required code snippets)
        - Emphasize including print statements for debugging, and try/except blocks for error handling
        Ex. - Replace each 'action.type' call with the correct syntax of 'action.action_type'
        Ex. - Implement a a new function that will weight all the available actions. Follow this pseudocode .....
</AGENTS>

<Guidelines>
    - Make sure to be clear and concise in your message
    - Do not include vague messages to your agents, 
    - Always keep your GOALS in mind and try to achieve them
        - Medium Level Goal must have a clear objective for the the next **5** iterations of evolving
        - Low Level Goal must have a clear objective for the next iteration of evolving
    - Only include one agent key (the output is parsed to detemine which agent to send it to)
</Guidelines>

<Output Format>
    - META MEDIUM LEVEL GOAL: <insert here>
    - META LOW LEVEL GOAL: <insert here>
    - CHOSEN AGENT: {ANALYZER_NAME} / {STRATEGIZER_NAME} / {RESEARCHER_NAME} / {CODER_NAME} (choose one)
    - AGENT OBJECTIVE: <insert your objective message for the agent here>
</Output Format>

                """
            )
            
            msgs = state["meta_messages"][-MAX_MESSAGES_IN_AGENT:]
            tools = [think_tool]
            output = tool_calling_state_graph(sys_msg, msgs, tools)

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
    - The most up to date performance history, with the scores and game results of the {FOO_TARGET_FILENAME} player accross evolutions
    - The most recent foo_player.py file (note previous messages might be referring to an older version)
    - The most recent game_output.txt file which contains the output from run game command
    - The most recent game_results json file which contains the breakdown of the {FOO_TARGET_FILENAME} player vs. the opponent
        - Note: The game_results json file will not be included if the game failed to run due to a syntax error
    - Your OBJECTIVE: The most recent message includes the task that you are responding to... starts with {ANALYZER_NAME}
</Your Inputs>

<Your Role>
    - You are the Game ANALYZER Expert for Evolving the {FOO_TARGET_FILENAME} player
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the analyzer, you are the forefront for the game output for the foo_player.py
    - You are aware of the nuances of the game output, and how to interpret the results
    - You are in charge of storing all the knowledge that you have learned
    - You can open any file from the performace history using the read_local_file tool
    - Ensure output from the game_output.txt matches the {FOO_TARGET_FILENAME} player
</Your Role>

<Your Task>
    1. Digest the your past inquiries, the performance history, the game output, the game results, and your OBJECTIVE
    2. Use any additional tools required to get the information you need
    3. Respond to your OBJECTIVE message following your guidelines
</Your Task>

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
    - End your response with 'Let me know if you need anything else'
</Your Guidelines>

<Your Tools>
    - read_local_file: Read the content of a file that is in the catanatron files
        Input: String rel_path - path of the file to read from catanatron files or {FOO_TARGET_FILENAME}
        Output: String - content of the file                        
    - think_tool: Reflect on your current situation and plan your next steps
        Input: String reflection -Your detailed reflection on research progress, findings, gaps, and next steps
        Output: String - Confirmation that reflection was recorded for decision-making
</Your Tools>

YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
Make sure to start your output with 'ANALYSIS:' and end with 'END ANALYSIS'.
Respond with No Commentary, just the Analysis.
                """
            )
            
            tools = [read_local_file, think_tool]

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            game_output_msg = HumanMessage(content=f"This is the current game_output.txt file\n\n{read_game_output_file()}")
            game_results_msg = HumanMessage(content=f"This is the current game_results json file\n\n{read_game_results_file()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")


            # Call the LLM with the provided tools
            #base_len = len(state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:])
            msgs = state["analyzer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, game_output_msg, game_results_msg, current_foo_msg, state["recent_meta_message"]]
            output = tool_calling_state_graph(sys_msg, msgs, tools)
            
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
    - Your OBJECTIVE: The most recent message includes the task that you are responding to... starts with {STRATEGIZER_NAME}
</Your Inputs>

<Your Role>
    - You are the Strategy Expert for Evolving the {FOO_TARGET_FILENAME} player
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the strategizer, you are the forefront for improvement the foo_player.py
    - You are **Creative**, and are always looking for new strategies to implement
    - If you feel like the current strategy is not working, feel free to include it in your response
    - You are in charge of storing all the different attempts at strategies, and the results of each strategy
</Your Role>

<Your Task>
    1. Digest the current performance history, the current foo_player.py, the past messages, and your OBJECTIVE
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
Make sure to start your output with 'STRATEGY:' and end with 'END STRATEGY'.
Respond with No Commentary, just the Strategy.

                """
            )

            tools = [read_local_file, read_game_results_file, read_older_foo_file, web_search_tool_call, think_tool]
            
            # Call the LLM with the provided tools
            #base_len = len(state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:])

            performance_msg = HumanMessage(content=f"This is the current performance history\n\n{read_full_performance_history()}")
            current_foo_msg = HumanMessage(content=f"This is the current foo_player.py file\n\n{read_foo()}")

            msgs = state["strategizer_messages"][-MAX_MESSAGES_IN_AGENT:] + [performance_msg, current_foo_msg, state["recent_meta_message"]]
            output = tool_calling_state_graph(sys_msg, msgs, tools)
            
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
Make sure to start your output with 'RESEARCH:' and end with 'END RESEARCH'.
Respond with No Commentary, just the Research.


                """
            )

            tools = [read_local_file, web_search_tool_call, think_tool]
            
            catanatron_files_msg = HumanMessage(content=f"This is the list of catanatron files\n\n{list_catanatron_files()}")
            # Call the LLM with the provided tools (Add 1 because no need to summarize catanatron files)
            #base_len = len(state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:]) + 1
            msgs = state["researcher_messages"][-MAX_MESSAGES_IN_AGENT:] + [catanatron_files_msg, state["recent_meta_message"]]
            output = tool_calling_state_graph(sys_msg, msgs, tools)
            
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
</Your Inputs>

<Your Role>
    - You are the Coding Expert for Evolving the {FOO_TARGET_FILENAME} player
    - As an expert, you can always use the think_tool to reflect and plan your next steps
    - As the coder, you are the forefront for implementation for the foo_player.py
    - You are in charge of storing all the coding nuances that you have learned
</Your Role>

<Your Task>
    1. Digest your past inquiries, the meta messages, your current OBJEECTIVE, and the current {FOO_TARGET_FILENAME}
    2. Call the write_foo tool call to write the new code to the {FOO_TARGET_FILENAME} file
    3. Create a report with the changes you made to the code
</Your Task>

<Coding Guidelines>
    - Focus on making sure the code implementes the solution in the most correct way possible
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

Make sure to start your report with 'CODER' and end with 'END CODER'.

                """
            )
           
            tools = [write_foo, replace_code_in_foo, think_tool]
            
            # # Give Coder The Last Number of Meta Messages
            if len(state["meta_messages"]) > MAX_META_MESSAGES_GIVEN_TO_CODER:
                meta_msgs = state["meta_messages"][-MAX_META_MESSAGES_GIVEN_TO_CODER:]
            else:
                meta_msgs = state["meta_messages"]

            # Call the LLM with the provided tools
            current_foo_msg = HumanMessage(content=f"This is the old foo_player.py file\nNow It is your turn to update it with the new recommendations from META\n\n{read_foo()}")
            #base_len = len(state["coder_messages"][-MAX_MESSAGES_IN_AGENT:])
            msgs = state["coder_messages"][-MAX_MESSAGES_IN_AGENT:] + meta_msgs + [current_foo_msg]
            output = tool_calling_state_graph(sys_msg, msgs, tools)
            
            # Add to Meta Messages
            response = HumanMessage(content=output["messages"][-1].content)
            meta_messages = state["meta_messages"] + [response]

            #Add To Node Messages: Meta Human Request --> AI Response(content = tool_call_summary) + AI Response(content = final_message)
            #Only summarize new messages
            #tool_call_summary = summarize_messages(output["messages"][base_len:])
            coder_messages = state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            
            # Add to Coder Messages
            #coder_messages = state["coder_messages"] + [state["recent_meta_message"], AIMessage(content=response.content)]
            
            return {
                "recent_helper_response": response, 
                "tool_calling_messages": output["messages"], 
                "meta_messages": meta_messages, 
                "coder_messages": coder_messages,
            }
 
        def meta_choice(state: CreatorGraphState):
            """
            Conditional edge for Meta
            """
            print("In Conditional Edge Meta")
        
            if (CreatorAgent.current_evolution > CREATOR_NUM_EVOLUTIONS):
                lists = ["meta_messages", "analyzer_messages", "strategizer_messages", "researcher_messages", "coder_messages"]
                for msg_list in lists:
                    log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{self.llm_name}_{msg_list}.txt")
                    with open(log_path, "a") as log_file:
                        for m in state[msg_list]:
                            log_file.write(m.pretty_repr())


                return END

            meta_message = state["meta_messages"][-1].content

            for key in AGENT_KEYS:
                if key in meta_message:
                    print(f"Meta Message: {key} - going to {key}")
                    return key
                
                # Default case if neither string is found
            print(f"Warning: Could not determine desired agent in recent meta message. Defaulting to {ANALYZER_NAME}")
            return ANALYZER_NAME

        def construct_graph():
            graph = StateGraph(CreatorGraphState)
            graph.add_node("init", init_node)
            
            graph.add_node(ANALYZER_NAME, analyzer_node)
            graph.add_node(STRATEGIZER_NAME, strategizer_node)
            graph.add_node(RESEARCHER_NAME, researcher_node)
            graph.add_node(CODER_NAME, coder_node)
            graph.add_node("run_player", run_player_node)

            graph.add_node("meta", meta_node)

            graph.add_edge(START, "init")
            graph.add_edge("init", "run_player")
            graph.add_edge("run_player", ANALYZER_NAME)
            graph.add_conditional_edges(
                "meta", 
                meta_choice,
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
                            append_log_file(msg.pretty_repr())

                    # if "tool_calling_messages" in update:
                    #     count = 0
                    #     for msg in update["tool_calling_messages"]:
                    #         #print(msg)
                    #         #msg.pretty_print()
                    #         if isinstance(msg, ToolMessage):
                    #             print("Tool Message: ", msg.name)
                    #         count += 1
                    #         log_file.write((msg).pretty_repr())
                    #     print(f"Number of Tool Calling Messages: {count}")
                    
                    # if "evolve_counter" in update:
                    #     print("ENVOLVE COUNTER: ", update["evolve_counter"])
                    #     log_file.write(f"Evolve Counter: {update['evolve_counter']}\n")
                    # if "validator_counter" in update:
                    #     print("VALIDATOR COUNTER: ", update["validator_counter"])
                    #     log_file.write(f"Validator Counter: {update['validator_counter']}\n")


            print("✅  graph finished")

            # Copy Result File to the new directory
            dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")

            shutil.copy2(                           
                (FOO_TARGET_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / ("final" + dt + FOO_TARGET_FILENAME))
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
    
    # # Copy Result File to the new directory
    # dt = datetime.now().strftime("%Y%m%d_%H%M%S_")

    # shutil.copy2(                           
    #     (FOO_TARGET_FILE).resolve(),
    #     (Path(CreatorAgent.run_dir) / (dt + FOO_TARGET_FILENAME))
    # )

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
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    Input: short_game (bool): If True, run a short game with a 30 second timeout.
    """

    if short_game:
        run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_vg")
    else:
        run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_fg")
    game_run_dir = Path(CreatorAgent.run_dir) / run_id
    game_run_dir.mkdir(exist_ok=True)
    
    cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
    # Save the current prompt used for this game
    shutil.copy2(
        FOO_TARGET_FILE.resolve(),
        cur_foo_path
    )
        
    MAX_CHARS = 20_000                      

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
                timeout=14400,
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
        game_results = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()
    

    # Save the output to a log file in the game run directory
    output_file_path = game_run_dir / "game_output.txt"
    
    with open(output_file_path, "w") as output_file:
        output_file.write(game_results)


    # Search for the JSON results file path in the output
    json_path = None
    import re
    path_match = re.search(r'results_file_path:([^\s]+)', game_results)
    if path_match:
        # Extract the complete path
        json_path = path_match.group(1).strip()

    # Load in the most recent JSON File Game Rur
    json_content = {}
    json_copy_path = "None"
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
            our_player = "FooPlayer"
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
        
    # Update performance history only on long game runs
    if not short_game:
        # Create or update the entry for this evolution
        evolution_key = CreatorAgent.current_evolution
        CreatorAgent.current_evolution += 1
        
        # Convert paths to be relative to run_dir
        rel_output_file_path = output_file_path.relative_to(Path(CreatorAgent.run_dir))
        rel_cur_foo_path = cur_foo_path.relative_to(Path(CreatorAgent.run_dir))
        # Handle the case where json_copy_path is a string "None"
        rel_json_copy_path = "None"
        if json_copy_path != "None" and isinstance(json_copy_path, Path):
            rel_json_copy_path = json_copy_path.relative_to(Path(CreatorAgent.run_dir))
        
        performance_history[f"Evolution {evolution_key}"] = {
            "wins": wins,
            "avg_score": avg_score,
            "avg_turns": avg_turns,
            "full_game_log_path": str(rel_output_file_path),
            "json_game_results_path": str(rel_json_copy_path),
            "cur_foo_player_path": str(rel_cur_foo_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #"summary": "Not yet summarized"
        }
        
        # Write updated performance history
        with open(performance_history_path, 'w') as f:
            json.dump(performance_history, f, indent=2)
        
    if json_content:
        return json.dumps(json_content, indent=2)
    else:
        # If we didn't find a JSON file, return a limited version of the game output
        # MAX_CHARS = 5_000
        # stdout_limited = result.stdout[-MAX_CHARS:]
        # stderr_limited = result.stderr[-MAX_CHARS:]
        return game_results
    # Extract the score from the game results

    # Create a folder in the Creator.run_dir with EvolveCounter#_FooScore#

    # Inside the folder, 
    #   place the game_results.txt file with the game results
    #   copy the FOO_TARGET_FILE as foo_player.py

    # Add a file with the stdout and stderr called catanatron_output.txt
    # output_file_path = latest_run_folder / "catanatron_output.txt"
    # with open(output_file_path, "w") as output_file:
    #     output_file.write(game_results)
        
    #print(game_results)
        # limit the output to a certain number of characters

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