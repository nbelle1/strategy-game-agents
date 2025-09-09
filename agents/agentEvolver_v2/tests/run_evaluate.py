import os
import json
from pathlib import Path
import sys
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import _convert_to_message as dict_to_message


# --- Path Setup ---
# This script assumes it is located in agents/agentEvolver_v2/tests
# and that the current working directory is the root of the repository.
try:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    # Add project root to path to allow absolute imports
    sys.path.insert(0, str(project_root))
    # Verify CWD is project root
    if not (Path.cwd() / 'agents').is_dir() or not (Path.cwd() / 'catanatron').is_dir():
        print(f"Error: This script must be run from the repository root, but CWD is {Path.cwd()}")
        sys.exit(1)
    from agents.agentEvolver_v2.creator_agent import CreatorAgent, DEFAULT_ANALYZE_MSG, CreatorGraphState
    from agents.agentEvolver_v2.tests.evaluators import evaluate_analyzer_output, setup_analyzer_test_environment
except (ImportError, IndexError) as e:
    print(f"Error setting up path: {e}")
    print("Please ensure you are running this script from the root of the 'strategy-game-agents' repository.")
    sys.exit(1)


# --- LangSmith Dataset Setup ---

def create_langsmith_dataset(dataset_name: str):
    """Creates or clears a LangSmith dataset for the evaluation."""
    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' already exists. Deleting to ensure a clean run.")
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation for the Analyzer agent's ability to parse game logs.",
    )
    print(f"Dataset '{dataset_name}' created.")

    # The input for our target function is the initial state of the graph
    mock_input_state = CreatorGraphState(
        analyzer_messages=[],
        recent_meta_message=HumanMessage(content=DEFAULT_ANALYZE_MSG.format(FOO_TARGET_FILENAME='foo_player.py')),
        meta_messages=[],
        strategizer_messages=[],
        researcher_messages=[],
        coder_messages=[],
        recent_helper_response=AIMessage(content=''),
        game_results=HumanMessage(content=''),
        tool_calling_messages=[]
    )
    

    # The reference output for our evaluators
    reference_output = {"criteria": ["Defaulting to Random Action", "Choose action with score: 0", "Player FooPlayer had an error."]}

    client.create_example(
        inputs=mock_input_state,
        outputs=reference_output,
        dataset_id=dataset.id,
    )
    print("Example added to the dataset.")
    return dataset_name

# --- Target Function Definition ---

def target_func(inputs: CreatorGraphState):
    """
    The function to be evaluated. It runs an isolated instance of the agent.

    Each call to this function by the evaluator will:
    1. Instantiate a new CreatorAgent, which creates its own unique run directory.
    2. Set up the mock data files within that specific directory using the
       setup function associated with the evaluator.
    3. Run the analyzer_node on the provided inputs.

    This ensures each evaluation is hermetic and free from side effects.
    """
    agent = CreatorAgent()
    run_dir = Path(agent.run_dir)
    # Setup mock data in this specific agent's run_dir using the imported setup function
    setup_analyzer_test_environment(run_dir)

    # The input from the langsmith evaluator is a dict, we need to
    # reconstruct the message objects.
    reconstructed_inputs = inputs.copy()
    for key, value in reconstructed_inputs.items():
        if isinstance(value, dict) and "type" in value and "content" in value:
             reconstructed_inputs[key] = dict_to_message(value)
        elif isinstance(value, list):
            reconstructed_inputs[key] = [dict_to_message(m) if isinstance(m, dict) and "type" in m and "content" in m else m for m in value]

    return agent._analyzer_node(reconstructed_inputs)

# --- Main Execution Logic ---

def main():
    """Main function to configure and run the evaluation."""
    print("Starting Analyzer Agent Evaluation...")
    print("Please ensure your LANGSMITH_API_KEY environment variable is set.")

    if "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    dataset_name = "Analyzer_Agent_Evaluation_v1"
    create_langsmith_dataset(dataset_name)

    print("Running evaluation...")
    experiment_results = evaluate(
        target_func,
        data=dataset_name,
        evaluators=[evaluate_analyzer_output],
        experiment_prefix="Analyzer-Agent-Test",
        description="Testing the analyzer's ability to detect errors in game logs.",
    )

    print("\n--- Evaluation Complete ---")
    if hasattr(experiment_results, 'url'):
        print(f"View results in LangSmith: {experiment_results.url}")

    df = experiment_results.to_pandas()
    if not df.empty:
        print("\nResults Summary:")
        print(df[['feedback.analyzer_accuracy', 'error']].head())


if __name__ == "__main__":
    main()