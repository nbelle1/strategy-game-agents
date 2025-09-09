import json
from pathlib import Path

from pydantic import BaseModel, Field

# --- Test Environment Setup ---
# This section contains functions to set up specific test environments
# for different evaluators. Each setup function should create the mock
# artifacts needed for its corresponding evaluator to run.

def setup_analyzer_test_environment(run_dir: Path):
    """Creates mock data files needed for the analyzer agent to run."""
    # Create a dummy game run directory inside the main run_dir
    game_run_id = "game_eval_run_for_test"
    game_run_dir = run_dir / game_run_id
    game_run_dir.mkdir(exist_ok=True)

    # 1. Dummy game_output.txt for the history
    game_output_content = """GAME RESULTS:\nPlayer FooPlayer had an error.\nDefaulting to Random Action.\nChoose action with score: 0"""
    (game_run_dir / "game_output.txt").write_text(game_output_content)

    # 2. Dummy performance_history.json
    performance_history_content = {
        "Evolution 0": {
            "wins": 0,
            "avg_score": 0,
            "avg_turns": 0,
            "full_game_log_path": str((game_run_dir / "game_output.txt").relative_to(run_dir)),
            "json_game_results_path": "None",
            "cur_foo_player_path": "None",
            "timestamp": "2025-09-08 23:20:00"
        }
    }
    with open(run_dir / "performance_history.json", "w") as f:
        json.dump(performance_history_content, f, indent=2)

    return run_dir

# --- Evaluator Definitions ---
# This section contains the actual evaluator functions and their
# associated Pydantic models.

# This Pydantic model is defined for potential future use with LLM-based evaluators.
# The current implementation uses a simpler, direct string-matching function.
class Criteria(BaseModel):
    criteria_text: str = Field(description="The specific success criteria being evaluated.")
    reasoning: str = Field(description="Detailed explanation of why this criteria is or isn't captured.")
    is_captured: bool = Field(description="Whether this specific criteria is adequately captured.")

def evaluate_analyzer_output(run, example):
    """
    Evaluates the analyzer's output by checking for the presence of specific criteria strings.

    This is a simple, deterministic string-matching evaluator based on the
    prototype notebook. It checks if the agent's response contains predefined
    success criteria.
    """
    # The agent's final response is in the 'recent_helper_response' field of the output state
    prediction = run.outputs.get('recent_helper_response', '')
    if hasattr(prediction, 'content'):
        prediction = prediction.content

    # The ground truth criteria are in the 'outputs' of the dataset example
    success_criteria = example.outputs.get("criteria", [])

    if not success_criteria:
        return {"key": "analyzer_accuracy", "score": 0.0, "comment": "No success criteria found in example."}

    captured_count = 0
    for criterion in success_criteria:
        if criterion in prediction:
            captured_count += 1

    score = captured_count / len(success_criteria)

    return {"key": "analyzer_accuracy", "score": score}

