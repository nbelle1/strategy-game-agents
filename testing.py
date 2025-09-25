import os
import subprocess
import shlex
import time
import re
from datetime import datetime

LLM = "GPTv2"  # The LLM used
EVAL_PLAYER = "BP"  # The player you want to evaluate (use the code/key)
NUM_GAMES = 1
NUM_REPETITIONS = 10

# Dictionary mapping player codes to their full names as they appear in summary
AGENTS = {
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
    "BP": "BestPlayer",
}

# List opponents from strongest to weakest
#OPPONENTS = ["AB", "F", "M", "W", "VP", "R"]
#OPPONENTS = ["VP", "R"]
OPPONENTS = ["AB"]

def run_game(eval_player, opponent, num_games):
    command = f"catanatron-play --players={opponent},{eval_player} --num={num_games} --output=data/ --json"
    print(f"Running: {command}")
    start = time.time()
    result = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=14400,
        check=False
    )
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.1f}s")
    return result.stdout + result.stderr


def parse_agent_wins(output, agent_code):
    """
    Extracts the number of wins for agent from the Player Summary section,
    regardless of color or extra info.
    
    Args:
        output: The complete output text from the game
        agent_code: The code for the agent (e.g., "LLM", "AB")
    
    Returns:
        Number of wins for the agent
    """
    agent_name = AGENTS.get(agent_code, agent_code)  # Get full name or use code if not found
    
    in_summary = False
    for line in output.splitlines():
        if "Player Summary" in line:
            in_summary = True
            print("Found Player Summary: " + line)
            continue
        if in_summary:
            # Look for the line that contains the agent name
            if agent_name in line:
                # Split on │ and get the WINS column (first after the player name)
                parts = [p.strip() for p in line.split("│")]
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
                # fallback: try to find the first integer after the player name
                for part in parts[1:]:
                    if part.isdigit():
                        return int(part)
                return 0
            # End of summary table (empty line or table border)
            if line.strip() == "" or line.strip().startswith("╵"):
                break
    return 0

def main():
    # Create a unique directory name based on current date/time
    timestamp = datetime.now().strftime(f"trial_%Y%m%d_%H%M%S_{NUM_REPETITIONS}Reps_{NUM_GAMES}Games")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the full name of the evaluation player
    eval_player_name = AGENTS.get(EVAL_PLAYER, EVAL_PLAYER)
    
    for repetition in range(NUM_REPETITIONS):
        print(f"Results will be saved to: {results_dir}")
        print(f"Testing {eval_player_name} ({EVAL_PLAYER}) against opponents...")

        # Track all results for summary
        summary_results = []

        for opponent in OPPONENTS:
            opponent_name = AGENTS.get(opponent, opponent)
            print(f"\n=== Evaluating {eval_player_name} vs {opponent_name} ===")
            output = run_game(EVAL_PLAYER, opponent, NUM_GAMES)

            # Save output to the timestamped directory
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = f"{LLM}_{EVAL_PLAYER}_vs_{opponent}_{timestamp}.txt"
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "w") as f:
                f.write(output)

            # Parse wins for both players
            eval_wins = parse_agent_wins(output, EVAL_PLAYER)
            opponent_wins = parse_agent_wins(output, opponent)
            
            print(f"{eval_player_name} wins: {eval_wins}/{NUM_GAMES}")
            print(f"{opponent_name} wins: {opponent_wins}/{NUM_GAMES}")
            
            # Store results for summary
            summary_results.append({
                "opponent": opponent,
                "opponent_name": opponent_name,
                "eval_wins": eval_wins,
                "opponent_wins": opponent_wins
            })

            if eval_wins > NUM_GAMES / 2:
                print(f"{eval_player_name} beats {opponent_name} ({eval_wins}/{NUM_GAMES}). Stopping evaluation.")
                break
            else:
                print(f"{eval_player_name} did not beat {opponent_name} ({eval_wins}/{NUM_GAMES}). Trying next opponent...")

    # Create a summary file
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Trial: {timestamp}\n")
        f.write(f"LLM: {LLM}\n")
        f.write(f"Player: {EVAL_PLAYER} ({eval_player_name})\n")
        f.write(f"Games per opponent: {NUM_GAMES}\n\n")
        f.write("Results:\n")
        
        for result in summary_results:
            f.write(f"- vs {result['opponent']} ({result['opponent_name']}): ")
            f.write(f"{result['eval_wins']}-{result['opponent_wins']}")
            if NUM_GAMES > 0:
                win_rate = result['eval_wins'] / NUM_GAMES * 100
                f.write(f" ({win_rate:.1f}% win rate)\n")
            else:
                f.write(" (no games completed)\n")

    print(f"\nEvaluation complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()