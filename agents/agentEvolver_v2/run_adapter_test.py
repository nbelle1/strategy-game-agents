import sys
import inspect
from pathlib import Path

# Add the parent and catanatron directories to the path to ensure imports work
# This mirrors the setup of your main agent
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "catanatron"))

import adapters
from catanatron.game import CatanatronGame
from catanatron.models.enums import Resource

def run_tests():
    """
    Dynamically finds and calls functions in adapters.py to check for runtime errors.
    """
    print("--- Starting Adapter Runtime Test ---")
    
    # 1. Create a dummy game instance to get a valid 'game_state' object
    try:
        # Using a minimal config to speed up initialization
        game = CatanatronGame.from_config(player_names=["p1", "p2"], vps_to_win=10, config_map="MINI")
        game_state = game.state
        print("Successfully created a dummy Catanatron game instance.")
    except Exception as e:
        print(f"FATAL: Could not initialize CatanatronGame. Error: {e}")
        sys.exit(1)

    # 2. Find all functions defined in the adapters module
    functions_to_test = [
        obj for name, obj in inspect.getmembers(adapters) 
        if inspect.isfunction(obj) and obj.__module__ == 'adapters'
    ]
    
    if not functions_to_test:
        print("No functions found in adapters.py to test. Exiting.")
        sys.exit(0)
        
    print(f"Found {len(functions_to_test)} functions to test in adapters.py")
    all_passed = True

    # 3. Loop through and try to call each function
    for func in functions_to_test:
        func_name = func.__name__
        sig = inspect.signature(func)
        
        # 4. Naively construct dummy arguments based on parameter names
        dummy_args = {}
        try:
            for param in sig.parameters.values():
                if 'state' in param.name:
                    dummy_args[param.name] = game_state
                elif 'player_id' in param.name:
                    dummy_args[param.name] = 0
                elif 'resource' in param.name:
                    dummy_args[param.name] = Resource.WOOD # Just pick one
                # Add other common dummy args as needed
                else:
                    # For any other parameter, pass None as a default guess
                    dummy_args[param.name] = None
            
            print(f"Testing {func_name} with args: { {k: type(v).__name__ for k, v in dummy_args.items()} }...")
            func(**dummy_args)
            print(f"  ✅ PASSED: {func_name}")

        except Exception as e:
            print(f"  ❌ FAILED: {func_name}")
            print(f"     Error Type: {type(e).__name__}")
            print(f"     Error: {e}")
            all_passed = False
            # We don't exit here, to see if other functions fail too
    
    if not all_passed:
        print("\n--- Some runtime tests failed. See errors above. ---")
        sys.exit(1) # Exit with a non-zero code to signal failure
    
    print("\n--- All adapter functions passed runtime checks! ---")
    sys.exit(0) # Success!

if __name__ == "__main__":
    run_tests()