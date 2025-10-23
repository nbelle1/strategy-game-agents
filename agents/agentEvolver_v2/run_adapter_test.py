import sys
import inspect
from pathlib import Path

agent_dir = Path(__file__).parent  # .../agents/agentEvolver_v2
sys.path.insert(0, str(agent_dir))  # import adapters as top-level module

# Ensure catanatron importable (fallback to repo layout if needed)
try:
    from catanatron.game import Game
    from catanatron.models.player import Player, Color  # add for manual construction
except Exception:
    project_root = agent_dir.parent.parent  # .../strategy-game-agents
    catan_dir = project_root / "catanatron"
    sys.path.insert(0, str(catan_dir))
    from catanatron.game import Game  # type: ignore
    from catanatron.models.player import Player, Color  # type: ignore

import adapters  # must resolve to module "adapters"

SKIP_PATTERNS = ("_", "file", "path", "save", "load", "write", "read", "open", "http", "socket")

def _make_dummy_game():
    # Prefer adapter helper if present
    if hasattr(adapters, "create_game") and callable(getattr(adapters, "create_game")):
        try:
            return adapters.create_game(players=3, seed=42, vps_to_win=10, catan_map=None, initialize=True)
        except Exception:
            pass

    # Fallback 1: use Catanatron factory if available
    if hasattr(Game, "from_config"):
        # Try with player names
        try:
            return Game.from_config(player_names=["p1", "p2", "p3"], vps_to_win=10, config_map="MINI")
        except Exception:
            pass
        # Try with player colors
        try:
            return Game.from_config(player_colors=[Color.BLUE, Color.RED, Color.WHITE], vps_to_win=10, config_map="MINI")
        except Exception:
            pass

    # Fallback 2: construct Players and call Game(...) directly
    colors = [getattr(Color, name) for name in ("BLUE", "RED", "WHITE", "ORANGE") if hasattr(Color, name)]
    players = []
    for i, c in enumerate(colors[:3]):
        try:
            # Some versions accept (color, name) positionally
            players.append(Player(c, f"p{i+1}"))
        except TypeError:
            # Others accept only (color)
            players.append(Player(c))

    # Try common ctor signatures
    for kwargs in (
        dict(players=players, vps_to_win=10, config_map="MINI"),
        dict(players=players, vps_to_win=10),
        dict(players=players),
    ):
        try:
            return Game(**kwargs)
        except Exception:
            continue

    raise RuntimeError("Unable to construct Game with any known pattern (from_config or direct players list).")

def _first_playable_action(game):
    # Try using adapter helpers if available
    try:
        if hasattr(adapters, "playable_actions"):
            try:
                acts = adapters.playable_actions(game)  # game-first convention
            except TypeError:
                acts = adapters.playable_actions(game.state)  # some APIs take state
            return acts[0] if acts else None
    except Exception:
        pass
    return None

def _type_name(x):
    try:
        return type(x).__name__
    except Exception:
        return "?"

def _build_args(func, game, game_state, sample_action):
    sig = inspect.signature(func)
    ann = func.__annotations__ if hasattr(func, "__annotations__") else {}
    AColor = getattr(adapters, "Color", None)
    AActionType = getattr(adapters, "ActionType", None)
    DEFAULT_WEIGHTS = getattr(adapters, "DEFAULT_WEIGHTS", None)
    copy_game = getattr(adapters, "copy_game", None)

    args = {}
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect._empty:
            # Prefer using defaults where provided
            continue

        name = p.name.lower()
        pann = p.annotation

        # Annotation-based hints
        if pann is Game or (isinstance(pann, str) and pann.endswith("Game")):
            args[p.name] = copy_game(game) if callable(copy_game) else game
            continue

        # Name-based heuristics
        if name in ("game", "g") or name.endswith("_game"):
            args[p.name] = copy_game(game) if callable(copy_game) else game
        elif "state" in name:
            args[p.name] = game_state
        elif name in ("player_id", "playerindex", "player_idx", "idx"):
            args[p.name] = 0
        elif "color" in name and AColor is not None:
            args[p.name] = AColor.BLUE
        elif "action_type" in name and AActionType is not None:
            args[p.name] = AActionType.END_TURN
        elif name == "action":
            args[p.name] = sample_action
        elif name == "validate":
            args[p.name] = False
        elif "seed" in name:
            args[p.name] = 42
        elif "vps" in name:
            args[p.name] = 10
        elif "weights" in name and DEFAULT_WEIGHTS is not None:
            args[p.name] = DEFAULT_WEIGHTS
        elif name.endswith("_fn") or name.endswith("_func") or name.endswith("callback"):
            args[p.name] = (lambda *a, **k: None)
        else:
            args[p.name] = None
    return args

def run_tests():
    print("--- Starting Adapter Runtime Test ---")

    # Create a valid game and state
    try:
        game = _make_dummy_game()
        game_state = game.state
        print("Game initialized.")
    except Exception as e:
        print(f"FATAL: Could not initialize Game. Error: {e}")
        sys.exit(1)

    # Optional: get one legal action
    sample_action = _first_playable_action(game)

    # Collect only functions defined in adapters.py (not re-exports)
    functions = [
        obj for _, obj in inspect.getmembers(adapters)
        if inspect.isfunction(obj) and obj.__module__ == "adapters"
    ]
    # Skip risky/private utilities
    functions = [
        f for f in functions
        if not any(tok in f.__name__.lower() for tok in SKIP_PATTERNS)
    ]

    if not functions:
        print("No functions defined in adapters.py to test. Exiting.")
        sys.exit(0)

    print(f"Found {len(functions)} functions to test in adapters.py")
    all_passed = True

    for func in functions:
        args = _build_args(func, game, game_state, sample_action)
        try:
            arg_types = ", ".join(f"{k}: {_type_name(v)}" for k, v in args.items())
            print(f"Testing {func.__name__}({arg_types})")
            func(**args)
            print(f"  ✅ PASSED: {func.__name__}")
        except Exception as e:
            all_passed = False
            print(f"  ❌ FAILED: {func.__name__}")
            print(f"     {type(e).__name__}: {e}")

    if not all_passed:
        print("\n--- Some runtime tests failed. See errors above. ---")
        sys.exit(1)

    print("\n--- All adapter functions passed runtime checks! ---")
    sys.exit(0)

if __name__ == "__main__":
    run_tests()