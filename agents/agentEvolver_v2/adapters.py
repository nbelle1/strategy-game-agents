"""Unified adapter for Catanatron agents.

Expose a small, stable surface for multi-agent systems to:
- Inspect game state
- Enumerate legal actions
- Execute hypothetical moves (with/without validation)
- Expand chance outcomes (dice, dev cards, robber)
- Use pruning helpers
- Build/evaluate heuristics

Everything here is a thin re-export or trivial wrapper from catanatron & friends.
"""
### KEEP THESE IMPORTS BELOW THIS LINE ###
from catanatron.game import Game  # has .state, .copy(), .execute(), .winning_color()
from catanatron.models.player import Player, Color
### KEEP THESE IMPORTS ABOVE THIS LINE ###

from catanatron.models.actions import Action
from catanatron.models.enums import Resource
from typing import Union, List, Dict, Any, Optional
from collections import Counter as CounterType
from enum import Enum as _PyEnum


def _resolve_player(game: Game, player: Union[Player, Color, str, None]) -> Player:
    """Resolve a player specifier to a Player instance from the game.

    Args:
        game: Game instance with a .state.players sequence.
        player: Player instance, Color enum, player name (str), or None.

    Returns:
        The resolved Player object.

    Raises:
        ValueError: if the player could not be resolved.
        TypeError: if the game does not expose player information.
    """
    # Obtain players list
    players = None
    if hasattr(game, "state") and hasattr(game.state, "players"):
        players = getattr(game.state, "players")
    elif hasattr(game, "players"):
        players = getattr(game, "players")
    else:
        raise TypeError("Provided game object does not expose players (game.state.players)")

    if player is None:
        # Default to current player if available
        current = getattr(game.state, "current_player", None)
        if current is None:
            raise ValueError("No player specified and game has no current_player")
        return current

    # If already a Player instance
    if isinstance(player, Player):
        return player

    # If a Color enum, match by player.color
    if isinstance(player, Color):
        for p in players:
            if getattr(p, "color", None) == player:
                return p
        raise ValueError(f"No player with color {player} found in game")

    # If a string, match by name or color name (case-insensitive)
    if isinstance(player, str):
        key = player.strip().lower()
        for p in players:
            name = getattr(p, "name", None)
            color = getattr(p, "color", None)
            color_name = None
            if color is not None:
                # Color enum -> name attribute or str()
                color_name = getattr(color, "name", None) or str(color)
            if (isinstance(name, str) and name.lower() == key) or (
                isinstance(color_name, str) and color_name.lower() == key
            ):
                return p
        raise ValueError(f"No player with name or color '{player}' found in game")

    raise TypeError("player must be a Player, Color, str, or None")


def get_player_resources(
    game: Game,
    player: Union[Player, Color, str, None] = None,
    serialize: bool = True,
) -> Union[Dict[str, int], 'CounterType[Resource]']:
    """Return the resource counts for the specified player.

    Args:
        game: The Catanatron Game instance to query. Must expose player resources.
        player: The player to query. If None, defaults to the current player.
                Can be a Player object, Color enum, or player name/color string.
        serialize: If True, returns resources as a dictionary mapping resource names (str) to counts (int).
                   If False, returns the raw Counter[Resource] object.

    Returns:
        If `serialize` is True, returns a dictionary with resource names (e.g., "brick", "wood") as keys and integer counts as values.
        If `serialize` is False, returns the raw Counter[Resource] object.

    Raises:
        ValueError: If the player cannot be resolved (e.g., invalid name/color or no current player).
        TypeError: If the game does not expose player resources or the player lacks a resources attribute.

    Notes:
        - Defaults to the game's current player if `player` is None.
        - Resource names are derived from the Resource enum (e.g., Resource.BRICK -> "brick").
        - If the player's `to_dict()` method is available, it is used to extract serialized resources.
        - Falls back to direct access of `player.resources` if `to_dict()` is unavailable or fails.
        - Defensive against missing attributes and non-standard resource containers.

    Examples:
        ```python
        # Get serialized resources for the current player
        resources = get_player_resources(game)
        # Example output: {"brick": 2, "wood": 1, "sheep": 0}

        # Get raw Counter[Resource] for a specific player
        raw_resources = get_player_resources(game, player="Alice", serialize=False)
        # Example output: Counter({Resource.BRICK: 2, Resource.WOOD: 1})
        ```
    """
    # Resolve the Player instance (may raise ValueError/TypeError)
    player_obj: Optional[Player] = None

    if player is None:
        # Prefer module helper get_current_player if available to centralize resolution logic
        gcp = globals().get("get_current_player")
        if callable(gcp):
            try:
                # Ask for the raw Player object
                player_obj = gcp(game, serialize=False)  # type: ignore[arg-type]
            except Exception:
                # Fall back to basic resolver
                player_obj = _resolve_player(game, None)
        else:
            player_obj = _resolve_player(game, None)
    else:
        # Resolve provided specifier (Player|Color|str)
        player_obj = _resolve_player(game, player)

    # Ensure the player was resolved
    if player_obj is None:
        raise ValueError("Player could not be resolved from provided specifier")

    # Ensure the player has a resources attribute
    if not hasattr(player_obj, "resources"):
        raise TypeError("Resolved player does not expose a 'resources' attribute")

    resources = getattr(player_obj, "resources")

    # If caller wants the raw resources container, return it directly
    if not serialize:
        return resources

    # If player defines to_dict(), prefer its serialized resources mapping
    if hasattr(player_obj, "to_dict") and callable(getattr(player_obj, "to_dict")):
        try:
            pd = player_obj.to_dict()
            if isinstance(pd, dict) and "resources" in pd:
                # Normalize keys to lowercase strings and ensure int counts
                out: Dict[str, int] = {}
                res_map = pd.get("resources") or {}
                if isinstance(res_map, dict):
                    for k, v in res_map.items():
                        if k is None:
                            continue
                        out[str(k).lower()] = int(v) if v is not None else 0
                    return out
                # If resources not a dict, fall back to generic handling below
        except Exception:
            # Fall through to direct resources inspection
            pass

    # Defensive conversion from underlying resources container to dict[str,int]
    out: Dict[str, int] = {}
    # If resources behaves like a mapping (has items()), iterate through it
    if hasattr(resources, "items"):
        try:
            for k, v in resources.items():
                # Convert Resource enum keys to their name, otherwise string-ify
                if hasattr(k, "name"):
                    key = getattr(k, "name")
                else:
                    key = str(k)
                out[key.lower()] = int(v) if v is not None else 0
            return out
        except Exception:
            # Fall back to attempting to convert to dict
            pass

    # If resources is iterable of pairs
    try:
        for k, v in list(resources):
            if hasattr(k, "name"):
                key = getattr(k, "name")
            else:
                key = str(k)
            out[key.lower()] = int(v) if v is not None else 0
        return out
    except Exception:
        # As a last resort, try to stringify the entire resources object
        raise TypeError("Unable to serialize player resources; unexpected resources container type")


# --- Serialization / deserialization helpers for actions/params ---

def _serialize_param(value: Any) -> Any:
    """Recursively convert a parameter value to JSON-safe primitives.

    Rules:
      - Primitives (str, int, float, bool, None) are returned as-is.
      - Player objects -> player.name
      - Enums -> enum.name (Resource enums lower-cased by callers where needed)
      - Lists/tuples/sets -> list of serialized elements
      - Dicts -> dict with serialized values
      - Other objects (Vertex/Edge/etc) -> str(obj)
    """
    # Primitives
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Player objects
    if isinstance(value, Player):
        return getattr(value, "name", str(value))
    # Enums
    if isinstance(value, _PyEnum):
        # For generic enums, return their name; callers may lower-case Resource names
        return value.name
    # Mappings
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            # Normalize key to string
            key = k.name.lower() if hasattr(k, "name") and isinstance(k, Resource) else str(k)
            out[key] = _serialize_param(v)
        return out
    # Iterables
    if isinstance(value, (list, tuple, set)):
        return [_serialize_param(v) for v in value]
    # Fallback: stringify (used for Vertex/Edge objects)
    try:
        return str(value)
    except Exception:
        return repr(value)


def _find_board_object_by_str(board: Any, s: str) -> Any:
    """Search board containers for an object whose str() equals s.

    Looks through board.tiles, board.vertices, and board.edges keys and returns the
    matching key object (not the value). Returns None if not found.
    """
    if board is None:
        return None
    # Tiles
    tiles = getattr(board, "tiles", None)
    if tiles is not None:
        try:
            for k in tiles.keys():
                if str(k) == s:
                    return k
        except Exception:
            pass
    # Vertices
    vertices = getattr(board, "vertices", None)
    if vertices is not None:
        try:
            for k in vertices.keys():
                if str(k) == s:
                    return k
        except Exception:
            pass
    # Edges
    edges = getattr(board, "edges", None)
    if edges is not None:
        try:
            for k in edges.keys():
                if str(k) == s:
                    return k
        except Exception:
            pass
    return None


def _deserialize_param(value: Any, board: Any = None, players: Optional[List[Player]] = None) -> Any:
    """Recursively attempt to convert serialized param values back to engine objects.

    Best-effort conversions:
      - Strings: try to resolve to a Player by name (if players provided), then to a board object via _find_board_object_by_str.
      - Lists/tuples/dicts: recurse.
      - Others: return as-is.

    This is intentionally conservative: if no mapping is found, original value is returned.
    """
    # Primitives
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        # Try player resolution first
        if players is not None:
            key = value.strip()
            for p in players:
                name = getattr(p, "name", None)
                if isinstance(name, str) and name == key:
                    return p
        # Try board objects (locations)
        if board is not None:
            found = _find_board_object_by_str(board, value)
            if found is not None:
                return found
        # Nothing found: keep string
        return value
    # Mappings
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            new_k = k
            # Normalize resource keys back to Resource enums if possible is intentionally omitted to avoid tight coupling.
            out[new_k] = _deserialize_param(v, board=board, players=players)
        return out
    # Iterables
    if isinstance(value, (list, tuple, set)):
        return [_deserialize_param(v, board=board, players=players) for v in value]
    # Other types: return as-is
    return value


def get_legal_moves(
    game: 'Game',
    player: Optional[Union['Player', str]] = None,
    serialize: bool = True,
) -> List[Union['Action', Dict[str, Any]]]:
    """Return all legal moves available to the specified player in the current game state.

    Args:
        game: The Catanatron Game instance to query. Must implement `get_legal_actions(player)`.
        player: The player to query moves for. If None, defaults to the current player.
                Can be a Player object or a player name string (e.g., "Alice").
        serialize: If True, returns moves as a list of JSON-safe dictionaries with normalized parameters.
                   If False, returns the raw Action objects.

    Returns:
        If `serialize` is True, returns a list of dictionaries where each dictionary represents an Action and contains:
            - "action_type": The name of the action type (string).
            - "player": The player's name (string).
            - "params": A dictionary of parameters where all engine objects are converted to strings or primitives:
                - Vertex/Edge/Tile objects are converted to their string representation (e.g., "Vertex(0, 0)").
                - Player objects are converted to their name (e.g., "Alice").
                - Resource enums are converted to their lowercase name (e.g., "brick").
                - Collections (lists, tuples, dicts) are recursively normalized.
                - Special param keys (e.g., "location", "resource", "offer", "request") are handled consistently.

        If `serialize` is False, returns a list of raw Action objects.

    Raises:
        TypeError: If the game does not implement `get_legal_actions(player)`.
        ValueError: If the player string cannot be resolved to a Player object.

    Notes:
        - Player resolution:
            - If `player` is None, defaults to the current player (via `game.state.current_player`).
            - If `player` is a string, resolves to a Player object by matching `player.name`.
        - Serialization rules for `params`:
            - Vertex/Edge/Tile objects are converted to their string representation.
            - Player objects are converted to their name.
            - Resource enums are converted to their lowercase name.
            - Collections are recursively normalized.
        - Common param keys and their serialized forms:
            - "location": String representation of Vertex/Edge (e.g., "Vertex(0, 0)").
            - "resource": Lowercase resource name (e.g., "brick").
            - "resources": Dictionary mapping resource names to counts (e.g., {"brick": 2, "wood": 1}).
            - "offer"/"request": Dictionary mapping resource names to counts.
        - Round-tripping:
            - Use `find_board_object(game, str)` to convert stringified locations back to Vertex/Edge objects.
            - Use `Action.from_dict(action_dict, players_list)` to reconstruct Action objects from serialized dicts.
            - Use `apply_move(game, action_dict)` to apply a serialized action.

    Examples:
        # Get serialized moves for the current player
        moves = get_legal_moves(game)
        # Example output:
        # [
        #     {
        #         "action_type": "build_road",
        #         "player": "Alice",
        #         "params": {
        #             "location": "Edge(Vertex(0, 0), Vertex(0, 1))",
        #             "resource": "brick"
        #         }
        #     },
        #     {
        #         "action_type": "trade",
        #         "player": "Alice",
        #         "params": {
        #             "offer": {"brick": 1},
        #             "request": {"wood": 1}
        #         }
        #     }
        # ]

        # Get raw Action objects for a specific player
        actions = get_legal_moves(game, player="Alice", serialize=False)
        # Example output: [Action(...), Action(...)]
    """
    # Basic validation: ensure underlying API exists
    if not hasattr(game, "get_legal_actions") or not callable(getattr(game, "get_legal_actions")):
        raise TypeError("Game does not implement get_legal_actions(player)")

    # Resolve player according to spec: None -> current player; str -> match by name
    resolved_player = None
    if player is None:
        state = getattr(game, "state", None)
        resolved_player = getattr(state, "current_player", None) if state is not None else None
        if resolved_player is None:
            raise ValueError("No current player found on game.state.current_player")
    elif isinstance(player, str):
        # Search players list for matching name
        players = getattr(getattr(game, "state", None), "players", None) or getattr(game, "players", None)
        if players is None:
            raise ValueError(f"Player {player} not found: game has no players list")
        found = None
        for p in players:
            name = getattr(p, "name", None)
            if isinstance(name, str) and name == player:
                found = p
                break
        if found is None:
            raise ValueError(f"Player {player} not found")
        resolved_player = found
    else:
        # Assume Player-like object
        resolved_player = player  # type: ignore[assignment]

    # Obtain legal actions
    actions = game.get_legal_actions(resolved_player)
    if actions is None:
        return []
    try:
        action_list = list(actions)
    except TypeError:
        raise TypeError("get_legal_actions did not return an iterable of actions")

    if not serialize:
        return action_list

    # Recursive normalizer for params
    def _normalize(val: Any) -> Any:
        # Primitives pass-through
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        # Player objects -> name
        if isinstance(val, Player):
            return getattr(val, "name", str(val))
        # Enums: Resources lower-case, others use .name
        if isinstance(val, _PyEnum):
            # Prefer Resource lower-case
            if isinstance(val, Resource):
                return val.name.lower()
            return val.name
        # Dict-like: normalize keys and values
        if isinstance(val, dict):
            out: Dict[str, Any] = {}
            for k, v in val.items():
                # Normalize key: Resource enum -> lower-case name; else str(k)
                if hasattr(k, "name"):
                    key_name = getattr(k, "name")
                    if isinstance(k, Resource):
                        key_name = key_name.lower()
                else:
                    key_name = str(k)
                # For common mappings like resources/offer/request we ensure int counts
                if isinstance(v, (int, float)):
                    out[key_name] = int(v)
                else:
                    out[key_name] = _normalize(v)
            return out
        # Iterable: list/tuple/set
        if isinstance(val, (list, tuple, set)):
            return [_normalize(v) for v in val]
        # Fallback: stringify engine objects (Vertex/Edge/Tile)
        try:
            return str(val)
        except Exception:
            return repr(val)

    serialized: List[Dict[str, Any]] = []
    for a in action_list:
        if a is None:
            continue
        # Use to_dict if provided for base shape
        if hasattr(a, "to_dict") and callable(getattr(a, "to_dict")):
            try:
                base = a.to_dict()
            except Exception:
                base = {
                    "action_type": getattr(getattr(a, "action_type", None), "name", str(getattr(a, "action_type", None))),
                    "player": getattr(getattr(a, "player", None), "name", str(getattr(a, "player", None))),
                    "params": getattr(a, "params", {}) or {},
                }
        else:
            base = {
                "action_type": getattr(getattr(a, "action_type", None), "name", str(getattr(a, "action_type", None))),
                "player": getattr(getattr(a, "player", None), "name", str(getattr(a, "player", None))),
                "params": getattr(a, "params", {}) or {},
            }
        params = base.get("params", {}) or {}
        # Special-case top-level known keys for clearer serialization
        normalized_params: Dict[str, Any] = {}
        if isinstance(params, dict):
            for k, v in params.items():
                k_str = str(k)
                if k_str in ("location", "location1", "location2", "target"):
                    # locations: stringify or normalize
                    normalized_params[k_str] = _normalize(v)
                elif k_str == "resource":
                    # single resource: enum or string -> lower-case name
                    if isinstance(v, _PyEnum):
                        if isinstance(v, Resource):
                            normalized_params[k_str] = v.name.lower()
                        else:
                            normalized_params[k_str] = v.name
                    elif isinstance(v, str):
                        normalized_params[k_str] = v.lower()
                    else:
                        normalized_params[k_str] = _normalize(v)
                elif k_str in ("resources", "offer", "request"):
                    # Expect mapping of resources to counts
                    if isinstance(v, dict):
                        cleaned: Dict[str, int] = {}
                        for rk, rv in v.items():
                            if hasattr(rk, "name"):
                                rk_name = getattr(rk, "name")
                                if isinstance(rk, Resource):
                                    rk_name = rk_name.lower()
                            else:
                                rk_name = str(rk)
                            try:
                                cleaned[rk_name] = int(rv)
                            except Exception:
                                # Fallback: attempt normalization then int
                                try:
                                    cleaned[rk_name] = int(_normalize(rv))
                                except Exception:
                                    cleaned[rk_name] = rv
                        normalized_params[k_str] = cleaned
                    else:
                        normalized_params[k_str] = _normalize(v)
                else:
                    # Generic normalization for other keys
                    normalized_params[str(k)] = _normalize(v)
        else:
            # params not a dict - normalize wholesale
            normalized_params = _normalize(params) if params is not None else {}
        base["params"] = normalized_params
        serialized.append(base)

    return serialized


def apply_move(
    game: 'Game',
    action: Union['Action', Dict[str, Any]],
    player: Optional[Union['Player', str]] = None,
) -> None:
    """Apply a move to the game.

    This helper accepts either a raw Action object or a serialized action dictionary (as produced by
    `get_legal_moves(..., serialize=True)`) and applies it to the given game. It performs best-effort
    deserialization of parameters (locations, player names) using `find_board_object` and the game's
    player list, then delegates to the game's `apply_action(Action)` method which performs validation.

    Args:
        game: Game instance with `apply_action(Action)`.
        action: Action object or serialized action dict.
        player: Optional player override (Player object or player name). If provided, this will replace
                the action's player before applying.

    Raises:
        TypeError: If required APIs are missing (game.apply_action or players list for deserialization).
        ValueError: If deserialization fails or the engine rejects the action (propagated from engine).
    """
    # Ensure the engine exposes apply_action
    if not hasattr(game, "apply_action") or not callable(getattr(game, "apply_action")):
        raise TypeError("Game does not implement apply_action(action)")

    # Determine players list (needed when deserializing from dict)
    players_list = getattr(getattr(game, "state", None), "players", None) or getattr(game, "players", None)

    # Determine board for locating Vertex/Edge objects
    board = getattr(getattr(game, "state", None), "board", None) or getattr(game, "board", None)

    action_obj: Optional[Action] = None

    if isinstance(action, dict):
        if players_list is None:
            raise TypeError("Deserializing action dict requires access to game.state.players or game.players")
        # First, construct an Action via Action.from_dict if available
        if hasattr(Action, "from_dict") and callable(getattr(Action, "from_dict")):
            try:
                action_obj = Action.from_dict(action, players_list)
            except Exception as e:
                raise ValueError(f"Failed to deserialize Action from dict: {e}")
        else:
            raise TypeError("Action.from_dict is required to deserialize action dicts")
        # Best-effort: convert stringified params back to engine objects
        try:
            raw_params = getattr(action_obj, "params", {}) or {}
            deserialized = _deserialize_param(raw_params, board=board, players=players_list)
            action_obj.params = deserialized
        except Exception:
            # Leave params as-is; engine will validate when applying
            pass
    elif isinstance(action, Action):
        action_obj = action
    else:
        raise TypeError("action must be an Action instance or a serialized dict")

    # If a player override is provided, resolve and set it
    if player is not None:
        resolved = _resolve_player(game, player)
        action_obj.player = resolved

    # Final check
    if not isinstance(action_obj, Action):
        raise TypeError("Deserialized or provided action is not an Action instance")

    # Apply — engine will raise on invalid actions
    game.apply_action(action_obj)
    return None


# --- Board inspection helpers ---

def get_board_state(
    game: 'Game',
    serialize: bool = True,
) -> 'Union[Dict[str, Any], "Board"]':
    """Return the game's board state.

    Args:
        game: The Catanatron Game instance to inspect. Must expose a Board via `game.state.board`
              or `game.board`.
        serialize: If True (default), return a JSON-serializable dictionary describing tiles,
                   vertices, edges, ports, and robber position. If False, return the raw Board object.

    Returns:
        If `serialize` is True, returns a dictionary with the following top-level keys:
            - "tiles": mapping of stringified Vertex -> {"tile_type": <str>, "number_token": <int|None>, "robber": <bool>}
            - "vertices": mapping of stringified Vertex -> player_name_or_None
            - "edges": mapping of stringified Edge -> player_name_or_None
            - "ports": mapping of stringified Vertex -> port_type_name
            - "robber_position": stringified Vertex
        If `serialize` is False, returns the Board object instance held by the game.

    Raises:
        TypeError: If the game does not expose a board (no `game.state.board` and no `game.board`).
        Exception: Any unexpected exceptions from board.to_dict() are allowed to propagate.

    Notes:
        - Prefer calling `board.to_dict()` if available (it yields the exact string keys used elsewhere).
        - If `board.to_dict()` is not available, the adapter should build an equivalent dictionary by iterating
          `board.tiles`, `board.vertices`, `board.edges`, `board.ports`, and reading `board.robber_position`.
        - The serialized form intentionally uses str() representations of Vertex/Edge (e.g., "Vertex(0, 0)", "Edge(Vertex(...), Vertex(...))")
          because those are the canonical keys used across adapters for lookup and round-trip operations.

    Examples:
        # Get a JSON-friendly snapshot of the board
        snapshot = get_board_state(game)

        # Work with the raw Board object for low-level operations
        board = get_board_state(game, serialize=False)
    """
    # Resolve the board from game.state.board or game.board
    state = getattr(game, "state", None)
    board = None
    if state is not None and hasattr(state, "board"):
        board = getattr(state, "board")
    elif hasattr(game, "board"):
        board = getattr(game, "board")

    if board is None:
        raise TypeError("Game does not expose a board via game.state.board or game.board")

    # If caller wants the raw Board, return it directly
    if not serialize:
        return board

    # Prefer board.to_dict() when available for canonical serialization
    if hasattr(board, "to_dict") and callable(getattr(board, "to_dict")):
        # Let any exceptions from to_dict propagate to caller
        return board.to_dict()

    # Defensive manual serialization if to_dict() is absent
    result: Dict[str, Any] = {}

    # Tiles
    tiles = getattr(board, "tiles", {}) or {}
    tiles_out: Dict[str, Dict[str, Any]] = {}
    try:
        for vertex_key, tile in tiles.items():
            key_str = str(vertex_key)
            tile_type = getattr(tile, "tile_type", None)
            tt = getattr(tile_type, "name", tile_type) if tile_type is not None else None
            number = getattr(tile, "number_token", None)
            robber_flag = getattr(tile, "robber", False)
            tiles_out[key_str] = {"tile_type": tt, "number_token": number, "robber": bool(robber_flag)}
    except Exception:
        # If iteration fails, leave tiles_out empty
        tiles_out = {}
    result["tiles"] = tiles_out

    # Vertices
    vertices = getattr(board, "vertices", {}) or {}
    verts_out: Dict[str, Optional[str]] = {}
    try:
        for v_key, owner in vertices.items():
            key_str = str(v_key)
            if owner is None:
                verts_out[key_str] = None
            else:
                verts_out[key_str] = getattr(owner, "name", str(owner))
    except Exception:
        verts_out = {}
    result["vertices"] = verts_out

    # Edges
    edges = getattr(board, "edges", {}) or {}
    edges_out: Dict[str, Optional[str]] = {}
    try:
        for e_key, owner in edges.items():
            key_str = str(e_key)
            if owner is None:
                edges_out[key_str] = None
            else:
                edges_out[key_str] = getattr(owner, "name", str(owner))
    except Exception:
        edges_out = {}
    result["edges"] = edges_out

    # Ports
    ports = getattr(board, "ports", {}) or {}
    ports_out: Dict[str, Any] = {}
    try:
        for p_key, port_type in ports.items():
            key_str = str(p_key)
            pt = getattr(port_type, "name", port_type) if port_type is not None else None
            ports_out[key_str] = pt
    except Exception:
        ports_out = {}
    result["ports"] = ports_out

    # Robber position
    robber_pos = getattr(board, "robber_position", None)
    result["robber_position"] = str(robber_pos) if robber_pos is not None else None

    return result


def find_board_object(
    game: 'Game',
    key: str,
) -> 'Optional[Union["Vertex", "Edge", "Tile"]]':
    """Find a board object (Vertex, Edge, or Tile) by its string key.

    Args:
        game: The Catanatron Game instance containing the board (accessible via `game.state.board` or `game.board`).
        key: The string key to look up (produced by Board.to_dict() or by str(Vertex)/str(Edge)).

    Returns:
        The matching object from the Board (Vertex, Edge, or Tile) if found; otherwise None.

    Behavior / Lookup order:
        1. Resolve the Board: try `getattr(game, "state", None).board`, then `getattr(game, "board", None)`.
           If no board is found, raise TypeError.
        2. Search `board.vertices` keys: for each vertex `v`, if `str(v) == key` return `v`.
        3. Search `board.edges` keys: for each edge `e`, if `str(e) == key` return `e`.
        4. Search `board.tiles` keys: for each tile_key `t` (typically a Vertex key representing tile center), if `str(t) == key` return `t`.
        5. If no match is found, return None (do not raise). Let callers decide how to handle misses.

    Notes:
        - This is a best-effort, deterministic string-based lookup intended to support round-tripping of serialized
          actions and board snapshots. It relies on the repository's Vertex.__str__ and Edge.__str__ formats.
        - It does not attempt fuzzy matching; key must exactly equal str(object).

    Examples:
        v = find_board_object(game, "Vertex(0, 0)")
        e = find_board_object(game, "Edge(Vertex(0, 0), Vertex(0, 1))")
        missing = find_board_object(game, "Vertex(99, 99)")  # returns None
    """
    # Resolve board
    state = getattr(game, "state", None)
    board = None
    if state is not None and hasattr(state, "board"):
        board = getattr(state, "board")
    elif hasattr(game, "board"):
        board = getattr(game, "board")

    if board is None:
        raise TypeError("Game does not expose a board via game.state.board or game.board")

    # 2. Search vertices
    vertices = getattr(board, "vertices", None)
    if vertices is not None:
        try:
            for v in vertices.keys():
                if str(v) == key:
                    return v
        except Exception:
            pass

    # 3. Search edges
    edges = getattr(board, "edges", None)
    if edges is not None:
        try:
            for e in edges.keys():
                if str(e) == key:
                    return e
        except Exception:
            pass

    # 4. Search tiles
    tiles = getattr(board, "tiles", None)
    if tiles is not None:
        try:
            for t in tiles.keys():
                if str(t) == key:
                    return t
        except Exception:
            pass

    # Not found
    return None


# The rest of the module (serialization helpers, get_board_state, get_game_state, get_current_player)
# were implemented earlier in this file and rely on the helpers above.

# (No further changes below)

def get_game_state(
    game: 'Game',
    serialize: bool = True,
) -> 'Union[Dict[str, Any], "GameState"]':
    """Return a consolidated view of the game's state.

    Args:
        game: The Catanatron Game instance to inspect. Must expose a GameState via `game.state` (or be a GameState-like object).
        serialize: If True (default), return a JSON-serializable dictionary summarizing the game state.
                   If False, return the raw GameState object.

    Returns:
        If `serialize` is True, returns a dictionary with the following keys:
            - "players": a list of player dictionaries (prefer player.to_dict() when available).
            - "board": a JSON-serializable board snapshot (as returned by get_board_state(game, serialize=True)).
            - "current_player": the current player's name (string) or None.
            - "phase": the current phase of the game (string).
            - "turn_count": the integer turn counter.
            - "winner": the winner player's name (string) or None.
        If `serialize` is False, returns the raw GameState object instance (the object at `game.state`).

    Raises:
        TypeError: If `game` does not expose `state` (or a GameState-like object) and `serialize` is False.
        ValueError: If expected sub-objects (e.g., players list) cannot be found when serializing.
        Exception: Unexpected exceptions from nested serializers (e.g., board.to_dict()) are allowed to propagate.

    Notes / Implementation details:
        - Resolution order for the GameState:
            1. Try getattr(game, "state", None) and treat that as the GameState object.
            2. If `game.state` is None but `game` itself looks like a GameState, allow `game` to be used directly (duck-typing).
        - When `serialize` is True:
            - Use player.to_dict() for each player if available; otherwise, attempt to extract player.name and player.resources (use the existing get_player_resources helper where possible).
            - Use get_board_state(game, serialize=True) to get the board snapshot (this ensures consistent string keys).
            - current_player should be the resolved GameState.current_player.name if present, otherwise None.
            - The serialized result must contain only JSON-safe primitives, lists, and dicts.
        - The function should be defensive but not over-eager: if some optional field is missing, raise a clear ValueError rather than silently omitting keys.
        - Prefer reusing existing adapters (get_board_state, get_player_resources) to maintain consistent serialization formats.

    Examples:
        # Get a JSON-friendly summary of the game
        snapshot = get_game_state(game)

        # Work with the raw GameState object for lower-level inspection
        state_obj = get_game_state(game, serialize=False)
    """
    # Resolve GameState candidate
    state = getattr(game, "state", None)
    # If serialize is False, return raw state or the game if it *is* a GameState-like object
    if not serialize:
        if state is not None:
            return state
        # Heuristic: if game looks like a GameState (has players/current_player/phase), allow returning it
        if any(hasattr(game, attr) for attr in ("players", "current_player", "phase")):
            return game
        raise TypeError("Game does not expose a GameState via game.state and serialize=False")

    # For serialization, prefer using resolved state object, but allow duck-typing game when state is None
    gs = state if state is not None else game

    # Locate players list; this is essential for serialized output
    players = getattr(gs, "players", None)
    if players is None:
        # Try falling back to top-level game.players
        players = getattr(game, "players", None)
    if players is None:
        raise ValueError("Game object has no players accessible via game.state.players or game.players")

    # Serialize players: prefer player.to_dict() when available
    serialized_players: List[Dict[str, Any]] = []
    for p in players:
        if p is None:
            continue
        if hasattr(p, "to_dict") and callable(getattr(p, "to_dict")):
            try:
                pd = p.to_dict()
                # Ensure JSON-safe primitives by normalizing enums to .name and resource keys to strings
                serialized_players.append(pd)
                continue
            except Exception:
                # Fall back to manual extraction
                pass
        # Manual player serialization
        pl: Dict[str, Any] = {}
        pl_name = getattr(p, "name", None)
        pl["name"] = pl_name if pl_name is not None else str(p)
        # Color normalization
        color = getattr(p, "color", None)
        if color is not None and hasattr(color, "name"):
            pl["color"] = getattr(color, "name")
        else:
            pl["color"] = color
        # Resources: try to use existing helper if available
        gpr = globals().get("get_player_resources")
        if callable(gpr):
            try:
                pl["resources"] = gpr(game, p, serialize=True)  # type: ignore[arg-type]
            except Exception:
                # Fall back to direct access
                if hasattr(p, "resources"):
                    try:
                        pl["resources"] = get_player_resources(game, p, serialize=True)
                    except Exception:
                        pl["resources"] = None
                else:
                    pl["resources"] = None
        else:
            # No helper; inspect p.resources directly
            if hasattr(p, "resources"):
                try:
                    pl["resources"] = get_player_resources(game, p, serialize=True)
                except Exception:
                    pl["resources"] = None
            else:
                pl["resources"] = None
        serialized_players.append(pl)

    # Board snapshot: prefer existing adapter
    gbs = globals().get("get_board_state")
    if callable(gbs):
        board_snapshot = gbs(game, serialize=True)
    else:
        # Try to locate board and call its to_dict if available, else None
        try:
            board_snapshot = get_board_state(game, serialize=True)  # type: ignore[name-defined]
        except Exception:
            board_snapshot = None

    # Current player: prefer helper if available
    gcp = globals().get("get_current_player")
    if callable(gcp):
        try:
            current_player = gcp(game, serialize=True)
        except Exception:
            cp_obj = getattr(gs, "current_player", None)
            current_player = getattr(cp_obj, "name", None) if cp_obj is not None else None
    else:
        cp_obj = getattr(gs, "current_player", None)
        current_player = getattr(cp_obj, "name", None) if cp_obj is not None else None

    # Phase and turn_count
    phase = getattr(gs, "phase", None)
    if phase is not None and hasattr(phase, "name"):
        phase = getattr(phase, "name")
    turn_count = getattr(gs, "turn_count", None)

    # Winner
    winner_obj = getattr(gs, "winner", None)
    winner = None
    if winner_obj is not None:
        winner = getattr(winner_obj, "name", None) if hasattr(winner_obj, "name") else str(winner_obj)

    # Assemble final dictionary
    result: Dict[str, Any] = {
        "players": serialized_players,
        "board": board_snapshot,
        "current_player": current_player,
        "phase": phase,
        "turn_count": turn_count,
        "winner": winner,
    }

    return result
