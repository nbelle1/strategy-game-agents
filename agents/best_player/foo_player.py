# Robust import that works both as a package and when run locally
try:
    from .adapters import (
        Game, Player, Color, Action, ActionType,
        playable_actions, pruned_actions, chance_children,
        make_value_fn, DEFAULT_WEIGHTS, value_production,
        production_features_sampler, winning_color, copy_game, call_llm
    )
except ImportError:
    # Fallback if executed as a script from within this folder
    from adapters import (
        Game, Player, Color, Action, ActionType,
        playable_actions, pruned_actions, chance_children,
        make_value_fn, DEFAULT_WEIGHTS, value_production,
        production_features_sampler, winning_color, copy_game
    )

from datetime import datetime
from pathlib import Path
import json
import dataclasses
import enum
import math

class FooPlayer(Player):
    def __init__(self, color, value_builder="base_fn", params=DEFAULT_WEIGHTS):
        super().__init__(color)
        self.V = make_value_fn(value_builder, params)  # callable(game, pov_color)

    # --- Logging helpers ---
    def _now_stamp(self) -> str:
        # Example: 20250917-143012-123456
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    def _ensure_log_dir(self, game) -> Path:
        # One folder per game: game_{time}. Attach to the game object so all players share it.
        log_dir = getattr(game, "_log_dir", None)
        if log_dir is None:
            log_dir = Path.cwd() / f"game_{self._now_stamp()}"
            log_dir.mkdir(parents=True, exist_ok=True)
            setattr(game, "_log_dir", log_dir)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _to_jsonable(self, obj, _seen: set[int] | None = None):
        if _seen is None:
            _seen = set()

        # Primitives/enums: return directly; do NOT add to _seen
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else str(obj)
        if isinstance(obj, enum.Enum):
            return obj.name

        # Track only complex objects/containers to break real cycles
        oid = id(obj)
        if oid in _seen:
            return f"<recursion:{type(obj).__name__}>"
        _seen.add(oid)

        # Dataclasses
        if dataclasses.is_dataclass(obj):
            return {k: self._to_jsonable(v, _seen) for k, v in dataclasses.asdict(obj).items()}

        # Dicts
        if isinstance(obj, dict):
            return {str(self._to_jsonable(k, _seen)): self._to_jsonable(v, _seen) for k, v in obj.items()}

        # Iterables
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [self._to_jsonable(v, _seen) for v in obj]

        # Objects with __dict__
        if hasattr(obj, "__dict__"):
            # Skip private attrs and callables (methods/functions) to keep logs readable
            data = {k: v for k, v in obj.__dict__.items() if not k.startswith("_") and not callable(v)}
            if not data:
                data = {k: v for k, v in obj.__dict__.items() if not callable(v)}
            return {
                "__type__": type(obj).__name__,
                **{k: self._to_jsonable(v, _seen) for k, v in data.items()},
            }

        # Fallback
        try:
            return str(obj)
        except Exception:
            return f"<unserializable:{type(obj).__name__}>"

    def _log_state(self, game):
        log_dir = self._ensure_log_dir(game)
        fname = log_dir / f"state_{self._now_stamp()}.json"
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "player_color": getattr(self, "color", None).name if hasattr(self, "color") else None,
                "game_state": self._to_jsonable(game),
            }
            with fname.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        except Exception as e:
            # Keep gameplay resilient if logging fails
            print(f"[FooPlayer] Failed to write state log: {e}")

    def decide(self, game, _playable):
        # Log the full state at the start of this turn
        self._log_state(game)

        acts = pruned_actions(game)
        if not acts:
            acts = playable_actions(game)
        if len(acts) == 1:
            return acts[0]
    
        # Expectation over chance outcomes, mirroring AlphaBeta’s spectrum
        exp = {}
        for a, outs in chance_children(game, acts).items():
            exp[a] = sum(p * self.V(gp, self.color) for gp, p in outs)

        # choose max-EV action
        print(f"FooPlayer {self.color} chose action {max(exp, key=exp.get)} with expected value {max(exp.values())}")
        
        return max(exp, key=exp.get)
