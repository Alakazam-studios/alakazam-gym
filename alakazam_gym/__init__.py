"""alakazam-gym — train policies inside Alakazam world models.

Three surfaces:
  LocalDreamEnv  — the public DOOM Dungeon HG world model, locally (free).
  RemoteSimEnv   — the hosted simulation gym (/v1/sim sessions, SNN contract).
  ExamClient     — certification in the frozen Webots oracle (Forge API).

Honest-evaluation rules (docs/HONEST_EVAL.md): never score post-reset
blindness steps; dream/gym scores are telemetry, the frozen exam is the sole
scoreboard.

ExamClient is stdlib-only; the envs require gymnasium (+ onnxruntime for
LocalDreamEnv) and import lazily so cert-only use works without them.
"""
from .exam import ExamClient, ExamError, PolicyBundle
from .weights import ensure_weights

__version__ = "0.3.0"
__all__ = ["LocalDreamEnv", "RemoteSimEnv", "ExamClient", "ExamError",
           "PolicyBundle", "ensure_weights", "ACTION_NAMES"]

_LAZY = {"LocalDreamEnv": ".dream_env", "ACTION_NAMES": ".dream_env",
         "RemoteSimEnv": ".remote_sim"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:  # optional gymnasium registry entry: gym.make("Alakazam/DoomDream-v0")
    from gymnasium.envs.registration import register

    register(id="Alakazam/DoomDream-v0",
             entry_point="alakazam_gym.dream_env:LocalDreamEnv")
except Exception:  # pragma: no cover — registration is best-effort
    pass
