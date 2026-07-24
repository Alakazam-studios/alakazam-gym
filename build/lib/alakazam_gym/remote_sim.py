"""RemoteSimEnv — Gymnasium wrapper for the hosted simulation gym
(`/v1/sim/sessions`, SNN observation contract).

Each session holds a REAL world-model GPU stream on the host side — sessions
are metered by wall-time. Always close() (tears the session down); an
orphaned session keeps billing.

Contract (see the Forge API reference, Simulation gym tag):
  observation: proximity {left,right} in [0,1], collision spike (edge-
  triggered), collision_count, done, t_ms, sensor_age_ms, labels, optional
  camera (data-URL JPEG).
  action: discrete left/right/forward/none, or SNN-native differential
  `wheels` — this env's default action space (Box(-1, 1, (2,))).
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("alakazam-gym requires gymnasium: pip install gymnasium") from e

DISCRETE_ACTIONS = ("none", "forward", "left", "right")


class RemoteSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url: str, key: str | None = None,
                 world: str = "epuck", camera: int = 0,
                 terminal_collision: bool = False, hold_ms: int = 300,
                 action_mode: str = "wheels", timeout_s: float = 60.0):
        super().__init__()
        self.base = base_url.rstrip("/")
        self.key = key
        self.world = world
        self.camera = int(camera)
        self.terminal_collision = bool(terminal_collision)
        self.hold_ms = int(hold_ms)
        self.timeout_s = timeout_s
        self.session_id: str | None = None

        if action_mode not in ("wheels", "discrete"):
            raise ValueError("action_mode must be 'wheels' or 'discrete'")
        self.action_mode = action_mode
        if action_mode == "wheels":
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        obs_spaces: dict[str, Any] = {
            # {"left","right"} dict form — same shape as the wire API, the
            # local dream env, and the exam's policy slot
            "proximity": spaces.Dict({
                "left": spaces.Box(0.0, 1.0, (), np.float32),
                "right": spaces.Box(0.0, 1.0, (), np.float32),
            }),
            "collision": spaces.Discrete(2),
        }
        if self.camera:
            obs_spaces["camera"] = spaces.Box(
                0, 255, (self.camera, self.camera, 3), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

    # -- http ---------------------------------------------------------------
    def _req(self, method: str, path: str, body=None):
        r = urllib.request.Request(self.base + path, method=method)
        if self.key:
            r.add_header("Authorization", f"Bearer {self.key}")
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            r.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(r, data, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:200]}") from e

    def _to_obs(self, raw):
        obs = {
            "proximity": {"left": float(raw["proximity"]["left"]),
                          "right": float(raw["proximity"]["right"])},
            "collision": int(bool(raw.get("collision"))),
        }
        if self.camera and raw.get("camera", {}).get("image"):
            try:
                from PIL import Image
                b64 = raw["camera"]["image"].split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                obs["camera"] = np.asarray(img, np.uint8)
            except ImportError:
                obs["camera"] = np.zeros((self.camera, self.camera, 3), np.uint8)
        return obs, raw

    # -- gymnasium API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.session_id is None:
            s = self._req("POST", "/v1/sim/sessions", {
                "world": self.world, "camera": self.camera,
                "terminal_collision": self.terminal_collision})
            self.session_id = s["session_id"]
            raw = self._req("GET", f"/v1/sim/sessions/{self.session_id}/obs")
        else:
            raw = self._req("POST", f"/v1/sim/sessions/{self.session_id}/reset")
        obs, raw = self._to_obs(raw)
        return obs, {"raw": raw}

    def step(self, action):
        if self.session_id is None:
            raise RuntimeError("call reset() first")
        if self.action_mode == "wheels":
            body = {"wheels": {"left": float(action[0]), "right": float(action[1])},
                    "holdMs": self.hold_ms}
        else:
            body = {"action": DISCRETE_ACTIONS[int(action)], "holdMs": self.hold_ms}
        raw = self._req("POST", f"/v1/sim/sessions/{self.session_id}/step", body)
        obs, raw = self._to_obs(raw)
        terminated = bool(raw.get("done"))
        info = {"raw": raw, "t_ms": raw.get("t_ms"),
                "collision_count": raw.get("collision_count"),
                "sensor_age_ms": raw.get("sensor_age_ms"),
                "applied_action": raw.get("applied_action")}
        return obs, 0.0, terminated, False, info

    def close(self):
        if self.session_id is not None:
            try:
                self._req("DELETE", f"/v1/sim/sessions/{self.session_id}")
            finally:
                self.session_id = None
