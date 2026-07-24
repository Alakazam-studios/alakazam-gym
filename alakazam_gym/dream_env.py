"""LocalDreamEnv — Gymnasium environment over the public DOOM Dungeon HG
world model, running locally via onnxruntime (CPU by default).

PROVENANCE: the world-model machinery (HeightRaycaster, 12-channel minimap
builder, pose integration, EDM sampler constants) is ported from
`doom_env_v55.py` (dream-robot release; also published at
huggingface.co/Karajan42/dream-robot-gauntlet-policies `env/doom_env_v55.py`).
The serving contract is documented in HOSTING.md of the weights repo
(huggingface.co/alakazamworld/doom-dungeon-hg): 7-input ONNX, 1-step sampler,
sigma_cond=0.5, noise_aug_bucket=5, 12-ch conditioning minimap. Constants are
that checkpoint's Self-Forcing recipe — do not change them.

Honest-evaluation contract built into this env:
- RESET BLINDNESS: the first `blind_steps` (default 2) observations after
  reset carry `sensor_valid=False` and zeroed proximity — never sense or
  score them (the certification exam won't either).
- The env emits NO reward by default (`reward=0.0`); reward shaping is the
  caller's, and dream performance is never a capability claim — the frozen
  Webots exam is the sole scoreboard (see ExamClient).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("alakazam-gym requires gymnasium: pip install gymnasium") from e

from .weights import ensure_weights

MAP_SIZE, TARGET = 128, 64
ARENA = 1856.0
FOV_HALF_PLANE = 1.0  # tan(45 deg)
NUM_RAYS, MAX_DEPTH = 64, 3000.0
MOVE, TURN = 13.15, 14.06
A_NOOP, A_FWD, A_TL, A_TR, A_ATK = 0, 1, 2, 3, 4
SCALE = MAP_SIZE / ARENA
STEP_UP_LIMIT = 41.0
PROX_RANGE = 117.0  # px; ~30 cm at robot scale (calibrated readout range)
ACTION_NAMES = ("noop", "forward", "turn_left", "turn_right", "attack")


def _norm_tex(tid: float) -> float:
    return (max(-1.0, min(8.0, float(tid))) / 8.0) * 2.0 - 1.0


class _HeightRaycaster:
    """Camera-plane DDA raycaster over the 128x128 wall/height grids.
    Port of doom_env_v55.HeightRaycaster (see module provenance)."""

    def __init__(self, maps):
        self.m = maps

    def is_wall(self, col, row, player_floor_h):
        if col < 0 or col >= MAP_SIZE or row < 0 or row >= MAP_SIZE:
            return True  # OOB = wall (reference behavior)
        i = row * MAP_SIZE + col
        if self.m["walls"][i] > 0.9:
            return True
        return (self.m["floorH"][i] - player_floor_h) > STEP_UP_LIMIT

    def check_collision(self, wx, wy):
        col = int(wx * SCALE)
        row = int((ARENA - wy) * SCALE)
        return self.is_wall(col, row, self._floor_at(wx, wy))

    def _floor_at(self, wx, wy):
        col = min(MAP_SIZE - 1, max(0, int(wx * SCALE)))
        row = min(MAP_SIZE - 1, max(0, int((ARENA - wy) * SCALE)))
        return self.m["floorH"][row * MAP_SIZE + col]

    def cast(self, wx, wy, yaw_deg):
        px, py = wx * SCALE, (ARENA - wy) * SCALE
        pf = self._floor_at(wx, wy)
        yaw = math.radians(yaw_deg)
        out = np.zeros((NUM_RAYS, 6), np.float32)
        for r in range(NUM_RAYS):
            plane_x = FOV_HALF_PLANE - (r + 0.5) / NUM_RAYS * 2.0 * FOV_HALF_PLANE
            ang = yaw + math.atan(plane_x)
            dx, dy = math.cos(ang), -math.sin(ang)
            eps = 1e-10
            dx = dx if abs(dx) > eps else eps
            dy = dy if abs(dy) > eps else eps
            mx, my = int(math.floor(px)), int(math.floor(py))
            sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
            tmx = ((mx + 1 - px) / dx) if dx > 0 else ((mx - px) / dx)
            tmy = ((my + 1 - py) / dy) if dy > 0 else ((my - py) / dy)
            tdx, tdy = abs(1.0 / dx), abs(1.0 / dy)
            last_open = my * MAP_SIZE + mx
            hit = None
            for _ in range(256):
                if self.is_wall(mx, my, pf):
                    t = min(tmx, tmy)
                    hx, hy = px + t * dx, py + t * dy
                    d = max(math.hypot(hx - px, hy - py), 0.01)
                    if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                        hit = (d, self.m["wallTex"][my * MAP_SIZE + mx])
                    else:
                        hit = (d, -1.0)
                    break
                if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                    last_open = my * MAP_SIZE + mx
                if tmx < tmy:
                    tmx += tdx
                    mx += sx
                else:
                    tmy += tdy
                    my += sy
            if hit is None:
                out[r] = (MAX_DEPTH, -1.0, 0.0, 0.0, 128.0, 0.0)
            else:
                d, wtex = hit
                out[r] = (d / SCALE, wtex, self.m["floorH"][last_open],
                          self.m["floorTex"][last_open], self.m["ceilH"][last_open],
                          self.m["ceilTex"][last_open])
        return out


def open_cells(walls_flat, margin=3):
    """Spawnable world coordinates: cells with a `margin`-cell clear box."""
    out = []
    w = np.asarray(walls_flat).reshape(MAP_SIZE, MAP_SIZE) > 0.5
    for r in range(margin, MAP_SIZE - margin):
        for c in range(margin, MAP_SIZE - margin):
            if not w[r - margin:r + margin + 1, c - margin:c + margin + 1].any():
                out.append(((c + 0.5) / SCALE, ARENA - (r + 0.5) / SCALE))
    return out


class LocalDreamEnv(gym.Env):
    """Train inside the dream, on your own hardware, for free.

    Observation (Dict):
      frame      Box(-1, 1, (3, 64, 64), float32) — the model's rendered view
      proximity  Box(0, 1, (2,), float32) — calibrated virtual range readout,
                 left/right half of the 64-ray depth fan; 0 = clear,
                 1 = contact range (PROX_RANGE = 117 px ≈ 30 cm robot scale)
      collision  Discrete(2) — 1 when a forward move was blocked by a wall
                 this step (the dream analog of a contact)
      sensor_valid Discrete(2) — 0 during post-reset blindness; NEVER score
                 steps with sensor_valid == 0

    Action: Discrete(5) — noop / forward / turn_left / turn_right / attack.

    Reward: always 0.0 — bring your own shaping. Dream scores are not
    capability claims; certify via the Webots exam (ExamClient).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, weights_dir: str | None = None, num_steps: int = 1,
                 map_path: str | None = None, blind_steps: int = 2,
                 prox_range: float = PROX_RANGE, seed: int | None = None):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "alakazam-gym requires onnxruntime: pip install onnxruntime") from e
        wdir = ensure_weights(weights_dir)
        self._sess = ort.InferenceSession(
            os.path.join(wdir, "denoiser.onnx"), providers=["CPUExecutionProvider"])
        init = json.load(open(os.path.join(wdir, "init_state.json")))
        self._init_obs = np.array(init["obs_buffer"], np.float32).reshape(12, 64, 64)
        self._init_act = np.array(init["act_buffer"], np.int64)

        map_path = map_path or os.path.join(
            os.path.dirname(__file__), "data", "default_map.json")
        grid = np.array(json.load(open(map_path)), np.float32)
        walls = (grid > 0.9).astype(np.float32)
        n = MAP_SIZE * MAP_SIZE
        self._maps = {
            "walls": np.where(walls.reshape(-1) > 0.5, 1.0, -1.0).astype(np.float32),
            "floorH": np.zeros(n, np.float32),
            "ceilH": np.full(n, 128.0, np.float32),
            "wallTex": np.where(walls.reshape(-1) > 0.5, 0.0, -1.0).astype(np.float32),
            "floorTex": np.zeros(n, np.float32),
            "ceilTex": np.zeros(n, np.float32),
        }
        self._rc = _HeightRaycaster(self._maps)
        w = self._maps["walls"].reshape(MAP_SIZE, MAP_SIZE)
        self._walls_down = np.where(w[::2, ::2] > 0.9, 1.0, -1.0).astype(np.float32)
        self._spawns = open_cells(self._maps["walls"])
        if not self._spawns:
            raise RuntimeError("map has no spawnable open cells")

        rho, smin, smax = 7.0, 0.002, 5.0
        if num_steps == 1:
            self._sigmas = [smax, 0.0]
        else:
            i = np.arange(num_steps)
            self._sigmas = ((smax ** (1 / rho) + i / (num_steps - 1)
                             * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
                            ).tolist() + [0.0]

        self._blind_steps = int(blind_steps)
        self._prox_range = float(prox_range)
        self._steps_since_reset = 0
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Dict({
            "frame": spaces.Box(-1.0, 1.0, (3, 64, 64), np.float32),
            "proximity": spaces.Box(0.0, 1.0, (2,), np.float32),
            "collision": spaces.Discrete(2),
            "sensor_valid": spaces.Discrete(2),
        })
        self.action_space = spaces.Discrete(5)

    # -- internals ----------------------------------------------------------
    def _marker(self, ch, wx, wy, yaw):
        ci = int(wx * TARGET / ARENA)
        ri = int((ARENA - wy) * TARGET / ARENA)
        if 0 <= ri < TARGET and 0 <= ci < TARGET:
            ch[ri, ci] = 1.0
        yr = math.radians(yaw)
        for s in range(1, 5):
            pc = round(ci + math.cos(yr) * s)
            pr = round(ri - math.sin(yr) * s)
            if 0 <= pr < TARGET and 0 <= pc < TARGET:
                ch[pr, pc] = 1.0

    def _minimap(self, wx, wy, yaw, rays):
        mm = np.zeros((12, TARGET, TARGET), np.float32)
        depths = rays[:, 0]
        clamped = np.clip(depths, 1.0, MAX_DEPTH)
        disp = (np.log(clamped + 1.0) / math.log(MAX_DEPTH + 1.0) * 2 - 1).astype(np.float32)
        mm[0] = np.clip(disp, -1, 1)[None, :].repeat(TARGET, 0)
        mm[1] = self._walls_down.copy()
        self._marker(mm[1], wx, wy, yaw)
        self._marker(mm[2], wx, wy, yaw)   # items_td: empty arena, marker only
        mm[4].fill(-1.0)                    # enemies_td background
        self._marker(mm[4], wx, wy, yaw)
        mm[7] = np.array([_norm_tex(t) for t in rays[:, 1]], np.float32)[None, :].repeat(TARGET, 0)
        mm[8] = np.clip(rays[:, 2] / 256.0, -1, 1)[None, :].repeat(TARGET, 0)
        mm[9] = np.array([_norm_tex(t) for t in rays[:, 3]], np.float32)[None, :].repeat(TARGET, 0)
        mm[10] = np.clip(rays[:, 4] / 256.0, -1, 1)[None, :].repeat(TARGET, 0)
        mm[11] = np.array([_norm_tex(t) for t in rays[:, 5]], np.float32)[None, :].repeat(TARGET, 0)
        return mm

    def _denoise(self, x, sigma):
        return self._sess.run(None, {
            "noisy_next_obs": x[None],
            "sigma": np.array([sigma], np.float32),
            "sigma_cond": np.array([0.5], np.float32),   # checkpoint constant
            "obs": self._obs[None],
            "act": self._act[None],
            "minimap": self._mm[None],
            "noise_aug_bucket": np.array([5], np.int64),  # checkpoint constant
        })[0][0]

    def _proximity(self, depths):
        half = NUM_RAYS // 2
        dl, dr = depths[:half].min(), depths[half:].min()
        f = lambda d: float(np.clip(1.0 - d / self._prox_range, 0.0, 1.0))
        return f(dl), f(dr)

    def _obs_dict(self, frame, depths, blocked):
        valid = self._steps_since_reset > self._blind_steps
        if valid:
            pl, pr = self._proximity(depths)
        else:
            pl = pr = 0.0
        return {
            "frame": frame.astype(np.float32),
            "proximity": np.array([pl, pr], np.float32),
            "collision": int(bool(blocked)) if valid else 0,
            "sensor_valid": int(valid),
        }

    # -- gymnasium API ------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "pose" in options:
            x, y, yaw = options["pose"]
        else:
            x, y = self._spawns[int(self._rng.integers(len(self._spawns)))]
            yaw = float(self._rng.uniform(0.0, 360.0))
        self._obs = self._init_obs.copy()
        self._act = self._init_act.copy()
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        rays = self._rc.cast(self.x, self.y, self.yaw)
        self._mm = self._minimap(self.x, self.y, self.yaw, rays)
        self._steps_since_reset = 0
        obs = self._obs_dict(self._obs[-3:], rays[:, 0], blocked=False)
        return obs, {"pose": (self.x, self.y, self.yaw)}

    def step(self, action):
        a = int(action)
        blocked = False
        if a == A_FWD:
            rad = math.radians(self.yaw)
            nx, ny = self.x + math.cos(rad) * MOVE, self.y + math.sin(rad) * MOVE
            if not self._rc.check_collision(nx, ny):
                self.x, self.y = nx, ny
            else:
                blocked = True
        elif a == A_TL:
            self.yaw += TURN
        elif a == A_TR:
            self.yaw = ((self.yaw - TURN) % 360 + 360) % 360

        rays = self._rc.cast(self.x, self.y, self.yaw)
        self._mm = self._minimap(self.x, self.y, self.yaw, rays)
        self._act = np.roll(self._act, -1)
        self._act[-1] = a
        last = self._obs[-3:]
        x = last + self._rng.normal(0, self._sigmas[0], last.shape).astype(np.float32)
        for i in range(len(self._sigmas) - 1):
            den = np.clip(self._denoise(x, self._sigmas[i]), -1, 1)
            d = (x - den) / self._sigmas[i]
            x = x + d * (self._sigmas[i + 1] - self._sigmas[i])
        frame = np.clip(x, -1, 1).astype(np.float32)
        self._obs = np.concatenate([self._obs[3:], frame], 0)
        self._steps_since_reset += 1

        obs = self._obs_dict(frame, rays[:, 0], blocked)
        info = {"pose": (self.x, self.y, self.yaw), "blocked": blocked}
        return obs, 0.0, False, False, info

    def render(self):
        return (((self._obs[-3:].transpose(1, 2, 0) + 1.0) / 2.0) * 255).astype(np.uint8)
