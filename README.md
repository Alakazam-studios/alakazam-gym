# alakazam-gym

Gymnasium-compatible environments for training policies inside Alakazam
world models.

Three surfaces, one package:

| Class | What it is | Cost |
|---|---|---|
| `LocalDreamEnv` | The public [DOOM Dungeon HG](https://huggingface.co/alakazamworld/doom-dungeon-hg) world model, running locally via onnxruntime | free |
| `RemoteSimEnv` | The hosted simulation gym (`/v1/sim/sessions`, SNN observation contract) | metered (GPU stream per session) |
| `ExamClient` | Certification in the frozen Webots oracle via the Train API | scheduled |

## Install

```bash
pip install git+https://github.com/Alakazam-studios/alakazam-gym
# or, from a clone:
pip install -e .
```

The first `LocalDreamEnv()` call downloads the public weights (~280 MB) into
`~/.cache/alakazam-gym/` and sha256-verifies them against the published pins.
Already have them? `export ALAKAZAM_GYM_WEIGHTS_DIR=/path/to/weights`.

## 10-line random agent

```python
import alakazam_gym, gymnasium as gym

env = gym.make("Alakazam/DoomDream-v0")
obs, info = env.reset(seed=7)
for t in range(100):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if obs["sensor_valid"]:
        print(t, "prox", obs["proximity"], "collision", obs["collision"])
env.close()
```

Each `step` is one world-model inference (~0.1–0.3 s on a laptop CPU;
the same graph runs under onnxruntime-web WebGPU in the browser clients).

## What the observation means

- `frame` — the model's rendered view, float32 `(3, 64, 64)` in `[-1, 1]`.
- `proximity` — calibrated virtual range readout `{"left","right"}` in `[0, 1]`
  (the same dict shape as the hosted sim API and the exam policy slot)
  from the geometry the world model is conditioned on (0 = clear,
  1 = contact range ≈ 30 cm at robot scale).
- `collision` — 1 when a forward move was blocked by a wall this step.
- `sensor_valid` — 0 during the first steps after reset. Never sense or score
  steps with `sensor_valid == 0` (post-reset blindness). The physics exam has
  no such window, so never make a policy depend on blindness. See
  `docs/HONEST_EVAL.md`.

Reward is always `0.0`; shaping is yours.

## SNN sketch

```python
from alakazam_gym import LocalDreamEnv

env = LocalDreamEnv(seed=7)
obs, _ = env.reset()
while True:
    # encode: bounded scalars -> input spike rates
    rates = obs["proximity"]            # {"left","right"} each in 0..1
    pain  = obs["collision"]            # punishment spike
    # your spiking network here: integrate for one tick, decode motor pops
    action = my_snn.tick(rates, pain)   # -> 0..4 (noop/fwd/left/right/attack)
    obs, _, _, _, _ = env.step(action)
```

Training against the hosted worlds instead (e-puck/robot, wheels commands,
camera): `RemoteSimEnv(base_url, key, world="epuck")`. Same loop, metered
sessions, and always `env.close()`.

## Certify: the exam is the scoreboard

Dream and gym numbers are training telemetry, never capability claims. When
your controller is worth a claim, certify it in the frozen Webots oracle:

```python
from alakazam_gym import ExamClient
c = ExamClient("https://api.alakazam.gg/train", key=TRAIN_KEY)
job = c.submit_exam_only("my-cert-001", genome6=[...])    # or genome9=[...]
result = c.wait(job["job_id"])
print(result["oracle"]["verdict"])                        # 4-bar verdict
```

## Certify your own policy

You are not limited to the 9-float controller family. Package any policy that
implements the `reset(seed)` + `act(obs)` contract and submit it as a sandboxed
python module. The `obs` your policy sees in the exam is the same dict the local
gym emits (`proximity`, `collision`, `sensor_valid`), and the action is the same
wheel-fraction vocabulary, so a policy you trained in `LocalDreamEnv` certifies
unchanged.

```python
# my_policy.py
class Policy:
    def reset(self, seed):
        self.net = load_my_snn_weights()      # numpy inference over exported weights
    def act(self, obs):
        L = obs["proximity"]["left"]; R = obs["proximity"]["right"]
        wl, wr = self.net.step(L, R, spike=obs["collision"])
        return {"wheels": {"left": wl, "right": wr}}   # each in [0,1]
```

```python
from alakazam_gym import ExamClient, PolicyBundle
c = ExamClient("https://api.alakazam.gg/train", key=TRAIN_KEY)
bundle = PolicyBundle("my_policy.py")          # or a directory -> zipped
job = c.submit_policy("my-cert-001", bundle)
print(c.wait(job["job_id"])["oracle"]["verdict"])
```

Sandbox: the module runs inside the exam container with no network egress
and resource/time limits (trusted-partner posture, key-gated). The container
ships numpy and onnxruntime; a policy needing torch or norse will not import,
so export your weights and run a numpy forward pass. There is deliberately no
ONNX detour: ship a python module, not a converted graph. The policy replaces
only the champion arm; the world, spawns, episode count, tick, proximity
remap, scoring, and the anti-exploit control arms stay frozen.

Full API reference and guides: https://docs.alakazam.gg/train

## Licensing

- Code in this repo: MIT (`LICENSE`).
- Model weights (`alakazamworld/doom-dungeon-hg`): CC BY-NC-SA. The
  downloader does not change that; commercial use of the weights needs a
  separate arrangement.

## Provenance

`LocalDreamEnv` ports the reference environment `doom_env_v55.py`
(dream-robot release; also published at
`huggingface.co/Karajan42/dream-robot-gauntlet-policies`) onto the
Gymnasium API, unchanged in its model contract (7-input ONNX, 1-step EDM
sampler, `sigma_cond=0.5`, `noise_aug_bucket=5`, 12-channel conditioning
minimap — see the weights repo's `HOSTING.md`).
