#!/usr/bin/env python
"""Random agent in the local dream — the smallest possible loop.

Run: python examples/random_agent.py [steps]
Uses ALAKAZAM_GYM_WEIGHTS_DIR if set, else downloads the public weights once.
"""
import sys

import gymnasium as gym

import alakazam_gym  # noqa: F401  (registers Alakazam/DoomDream-v0)

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
env = gym.make("Alakazam/DoomDream-v0")
obs, info = env.reset(seed=7)
print("spawn pose:", info["pose"])
collisions = 0
for t in range(steps):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if obs["sensor_valid"]:
        collisions += obs["collision"]
        if t % 10 == 0:
            print(f"t={t:3d} prox=({obs['proximity'][0]:.2f},{obs['proximity'][1]:.2f}) "
                  f"frame[{obs['frame'].min():+.2f},{obs['frame'].max():+.2f}] "
                  f"collisions={collisions}")
env.close()
print(f"done: {steps} steps, {collisions} blocked-forward events")
