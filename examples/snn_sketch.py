#!/usr/bin/env python
"""SNN controller sketch — a minimal LIF pair driving the local dream.

Not a trained network: a wiring template showing where a spiking model
plugs in (proximity -> input rates, collision -> punishment spike,
motor populations -> discrete action). Replace `TinyLIF` with your SNN.
"""
import numpy as np

from alakazam_gym import LocalDreamEnv


class TinyLIF:
    """Two leaky integrate-and-fire motor units (turn-left / turn-right).
    Crossed excitation: right proximity drives the turn-left unit and
    vice versa; forward when neither fires."""

    def __init__(self, tau=0.7, thresh=1.0):
        self.v = np.zeros(2)
        self.tau, self.thresh = tau, thresh

    def tick(self, prox_lr, pain):
        left_in, right_in = prox_lr
        drive = np.array([right_in, left_in])          # crossed
        self.v = self.tau * self.v + drive + (2.0 if pain else 0.0)
        spikes = self.v >= self.thresh
        self.v[spikes] = 0.0
        if spikes[0]:
            return 2   # turn_left
        if spikes[1]:
            return 3   # turn_right
        return 1       # forward


env = LocalDreamEnv(seed=11)
obs, _ = env.reset()
snn = TinyLIF()
survived = 0
for t in range(80):
    if obs["sensor_valid"]:
        a = snn.tick((obs["proximity"]["left"], obs["proximity"]["right"]), obs["collision"])
    else:
        a = 0  # blind steps: do nothing, never sense
    obs, _, _, _, info = env.step(a)
    survived = t
print(f"ran {survived + 1} steps; final prox L={obs['proximity']['left']:.2f} R={obs['proximity']['right']:.2f}")
env.close()
