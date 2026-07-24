"""reactive_policy — a trivially different, NON-LIF, pure-python reactive
controller. No spiking, no genome, no numpy: just if/else on proximity. Its
only purpose is to prove the exam's policy slot is REAL and not genome9-shaped
— an arbitrary partner controller runs end to end and gets a real verdict.

Behavior: veer away from the nearer side; crawl forward when clear. A braitenberg
avoider. It will NOT pass the frozen 4-bar verdict (nothing has), but it must
run all 40 episodes and produce a well-formed, non-vacuous result (cruiser arm
still fails as required).
"""


class Policy:
    def reset(self, seed):
        self._t = 0

    def act(self, obs):
        p = obs["proximity"]
        left, right = p["left"], p["right"]
        self._t += 1
        near = max(left, right)
        if near < 0.15:
            return {"wheels": {"left": 1.0, "right": 1.0}}        # clear: forward
        if left > right:
            return {"wheels": {"left": 1.0, "right": 0.2}}        # obstacle left: veer right
        return {"wheels": {"left": 0.2, "right": 1.0}}            # obstacle right: veer left
