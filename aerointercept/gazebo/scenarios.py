"""Seeded, bounded reference flights for a cooperative rendezvous target.

Trajectories depend only on episode seed and simulation time. They do not
react to the follower. Held-out flight shapes are separate from training.
"""

from dataclasses import dataclass, asdict
import math

import numpy as np


TRAIN_MODES = ("circle", "sinusoidal", "random_walk")
HELD_OUT_MODES = ("figure_eight", "stop_go")
MODES = ("hover", *TRAIN_MODES, *HELD_OUT_MODES)


@dataclass
class Scenario:
    mode: str
    seed: int
    start_ned: list[float]
    radius: float
    omega: float
    phase: float
    heading: float

    @classmethod
    def sample(cls, mode, seed, home=(0., 0., -6.), distance=10.):
        if mode not in MODES:
            raise ValueError(f"unsupported scenario: {mode}")
        rng = np.random.default_rng(seed)
        bearing = float(rng.uniform(-.15, .15))
        start = np.asarray(home) + [distance*math.cos(bearing), distance*math.sin(bearing), 0.]
        return cls(mode, int(seed), start.tolist(), float(rng.uniform(2., 4.)),
                   float(rng.uniform(.15, .35)), float(rng.uniform(-math.pi, math.pi)),
                   float(rng.uniform(-math.pi, math.pi)))

    def metadata(self):
        return {"version": "bounded_cooperative_v1", **asdict(self)}

    def position(self, seconds):
        t = max(0., float(seconds))
        # A gradual start also makes target resets physically repeatable.
        u = self.omega * (t - 1.5*(1.-math.exp(-t/1.5)))
        if self.mode == "hover":
            xy = [0., 0.]
        elif self.mode == "circle":
            xy = [math.sin(u), 1.-math.cos(u)]
        elif self.mode == "sinusoidal":
            xy = [1.3*math.sin(.6*u), .35*math.sin(2*u)]
        elif self.mode == "random_walk":
            # Seeded smooth quasiperiodic wandering; no uncontrolled RNG timer.
            p = self.phase
            xy = [.6*math.sin(u)+.3*(math.sin(1.7*u+p)-math.sin(p)),
                  .6*(math.cos(.8*u+p)-math.cos(p))+.3*math.sin(2.3*u)]
        elif self.mode == "figure_eight":
            xy = [math.sin(u), math.sin(u)*math.cos(u)]
        else:  # Held-out smooth stop-and-go motion along a different path.
            v = u - math.sin(u)
            xy = [math.sin(v), .7*(1.-math.cos(v))]
        c, s = math.cos(self.heading), math.sin(self.heading)
        x, y = self.radius*np.asarray(xy)
        return np.asarray(self.start_ned) + [c*x-s*y, s*x+c*y, 0.]

    def velocity(self, seconds):
        # Only the target setpoint feed-forward uses this deterministic derivative.
        dt = .001
        return (self.position(seconds+dt)-self.position(max(0.,seconds-dt))) / (
            dt + min(dt, max(0., seconds))
        )
