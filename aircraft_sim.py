import numpy as np

class AircraftSim:
    def __init__(self, model, dt):
        self.model = model
        self.dt = float(dt)

    def step(self, x, u):
        dt = self.dt

        k1 = np.array(self.model.f(x, u), dtype=float)
        k2 = np.array(self.model.f(x + 0.5 * dt * k1, u), dtype=float)
        k3 = np.array(self.model.f(x + 0.5 * dt * k2, u), dtype=float)
        k4 = np.array(self.model.f(x + dt * k3, u), dtype=float)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)