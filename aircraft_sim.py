import numpy as np

class AircraftSim:
    def __init__(self, model, dt, lqr, params):
        self.model = model
        self.dt = float(dt)
        self.lqr = lqr
        self.p = params

    def step(self, x, u):
        dt = self.dt
        k1 = np.array(self.model.f(x, u), dtype=float)
        k2 = np.array(self.model.f(x + 0.5 * dt * k1, u), dtype=float)
        k3 = np.array(self.model.f(x + 0.5 * dt * k2, u), dtype=float)
        k4 = np.array(self.model.f(x + dt * k3, u), dtype=float)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step_linear(self, x, u):
        def f_linear(x, u):
            dx = x - self.p.xr
            du = u - self.p.ur
            dxdot = self.lqr.A @ dx + self.lqr.B @ du
            return dxdot

        k1 = f_linear(x, u)
        k2 = f_linear(x + 0.5 * self.dt * k1, u)
        k3 = f_linear(x + 0.5 * self.dt * k2, u)
        k4 = f_linear(x + self.dt * k3, u)

        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)