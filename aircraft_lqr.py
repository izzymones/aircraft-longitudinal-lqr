import numpy as np
import casadi as ca
import scipy.linalg as la

class AircraftLQR:
    def __init__(self, params, model):
        self.p = params
        self.model = model
        self.A = None
        self.B = None
        self.K = None

        self.A_func = None
        self.B_func = None

    def build_linearization_funcs(self):
        A_sx = ca.jacobian(self.model.dynamics, self.model.state)
        B_sx = ca.jacobian(self.model.dynamics, self.model.control)

        self.A_func = ca.Function('A_fun', [self.model.state, self.model.control], [A_sx])
        self.B_func = ca.Function('B_fun', [self.model.state, self.model.control], [B_sx])

    def get_AB(self):
        self.build_linearization_funcs()

        x0 = np.asarray(self.p.xr, dtype=float).reshape((-1, 1))
        u0 = np.asarray(self.p.ur, dtype=float).reshape((-1, 1))

        self.A = np.array(self.A_func(x0, u0), dtype=float)
        self.B = np.array(self.B_func(x0, u0), dtype=float)
        return self.A, self.B

    def compute_K(self):
        A, B = self.get_AB()

        P = la.solve_continuous_are(A, B, self.p.Q, self.p.R)
        self.K = np.linalg.solve(self.p.R, B.T @ P)
        return self.K

    def control_law(self, x):
        du = -self.K @ (x - self.p.xr)
        return du