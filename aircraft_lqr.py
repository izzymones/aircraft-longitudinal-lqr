import numpy as np
import casadi as ca
import scipy.linalg as la

class LQRController:
    def __init__(self, params, model):
        self.p = params
        self.model = model
        self.K = None

        self._A_fun = None
        self._B_fun = None

    def _build_linearization_funs(self):
        A_sx = ca.jacobian(self.model.dynamics, self.model.state)
        B_sx = ca.jacobian(self.model.dynamics, self.model.control)

        self._A_fun = ca.Function('A_fun', [self.model.state, self.model.control], [A_sx])
        self._B_fun = ca.Function('B_fun', [self.model.state, self.model.control], [B_sx])

    def get_AB(self):
        if self._A_fun is None or self._B_fun is None:
            self._build_linearization_funs()

        x0 = np.asarray(self.p.xr, dtype=float).reshape((-1, 1))
        u0 = np.asarray(self.p.ur, dtype=float).reshape((-1, 1))

        A = np.array(self._A_fun(x0, u0), dtype=float)
        B = np.array(self._B_fun(x0, u0), dtype=float)
        return A, B

    def compute_K(self):
        A, B = self.get_AB()
        Q = np.asarray(self.p.Q, dtype=float)
        R = np.asarray(self.p.R, dtype=float)

        P = la.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)
        return self.K

    def control_law(self, state):
        x = np.asarray(state, dtype=float).reshape((-1, 1))
        xr = np.asarray(self.p.xr, dtype=float).reshape((-1, 1))

        if self.K is None:
            self.compute_K()

        du = -self.K @ (x - xr)
        return du