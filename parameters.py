import numpy as np
import casadi as ca

class Parameters:
    def __init__(self):
        self.dt = 0.02
        self.sim_seconds = 20
        self.num_iterations = int(round(self.sim_seconds / self.dt))

        self.g = 32.2

        self.X0 = 0.0
        self.Z0 = -self.g
        self.M0 = 0.0

        self.Xu = -0.22
        self.Xw = 0.060
        self.Zu = -0.15
        self.Zw = -0.85
        self.Mu = 0.01
        self.Mw = -0.0095
        self.Mq = -0.89
        self.Mwdot = -0.00127

        self.Xde = 0.12
        self.Zde = 4.58
        self.Mde = 1.195

        self.x0 = np.array([190.0, 20.0, 1.0, 1.0], dtype=float).reshape((4,1))
        self.u0 = np.array([0.0], dtype=float).reshape((1,1))
        self.xr = np.array([202.56, 0.0, 0.0, 0.0], dtype=float).reshape((4,1))
        self.ur = np.array([0.0], dtype=float).reshape((1,1))

        self.Q = ca.diag([1/10**2, 1/5**2, 1/0.2**2, 1/0.087**2])
        self.R = ca.diag([1/0.175**2])


        self.de_max = np.deg2rad(25.0)
        self.de_min = -self.de_max