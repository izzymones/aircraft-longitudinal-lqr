import numpy as np
import casadi as ca

class Parameters:
    def __init__(self):
        self.dt = 0.02
        self.sim_seconds = 20
        self.num_iterations = int(round(self.sim_seconds / self.dt))

        self.g = 32.2

        self.u0 = 202.56
        self.w0 = 0.0
        self.q0 = 0.0
        self.theta0 = 0.0
        self.de0 = 0.0

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

        self.xr = np.array([self.u0, self.w0, self.q0, self.theta0], dtype=float).reshape((4,1))
        self.ur = np.array([self.de0], dtype=float).reshape((1,1))

        self.Q = ca.diag([1.0, 1.0, 1.0, 1.0])
        self.R = ca.diag([1.0])


        self.de_max = np.deg2rad(25.0)
        self.de_min = -self.de_max