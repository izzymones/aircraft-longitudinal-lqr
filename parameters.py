import numpy as np
import casadi as ca

class Parameters:
    def __init__(self):
        self.dt = 0.02
        self.sim_seconds = 100
        self.num_iterations = int(round(self.sim_seconds / self.dt))
        self.g = 32.2
        self.m = 16300 / self.g

        V = 0.257 * 1116.4
        self.x0 = np.array([V, 0.0, 0.0, 0.1], dtype=float).reshape((4,1))
        self.u0 = np.array([0.0], dtype=float).reshape((1,1))
        self.xr = np.array([V, 0.0, 0.0, 0.0], dtype=float).reshape((4,1))
        self.ur = np.array([0.0], dtype=float).reshape((1,1))

        self.g = 32.2

        self.X0 = 0.0
        self.Z0 = -self.g
        self.M0 = 0.0

        self.Ixx = 3549
        self.Iyy = 58611
        self.Izz = 59669

        tau = 0.5
        eta = 0.95
        St = 50

        Q = 0.5 * 0.002377 * V**2
        S = 196.1
        b = 21.94
        c = 9.55

        CD0 = 0.263
        CDalpha = 0.45
        CDu = 0.0

        CL0 = 0.735
        CLalpha = 3.44

        Cmalpha = -0.64
        Cmq = -5.8
        Cmadot = -1.6
        CLalphat = 3

        Czde = - CLalphat * tau * eta * St / S
        Cmde = -1.46

        self.Xu = -((CDu+2*CD0)*Q*S)/(self.m*self.x0[0])
        self.Xw = -((CDalpha - CL0)*Q*S)/(self.m*self.x0[0])
        self.Zu = -((2*CL0)*Q*S)/(self.m*self.x0[0])
        self.Zw = -((CLalpha+CD0)*Q*S)/(self.m*self.x0[0])
        self.Mu = 0.0
        self.Mw = (Cmalpha*S*c*Q)/(self.x0[0]*self.Iyy)
        self.Mq = ((Cmq*c*Q*S*c)/(2*self.x0[0]))/(self.Iyy)
        self.Mwdot = (Cmadot*c*Q*S*c)/(2*self.x0[0]**2*self.Iyy)

        self.Xde = 0.0
        self.Zde = -Czde*Q*S/self.m
        self.Mde = (Cmde*Q*S*c)/self.Iyy

        self.Q = ca.diag([
            0.01,
            0.1,
            10.0,
            10.0
        ])

        self.R = ca.diag([
            100
        ])

        self.de_max = np.deg2rad(25.0)
        self.de_min = -self.de_max