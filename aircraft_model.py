import numpy as np
import casadi as ca

class AircraftModel:
    def __init__(self, parameters):
        self.p = parameters

        u = ca.SX.sym('u', 1, 1)
        w = ca.SX.sym('w', 1, 1)
        q = ca.SX.sym('q', 1, 1)
        theta = ca.SX.sym('theta', 1, 1)

        x = ca.vertcat(u, w, q, theta)

        de = ca.SX.sym('de', 1, 1)
        u_in = ca.vertcat(de)

        du = u - self.p.xr[0]
        dw = w - self.p.xr[1]
        dq = q - self.p.xr[2]
        dde = de - self.p.ur

        Mu_tilde = self.p.Mu + self.p.Mwdot * self.p.Zu
        Mw_tilde = self.p.Mw + self.p.Mwdot * self.p.Zw
        Mq_tilde = self.p.Mq + self.p.Mwdot * self.p.ur
        Mde_tilde = self.p.Mde + self.p.Mwdot * self.p.Zde

        X = self.p.X0 + (self.p.Xu*du + self.p.Xw*dw + self.p.Xde*dde)
        Z = self.p.Z0 + (self.p.Zu*du + self.p.Zw*dw + self.p.Zde*dde)
        M = self.p.M0 + (Mu_tilde*du + Mw_tilde*dw + Mq_tilde*dq + Mde_tilde*dde)

        u_dot = X - self.p.g * ca.sin(theta) - q * w
        w_dot = Z + self.p.g * ca.cos(theta) + q * u
        q_dot = M
        theta_dot = q

        x_dot = ca.vertcat(u_dot, w_dot, q_dot, theta_dot)

        self.state = x
        self.control = u_in
        self.dynamics = x_dot

        self.f = ca.Function('f', [x, u_in], [x_dot], ['x', 'u'], ['x_dot'])
