import numpy as np
import casadi as ca

class AircraftModel:
    def __init__(self, mc):
        self.mc = mc

        u = ca.SX.sym('u', 1, 1)
        w = ca.SX.sym('w', 1, 1)
        q = ca.SX.sym('q', 1, 1)
        theta = ca.SX.sym('theta', 1, 1)

        x = ca.vertcat(u, w, q, theta)

        de = ca.SX.sym('de', 1, 1)
        u_in = ca.vertcat(de)

        du = u - mc.u0
        dw = w - mc.w0
        dq = q - mc.q0
        dde = de - mc.de0

        Mu_tilde = mc.Mu + mc.Mwdot * mc.Zu
        Mw_tilde = mc.Mw + mc.Mwdot * mc.Zw
        Mq_tilde = mc.Mq + mc.Mwdot * mc.u0
        Mde_tilde = mc.Mde + mc.Mwdot * mc.Zde

        X = mc.Xu * du + mc.Xw * dw + mc.Xde * dde
        Z = mc.Zu * du + mc.Zw * dw + mc.Zde * dde
        M = Mu_tilde * du + Mw_tilde * dw + Mq_tilde * dq + Mde_tilde * dde

        u_dot = X - mc.g * ca.sin(theta) - q * w
        w_dot = Z + mc.g * ca.cos(theta) + q * u
        q_dot = M
        theta_dot = q

        x_dot = ca.vertcat(u_dot, w_dot, q_dot, theta_dot)

        self.state = x
        self.control = u_in
        self.dynamics = x_dot

        self.f = ca.Function('f', [x, u_in], [x_dot], ['x', 'u'], ['x_dot'])


if __name__ == "__main__":


    mc = MC()
    model = LongitudinalModel(mc)

    xr = np.array([mc.u0, mc.w0, mc.q0, mc.theta0], dtype=float)
    ur = np.array([mc.de0], dtype=float)

    print_AB(model, xr, ur)