import casadi as ca

class LongitudinalModel:
    def __init__(self, mc):
        self.mc = mc

        u = ca.SX.sym('u')
        w = ca.SX.sym('w')
        q = ca.SX.sym('q')
        theta = ca.SX.sym('theta')

        x = ca.vertcat(u, w, q, theta)

        de = ca.SX.sym('de')
        u_in = ca.vertcat(de)

        u0 = mc.u0
        w0 = mc.w0 if hasattr(mc, 'w0') else 0.0
        q0 = mc.q0 if hasattr(mc, 'q0') else 0.0
        theta0 = mc.theta0 if hasattr(mc, 'theta0') else 0.0
        de0 = mc.de0 if hasattr(mc, 'de0') else 0.0

        du = u - u0
        dw = w - w0
        dq = q - q0
        dde = de - de0

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