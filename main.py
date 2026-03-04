import numpy as np

from parameters import Parameters
from aircraft_lqr import AircraftLQR
from aircraft_model import AircraftModel
from aircraft_sim import AircraftSim
from aircraft_plotter import AircraftPlotter

class DataLog:
    def __init__(self):
        self.data = {}

class FlightControlSystem:
    def __init__(self):
        self.p = Parameters()
        self.model = AircraftModel(self.p)
        self.lqr = AircraftLQR(self.p, self.model)
        self.lqr.compute_K()

        self.sim = AircraftSim(self.model, self.p.dt)

    def control(self, x):
        dde = self.lqr.control_law(x)
        de = self.p.ur + dde

        de = np.minimum(de, self.p.de_max)
        de = np.maximum(de, self.p.de_min)

        return de

    def run_controller(self):
        x = self.p.x0
        N = int(self.p.num_iterations)

        t_hist = np.zeros((N + 1, 1))
        x_hist = np.zeros((N + 1, x.shape[0]))
        u_hist = np.zeros((N + 1, 1))

        x_hist[0, :] = x.flatten()

        for k in range(N):
            de = self.control(x)
            x = self.sim.step(x, de)

            t_hist[k + 1, 0] = (k + 1) * self.p.dt
            x_hist[k + 1, :] = x.flatten()
            u_hist[k + 1, 0] = float(de[0, 0])

        log = DataLog()
        log.data['t'] = t_hist
        log.data['x'] = x_hist
        log.data['u'] = u_hist

        return log


if __name__ == "__main__":
    fcs = FlightControlSystem()
    log = fcs.run_controller()

    print("A:\n", fcs.lqr.A)
    print("B:\n", fcs.lqr.B)
    print("K:\n", fcs.lqr.K)
    print("final state:", log.data['x'][-1])

    plotter = AircraftPlotter(fcs, log)
    plotter.plot_state('x', title='States')
    plotter.plot_control('u', title='Elevator')