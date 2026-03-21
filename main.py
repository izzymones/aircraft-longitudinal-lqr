import numpy as np
from data import DataLog
from parameters import Parameters
from aircraft_lqr import AircraftLQR
from aircraft_model import AircraftModel
from aircraft_sim import AircraftSim
from aircraft_plotter import AircraftPlotter

class FlightControlSystem:
    def __init__(self):
        self.p = Parameters()
        self.model = AircraftModel(self.p)
        self.lqr = AircraftLQR(self.p, self.model)
        self.lqr.compute_K()
        self.sim = AircraftSim(self.model, self.p.dt, self.lqr, self.p)

    def control(self, x):
        dde = self.lqr.control_law(x)
        de = self.p.ur + dde
        de = np.minimum(de, self.p.de_max)
        de = np.maximum(de, self.p.de_min)
        return de

    def run_controller(self):
        x = np.asarray(self.p.x0, dtype=float).reshape((4,1))
        N = int(self.p.num_iterations)

        data = DataLog()
        data.allocate_data("time",   N + 1, 1)
        data.allocate_data("state",  N + 1, 4)
        data.allocate_data("control",N + 1, 1)

        data.add_point("time", 0, [0.0])
        data.add_point("state", 0, x)
        data.add_point("control", 0, self.p.ur)

        for k in range(N):
            de = self.control(x)
            # de = np.deg2rad(1.0)
            x = self.sim.step_linear(x, de)
            # x = self.sim.step(x, de)

            data.add_point("time", k + 1, [(k + 1) * self.p.dt])
            data.add_point("state", k + 1, x)
            data.add_point("control", k + 1, de)

        return data

if __name__ == "__main__":
    fcs = FlightControlSystem()
    data = fcs.run_controller()

    print("A:\n", fcs.lqr.A)
    print("B:\n", fcs.lqr.B)
    print("K:\n", fcs.lqr.K)
    print("final state:", data.data["state"][-1])

    plotter = AircraftPlotter(fcs.model, data)
    plotter.plot_state("state")
    plotter.plot_control("control")