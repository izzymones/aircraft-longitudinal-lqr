import numpy as np
import matplotlib.pyplot as plt

class AircraftPlotter():
    def __init__(self, fcs, dataLog):
        self.fcs = fcs
        self.data = dataLog.data

    def plot_state(self, key: str, title='State'):
        x = self.data[key]
        num_iterations = len(x)
        tspan = self.fcs.p.dt * np.arange(num_iterations)

        fig, axs = plt.subplots(4)
        fig.set_figheight(8)
        fig.suptitle(title)

        axs[0].plot(tspan, x[:,0])
        axs[0].set_ylabel('$u$ (ft/s)')

        axs[1].plot(tspan, x[:,1])
        axs[1].set_ylabel('$w$ (ft/s)')

        axs[2].plot(tspan, x[:,2])
        axs[2].set_ylabel('$q$ (rad/s)')

        axs[3].plot(tspan, x[:,3])
        axs[3].set_ylabel(r'$\theta$ (rad)')

        plt.xlabel('Time (s)')

    def plot_control(self, key: str, title='Control'):
        u = self.data[key]
        num_iterations = len(u)
        tspan = self.fcs.p.dt * np.arange(num_iterations)
        fig, ax = plt.subplots(1)
        fig.set_figheight(4)
        fig.suptitle(title)

        ax.plot(tspan, u[:,0])
        ax.set_ylabel(r'$\delta_e$ (rad)')
        plt.xlabel('Time (s)')
        plt.ioff()
        plt.show()
