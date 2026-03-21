import numpy as np

from parameters import Parameters
from aircraft_lqr import AircraftLQR
from aircraft_model import AircraftModel

class Aircraft_Analysis:
    def __init__(self):
        self.p = Parameters()
        self.model = AircraftModel(self.p)
        self.lqr = AircraftLQR(self.p, self.model)
        self.lqr.compute_K()

    def get_open_loop_eigs(self):
        A, B = self.lqr.get_AB()
        eigvals = np.linalg.eigvals(A)
        return eigvals

    def get_closed_loop_eigs(self):
        A, B = self.lqr.get_AB()
        K = self.lqr.K
        Acl = A - B @ K
        eigvals = np.linalg.eigvals(Acl)
        return eigvals

    def get_closed_loop_matrix(self):
        A, B = self.lqr.get_AB()
        K = self.lqr.K
        Acl = A - B @ K
        return Acl

if __name__ == "__main__":
    analysis = Aircraft_Analysis()

    A, B = analysis.lqr.get_AB()
    K = analysis.lqr.K
    Acl = analysis.get_closed_loop_matrix()

    open_loop_eigs = analysis.get_open_loop_eigs()
    closed_loop_eigs = analysis.get_closed_loop_eigs()

    print("A:\n", A)
    print("B:\n", B)
    print("K:\n", K)
    print("A - BK:\n", Acl)
    print("Open-loop eigenvalues:\n", open_loop_eigs)
    print("Closed-loop eigenvalues:\n", closed_loop_eigs)