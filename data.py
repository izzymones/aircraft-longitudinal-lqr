import numpy as np

class DataLog:
    def __init__(self):
        self.data = {}
        self._dims = {}

    def allocate_data(self, name: str, N: int, dim: int):
        self.data[name] = np.zeros((N, dim), dtype=float)
        self._dims[name] = dim

    def add_point(self, name: str, k: int, x):
        arr = np.asarray(x, dtype=float).reshape((-1,))
        dim = self._dims[name]
        self.data[name][k, :] = arr