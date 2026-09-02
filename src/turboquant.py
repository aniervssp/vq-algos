from base import VectorQuantizer
import numpy as np 
import rotations
import math 

class TurboQuantMSE(VectorQuantizer):
    def __init__(self, dim: int, bit_width: int):
        super().__init__(dim, bit_width)
        self.pi = rotations.generate_rotation_matrix(dim)
        k = 1<<bit_width # number of clusters
        self.centroids = np.random.random(k) * 2 - 1 # for now just random in [-1, 1]
        # but in reality it's actually the solutions to equation (4) in the turboquant paper
        # missing constructing the codebook to find the centroids
        # ...

    def quantize(self, X: np.ndarray):
        _, d = X.shape

        assert d == self.dim 
        X = X @ self.pi 

        distances = np.abs(X[..., None] - self.centroids[None, None, :])
        idx = np.argmin(distances, axis = -1).astype(np.uint8)

        return idx 

    def dequantize(self, idx: np.ndarray[np.uint8]):
        X = self.centroids[idx]
        return X @ self.pi.T

class QJL:
    def __init__(self, dim):
        self.dim = dim 
        self.S = np.random.randn(dim, dim)

    def quantize(self, X: np.ndarray):
        return np.sign(X @ self.S)

    def dequantize(self, X: np.ndarray):
        return (X @ self.S.T) * math.sqrt(math.pi / 2) / self.dim
    

class TurboQuantProd(VectorQuantizer):
    def __init__(self, dim, bit_width):
        super().__init__(dim, bit_width)  
        self.quant_mse = TurboQuantMSE(dim, bit_width - 1)
        self.qjl = QJL(dim)
        
    def quantize(self, X: np.ndarray):
        idx = self.quant_mse.quantize(X)
        X_approx = self.quant_mse.dequantize(idx)
        r = X - X_approx
        qjl_codes = self.qjl.quantize(r)
        gamma = np.linalg.norm(r, axis=-1)
        return (idx, qjl_codes, gamma)

    def dequantize(self, idx, qjl, gamma):
        X_mse = self.quant_mse.dequantize(idx)
        X_qjl = self.qjl.dequantize(qjl) * gamma[:, np.newaxis]
        return X_mse + X_qjl