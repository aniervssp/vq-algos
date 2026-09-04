from base import VectorQuantizer
import numpy as np
import rotations
import math
from scipy.integrate import quad
from scipy.special import gamma
import functools 

def max_lloyd(pdf_func, k: int, low: float, high: float, max_iter: int = 5000, tol: float = 1e-6):
    boundaries = np.linspace(low, high, k + 1)

    centroids = (boundaries[1:] + boundaries[:-1]) / 2
    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        for i in range(k):
            a = boundaries[i]
            b = boundaries[i + 1]
            centroids[i] = (
                quad(lambda x: x * pdf_func(x), a, b)[0] / quad(pdf_func, a, b)[0]
            )

        boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
        boundaries[0] = low
        boundaries[-1] = high
        
        if np.max(np.abs(centroids - prev_centroids)) < tol:
            break 

    return centroids

@functools.cache
def sphere_codebook(dim: int, bit_width: int) -> np.ndarray:
    """Lloyd-Max codebook for one coordinate of a random unit vector in R^dim.

    Deterministic in (dim, bit_width), so it is shared across evaluation seeds.
    The returned array is read-only because every caller holds the same object.
    """
    pdf = lambda x: (
        gamma(dim / 2)
        / (math.sqrt(math.pi) * gamma((dim - 1) / 2))
        * (1 - x * x) ** ((dim - 3) / 2)
    )
    centroids = max_lloyd(pdf, 1 << bit_width, -1.0, 1.0)
    centroids.flags.writeable = False
    return centroids

class TurboQuantMSE(VectorQuantizer):
    def __init__(self, dim: int, bit_width: int):
        super().__init__(dim, bit_width)
        self.pi = rotations.generate_rotation_matrix(dim)
        self.centroids = sphere_codebook(dim, bit_width)

    def quantize(self, X: np.ndarray):
        _, d = X.shape

        assert d == self.dim
        X = X @ self.pi

        distances = np.abs(X[..., None] - self.centroids[None, None, :])
        idx = np.argmin(distances, axis=-1).astype(np.uint8)

        return idx

    def dequantize(self, idx: np.ndarray):
        X = self.centroids[idx]
        return X @ self.pi.T


class QJL(VectorQuantizer):
    def __init__(self, dim, bit_width: int = 1):
        super().__init__(dim, bit_width)
        self.S = np.random.randn(dim, dim)

    def quantize(self, X: np.ndarray):
        return np.sign(X @ self.S)

    def dequantize(self, X: np.ndarray):
        return (X @ self.S.T) * math.sqrt(math.pi / 2) / self.dim


class TurboQuantProd(VectorQuantizer):
    def __init__(self, dim, bit_width):
        super().__init__(dim, bit_width)
        self.quant_mse = TurboQuantMSE(dim, bit_width - 1)
        self.qjl = QJL(dim, 1)

    def quantize(self, X: np.ndarray):
        idx = self.quant_mse.quantize(X)
        X_approx = self.quant_mse.dequantize(idx)
        r = X - X_approx
        qjl_codes = self.qjl.quantize(r)
        gamma = np.linalg.norm(r, axis=-1, keepdims=True)

        result = np.concat([idx, qjl_codes, gamma], axis=-1)

        return result

    def dequantize(self, codes: np.ndarray):
        idx = codes[..., :self.dim].astype(np.uint8)
        qjl = codes[..., self.dim : self.dim * 2]
        gamma = codes[..., self.dim*2:]
        X_mse = self.quant_mse.dequantize(idx)
        X_qjl = self.qjl.dequantize(qjl) * gamma
        return X_mse + X_qjl
