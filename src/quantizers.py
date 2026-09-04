from base import VectorQuantizer
import numpy as np
import rotations
import math
from scipy.integrate import quad
from scipy.special import gamma
import functools
from typing import NamedTuple

import packing

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

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """-> (n, ceil(bit_width * dim / 8)) uint8, packed bit_width bits per coordinate."""
        _, d = X.shape

        assert d == self.dim
        X = X @ self.pi

        distances = np.abs(X[..., None] - self.centroids[None, None, :])
        # uint8 only to feed np.packbits; the stored codes are bit_width bits wide
        idx = np.argmin(distances, axis=-1).astype(np.uint8)

        return packing.pack_codes(idx, self.bit_width)

    def dequantize(self, packed: np.ndarray) -> np.ndarray:
        idx = packing.unpack_codes(packed, self.bit_width, self.dim)
        return self.centroids[idx] @ self.pi.T


class QJL(VectorQuantizer):
    def __init__(self, dim, bit_width: int = 1):
        super().__init__(dim, bit_width)
        self.S = np.random.randn(dim, dim)

    def quantize(self, X: np.ndarray):
        return np.sign(X @ self.S)

    def dequantize(self, X: np.ndarray):
        return (X @ self.S.T) * math.sqrt(math.pi / 2) / self.dim


class ProdCodes(NamedTuple):
    """Packed output of TurboQuantProd -- Algorithm 2 line 8, (idx, qjl, ||r||_2)."""

    idx: np.ndarray  # packed (bit_width - 1)-bit codebook indices
    signs: np.ndarray  # packed 1-bit QJL signs
    norms: np.ndarray  # (n, 1) float32 residual norms

    @property
    def nbytes(self) -> int:
        return self.idx.nbytes + self.signs.nbytes + self.norms.nbytes


class TurboQuantProd(VectorQuantizer):
    def __init__(self, dim, bit_width):
        super().__init__(dim, bit_width)
        self.quant_mse = TurboQuantMSE(dim, bit_width - 1)
        self.qjl = QJL(dim, 1)

    def quantize(self, X: np.ndarray) -> ProdCodes:
        idx = self.quant_mse.quantize(X)
        r = X - self.quant_mse.dequantize(idx)

        return ProdCodes(
            idx=idx,
            # QJL signs are +-1; np.sign can emit 0, which packs as -1 (never seen
            # on continuous data, and harmless -- it is still a valid sign vector)
            signs=packing.pack_signs(self.qjl.quantize(r)),
            norms=np.linalg.norm(r, axis=-1, keepdims=True).astype(np.float32),
        )

    def dequantize(self, codes: ProdCodes) -> np.ndarray:
        X_mse = self.quant_mse.dequantize(codes.idx)
        signs = packing.unpack_signs(codes.signs, self.dim)
        return X_mse + self.qjl.dequantize(signs) * codes.norms
