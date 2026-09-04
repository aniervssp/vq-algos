"""Bit-level primitives for quantizer codes.

The quantizers emit exactly the b bits per coordinate that the paper's bit-width
refers to, and these are the routines they use to get there. numpy's packbits
works on uint8, so a uint8 array is the intermediate the packer consumes -- it is
never the stored representation.

Round-tripping is exact: `unpack_codes(pack_codes(c, b), b, d)` returns `c`.
"""

import numpy as np


def pack_codes(codes: np.ndarray, bit_width: int) -> np.ndarray:
    """(n, d) uint8 with values < 2**bit_width -> (n, ceil(bit_width*d/8)) uint8."""
    n, d = codes.shape
    if bit_width == 0:  # a single centroid carries no information
        return np.zeros((n, 0), dtype=np.uint8)
    bits = np.unpackbits(codes, axis=1, bitorder="little").reshape(n, d, 8)
    return np.packbits(
        bits[:, :, :bit_width].reshape(n, bit_width * d), axis=1, bitorder="little"
    )


def unpack_codes(packed: np.ndarray, bit_width: int, dim: int) -> np.ndarray:
    """Inverse of `pack_codes`."""
    n = packed.shape[0]
    if bit_width == 0:
        return np.zeros((n, dim), dtype=np.uint8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, : bit_width * dim]
    bits = bits.reshape(n, dim, bit_width)
    bits = np.concatenate([bits, np.zeros((n, dim, 8 - bit_width), np.uint8)], axis=2)
    return np.packbits(bits.reshape(n, dim * 8), axis=1, bitorder="little")


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """(n, d) of -1/+1 -> one bit per entry."""
    return pack_codes((signs > 0).astype(np.uint8), 1)


def unpack_signs(packed: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of `pack_signs`, as float64 -1/+1."""
    return unpack_codes(packed, 1, dim).astype(np.float64) * 2.0 - 1.0
