import numpy as np


def generate_rotation_matrix(dim: int):
    """
    Generate a random rotation matrix of size (dim, dim) by taking a random matrix and performing QR decomposition on it.
    """

    random_matrix = np.random.randn(dim, dim)

    q, r = np.linalg.qr(random_matrix)

    return q * np.sign(np.diag(r))
