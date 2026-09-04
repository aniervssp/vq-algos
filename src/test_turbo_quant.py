import numpy as np

from evals import eval_quant_model
from quantizers import QJL, TurboQuantMSE, TurboQuantProd

# testing the code runs, no logic tested


def test_turbo_quant_mse():
    dim = 16
    bit_width = 2
    n = 10
    X = np.random.randn(n, dim)

    quantizer = TurboQuantMSE(dim, bit_width)
    codes = quantizer.quantize(X)
    assert codes.shape == (n, -(-bit_width * dim // 8))  # bit_width bits per coordinate
    assert codes.dtype == np.uint8

    X_approx = quantizer.dequantize(codes)
    assert X_approx.shape == (n, dim)


def test_qjl():
    dim = 16
    n = 10
    X = np.random.randn(n, dim)

    qjl = QJL(dim)
    codes = qjl.quantize(X)
    assert codes.shape == (n, dim)

    X_approx = qjl.dequantize(codes)
    assert X_approx.shape == (n, dim)


def test_turbo_quant_prod():
    dim = 16
    bit_width = 3
    n = 10
    X = np.random.randn(n, dim)

    quantizer = TurboQuantProd(dim, bit_width)
    codes = quantizer.quantize(X)

    assert codes.idx.shape == (n, -(-(bit_width - 1) * dim // 8))
    assert codes.signs.shape == (n, -(-dim // 8))
    assert codes.norms.shape == (n, 1)
    assert codes.nbytes == codes.idx.nbytes + codes.signs.nbytes + codes.norms.nbytes

    X_approx = quantizer.dequantize(codes)
    assert X_approx.shape == (n, dim)

if __name__ == "__main__":
    print("Starting to test...\n")
    print("Testing QJL...")
    test_qjl()
    print("OK!")
    print("Testing Quant MSE...")
    test_turbo_quant_mse()
    print("OK!")
    print("Testing Quant Prod...")
    test_turbo_quant_prod()
    print("OK!")
    print("\nAll good!!!")
