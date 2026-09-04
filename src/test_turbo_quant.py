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
    idx = quantizer.quantize(X)
    assert idx.shape == (n, dim)
    assert idx.dtype == np.uint8

    X_approx = quantizer.dequantize(idx)
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
    idx, qjl_codes, gamma = quantizer.quantize(X)

    assert idx.shape == (n, dim)
    assert qjl_codes.shape == (n, dim)
    assert gamma.shape == (n,)

    X_approx = quantizer.dequantize(idx, qjl_codes, gamma)
    assert X_approx.shape == (n, dim)


def test_eval_quant_model_uses_reproducible_seed_averaged_worst_case():
    class StubQuantizer:
        def __init__(self, dim, bit_width):
            self.offset = np.random.random()

        def quantize(self, X):
            return X

        def dequantize(self, X):
            return X + self.offset

    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    bit_widths = np.array([1, 2])
    seeds = (1, 2)

    first = eval_quant_model(X, bit_widths, StubQuantizer, seeds)
    second = eval_quant_model(X, bit_widths, StubQuantizer, seeds)

    np.testing.assert_equal(first, second)
    assert first["mse_distortions"].shape == (len(bit_widths),)
    assert first["inner_prod_errors"].shape == (len(bit_widths),)

    offsets = [
        np.random.RandomState(seed).random_sample(2)
        for seed in seeds
    ]
    expected_mse = np.mean([2 * seed_offsets**2 for seed_offsets in offsets], axis=0)
    np.testing.assert_allclose(first["mse_distortions"], expected_mse)

    inner_prods = X @ X.T
    expected_inner_prod_errors = []
    for bit_width in range(len(bit_widths)):
        seed_errors = []
        for seed_offsets in offsets:
            X_approx = X + seed_offsets[bit_width]
            errors_by_row = np.sum((inner_prods - X @ X_approx.T) ** 2, axis=-1)
            seed_errors.append(np.max(errors_by_row))
        expected_inner_prod_errors.append(np.mean(seed_errors))
    np.testing.assert_allclose(
        first["inner_prod_errors"], expected_inner_prod_errors
    )


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
