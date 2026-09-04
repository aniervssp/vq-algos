import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from quantizers import TurboQuantMSE, TurboQuantProd

RANDOM_SEED = 42
EVALUATION_SEEDS = (42, 43, 44, 45, 46)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DATA_CACHE = CACHE_DIR / "glove_embeddings.npy"
SAMPLE_CACHE = CACHE_DIR / "glove_sample_5000.npy"
FIGURE_BIT_WIDTHS = (1, 2, 3, 4)
BUCKETED_BIT_WIDTH = 2
INNER_PRODUCT_BUCKETS = 4
HISTOGRAM_BINS = 120

def eval_quant_model(X, bit_widths, model, seeds=EVALUATION_SEEDS):
    d = X.shape[-1]

    per_seed_metrics = []
    inner_prods = X @ X.T
    off_diag = ~np.eye(X.shape[0], dtype=bool)

    for seed in seeds:
        np.random.seed(seed)
        mse_distortions = []
        inner_prod_mult_bias = []
        inner_prod_errors = []

        for b in bit_widths:
            quant_model = model(dim=d, bit_width=b)
            idx = quant_model.quantize(X)

            X_approx = quant_model.dequantize(idx)

            vector_mse = np.sum((X - X_approx) ** 2, axis=-1)
            mse_distortions.append(np.mean(vector_mse))

            inner_prods_approx = X @ X_approx.T
            sq_errors = (inner_prods - inner_prods_approx) ** 2

            inner_prod_mult_bias.append(
                np.mean(inner_prods_approx[off_diag]) / np.mean(inner_prods[off_diag])
            )
            inner_prod_errors.append(np.mean(sq_errors[off_diag]))

        per_seed_metrics.append(
            [mse_distortions, inner_prod_mult_bias, inner_prod_errors]
        )

    return {
        "mse_distortions": np.mean([metrics[0] for metrics in per_seed_metrics], axis=0),
        "inner_prod_mult_bias": np.mean(
            [metrics[1] for metrics in per_seed_metrics], axis=0
        ),
        "inner_prod_errors": np.mean(
            [metrics[2] for metrics in per_seed_metrics], axis=0
        ),
    }


def make_plots(bit_widths, metrics_mse: dict, metrics_prod: dict, upperbound_mse, lowerbound_mse, upperbound_inner_prod, lowerbound_inner_prod, d):
    fig, (mse_ax, bias_ax, inner_prod_ax) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evaluation of Quantization Models", fontsize=16)

    mse_ax.plot(bit_widths, metrics_mse["mse_distortions"], label="TurboQuantMSE", marker="o")
    mse_ax.plot(bit_widths, metrics_prod["mse_distortions"], label="TurboQuantProd", marker="o")
    mse_ax.plot(bit_widths, upperbound_mse(bit_widths, d), label="Upper Bound", linestyle="--", color="gray")
    mse_ax.plot(bit_widths, lowerbound_mse(bit_widths, d), label="Lower Bound", linestyle="--", color="black")
    mse_ax.legend()
    mse_ax.set_xlabel("Bit-widths")
    mse_ax.set_ylabel("MSE Distortions")
    mse_ax.set_title("MSE Distortions vs Bit-widths")
    mse_ax.set_yscale("log")

    bias_ax.plot(bit_widths, metrics_mse["inner_prod_mult_bias"], label="TurboQuantMSE", marker="o")
    bias_ax.plot(bit_widths, metrics_prod["inner_prod_mult_bias"], label="TurboQuantProd", marker="o")
    bias_ax.legend()
    bias_ax.set_xlabel("Bit-widths")
    bias_ax.set_ylabel("Inner Product Multiplicative Bias")
    bias_ax.set_title("Inner Product Multiplicative Bias vs Bit-widths")
    bias_ax.set_yscale("log")


    inner_prod_ax.plot(bit_widths, metrics_mse["inner_prod_errors"], label="TurboQuantMSE", marker="o")
    inner_prod_ax.plot(bit_widths, metrics_prod["inner_prod_errors"], label="TurboQuantProd", marker="o")
    inner_prod_ax.plot(bit_widths, upperbound_inner_prod(bit_widths, d), label="Upper Bound", linestyle="--", color="gray")
    inner_prod_ax.plot(bit_widths, lowerbound_inner_prod(bit_widths, d), label="Lower Bound", linestyle="--", color="black")
    inner_prod_ax.legend()
    inner_prod_ax.set_xlabel("Bit-widths")
    inner_prod_ax.set_ylabel("Inner Product Error")
    inner_prod_ax.set_title("Inner Product Error vs Bit-widths")
    inner_prod_ax.set_yscale("log")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("eval_turboquant.png")



def _fit_quantizer(X, model, bit_width, seed=RANDOM_SEED):
    np.random.seed(seed)
    quant_model = model(dim=X.shape[-1], bit_width=bit_width)
    return quant_model.dequantize(quant_model.quantize(X))


def _iter_error_blocks(X, inner_prods, X_approx, block=512):
    """Stream (true inner product, signed estimation error) over off-diagonal pairs.

    Blocked so that the n x n error matrix is never materialised at once.
    """
    n = X.shape[0]
    for start in range(0, n, block):
        stop = min(start + block, n)
        true_ip = inner_prods[start:stop]
        errors = true_ip - X[start:stop] @ X_approx.T

        keep = np.ones(errors.shape, dtype=bool)
        rows = np.arange(stop - start)
        keep[rows, start + rows] = False  # drop self-pairs, see README

        yield true_ip[keep], errors[keep]


def _error_range(X, inner_prods, approxes, width=4.0):
    """Bin range wide enough for every approximation, so panels stay comparable."""
    low, high = np.inf, -np.inf
    for X_approx in approxes:
        _, errors = next(_iter_error_blocks(X, inner_prods, X_approx))
        mean, std = errors.mean(), errors.std()
        low = min(low, mean - width * std)
        high = max(high, mean + width * std)
    return low, high


def _error_histogram(X, inner_prods, X_approx, bins):
    """Histogram counts plus the exact mean error (not clipped to the bin range)."""
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total, n_pairs = 0.0, 0
    for _, errors in _iter_error_blocks(X, inner_prods, X_approx):
        counts += np.histogram(errors, bins=bins)[0]
        total += errors.sum()
        n_pairs += errors.size
    return counts, total / n_pairs


def _inner_product_bucket_edges(inner_prods, n_buckets, sample=2_000_000, seed=RANDOM_SEED):
    """Quantile edges of the off-diagonal inner-product distribution."""
    rng = np.random.default_rng(seed)
    n = inner_prods.shape[0]
    i = rng.integers(0, n, size=sample)
    j = rng.integers(0, n, size=sample)
    keep = i != j
    return np.quantile(inner_prods[i[keep], j[keep]], np.linspace(0.0, 1.0, n_buckets + 1))


def _bucketed_error_histogram(X, inner_prods, X_approx, bins, edges):
    """Per-bucket histogram counts, mean true inner product, and error mean/std."""
    n_buckets = len(edges) - 1
    counts = np.zeros((n_buckets, len(bins) - 1), dtype=np.int64)
    ip_sum = np.zeros(n_buckets)
    err_sum = np.zeros(n_buckets)
    err_sq_sum = np.zeros(n_buckets)
    sizes = np.zeros(n_buckets, dtype=np.int64)

    for true_ip, errors in _iter_error_blocks(X, inner_prods, X_approx):
        which = np.clip(np.searchsorted(edges, true_ip, side="right") - 1, 0, n_buckets - 1)
        for k in range(n_buckets):
            selected = errors[which == k]
            counts[k] += np.histogram(selected, bins=bins)[0]
            ip_sum[k] += true_ip[which == k].sum()
            err_sum[k] += selected.sum()
            err_sq_sum[k] += np.square(selected).sum()
            sizes[k] += selected.size

    means = err_sum / sizes
    return counts, ip_sum / sizes, means, np.sqrt(err_sq_sum / sizes - means**2)


def make_error_distribution_plots(X, bit_widths=FIGURE_BIT_WIDTHS, seed=RANDOM_SEED):
    """Paper Fig. 1 -- distribution of the signed inner-product estimation error.

    TurboQuantProd stays centred on zero at every bit-width (Theorem 2's
    unbiasedness claim); TurboQuantMSE is visibly shifted, and only recentres as
    its multiplicative bias shrinks with b.
    """
    inner_prods = X @ X.T
    models = (("TurboQuantProd", TurboQuantProd), ("TurboQuantMSE", TurboQuantMSE))

    fig, axes = plt.subplots(
        len(models), len(bit_widths), figsize=(4 * len(bit_widths), 3.4 * len(models))
    )
    fig.suptitle("Inner-Product Estimation Error Distribution", fontsize=16)

    for col, b in enumerate(bit_widths):
        approxes = [_fit_quantizer(X, model, b, seed) for _, model in models]
        bins = np.linspace(*_error_range(X, inner_prods, approxes), HISTOGRAM_BINS + 1)

        for row, ((name, _), X_approx) in enumerate(zip(models, approxes)):
            counts, mean = _error_histogram(X, inner_prods, X_approx, bins)

            ax = axes[row, col]
            ax.stairs(counts, bins, fill=True, alpha=0.75, color=f"C{row}")
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.axvline(mean, color="crimson", linestyle="--", linewidth=1.2,
                       label=f"mean = {mean:+.4f}")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(f"{name}, b = {b}", fontsize=11)
            ax.set_xlabel("Inner product error")
            if col == 0:
                ax.set_ylabel("Frequency")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("error_distribution.png", dpi=120)


def make_error_vs_inner_product_plots(
    X, bit_width=BUCKETED_BIT_WIDTH, n_buckets=INNER_PRODUCT_BUCKETS, seed=RANDOM_SEED
):
    """Paper Fig. 2 -- error spread against the true inner product, at fixed bit-width.

    TurboQuantProd's error is pure variance, so it stays put across buckets;
    TurboQuantMSE's bias is multiplicative in <y,x>, so its error drifts and widens
    as the true inner product grows.
    """
    inner_prods = X @ X.T
    models = (("TurboQuantProd", TurboQuantProd), ("TurboQuantMSE", TurboQuantMSE))
    approxes = [_fit_quantizer(X, model, bit_width, seed) for _, model in models]

    edges = _inner_product_bucket_edges(inner_prods, n_buckets)
    bins = np.linspace(*_error_range(X, inner_prods, approxes), HISTOGRAM_BINS + 1)

    fig, axes = plt.subplots(
        len(models), n_buckets, figsize=(4 * n_buckets, 3.4 * len(models)), sharex=True
    )
    fig.suptitle(
        f"Inner-Product Error vs Size of the True Inner Product (b = {bit_width})",
        fontsize=16,
    )

    for row, ((name, _), X_approx) in enumerate(zip(models, approxes)):
        counts, ip_means, means, stds = _bucketed_error_histogram(
            X, inner_prods, X_approx, bins, edges
        )
        for col in range(n_buckets):
            ax = axes[row, col]
            ax.stairs(counts[col], bins, fill=True, alpha=0.75, color=f"C{row}")
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.axvline(means[col], color="crimson", linestyle="--", linewidth=1.2)
            ax.set_title(
                f"{name}\nAvg IP = {ip_means[col]:.2f}   "
                f"mean = {means[col]:+.4f}   std = {stds[col]:.4f}",
                fontsize=9,
            )
            if row == len(models) - 1:
                ax.set_xlabel("Inner product error")
            if col == 0:
                ax.set_ylabel("Frequency")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("error_vs_inner_product.png", dpi=120)


def measure_memory(X, bit_widths, metrics_mse, metrics_prod, seed=RANDOM_SEED):
    """Storage footprint of the packed codes that `quantize` now emits.

    Both quantizers return exactly bit_width bits per coordinate (TurboQuantProd
    plus one float32 residual norm per vector), so `.nbytes` is the real number --
    no separate packing step to remember.
    """
    n, d = X.shape
    raw_bytes = 4 * d  # float32 baseline, per vector
    rows = []

    for i, b in enumerate(bit_widths):
        np.random.seed(seed)
        quant_mse = TurboQuantMSE(dim=d, bit_width=b)
        codes_mse = quant_mse.quantize(X)

        np.random.seed(seed)
        quant_prod = TurboQuantProd(dim=d, bit_width=b)
        codes_prod = quant_prod.quantize(X)

        rows.append(
            {
                "b": int(b),
                "mse_bytes": codes_mse.nbytes / n,
                "mse_distortion": metrics_mse["mse_distortions"][i],
                "prod_bytes": codes_prod.nbytes / n,
                "prod_idx": codes_prod.idx.nbytes / n,
                "prod_signs": codes_prod.signs.nbytes / n,
                "prod_norms": codes_prod.norms.nbytes / n,
                "prod_distortion": metrics_prod["inner_prod_errors"][i],
            }
        )

    print(f"\nStorage footprint, bytes per vector (n = {n}, d = {d})")
    print(f"  float32, unquantized: {raw_bytes}\n")
    print(
        f"{'b':>2} | {'MSE':>6} {'bits/coord':>11} {'vs f32':>7}"
        f" | {'Prod':>6} {'idx':>5} {'signs':>6} {'norm':>5} {'bits/coord':>11} {'vs f32':>7}"
    )
    for r in rows:
        print(
            f"{r['b']:>2} | {r['mse_bytes']:6.1f} {r['mse_bytes'] * 8 / d:11.2f} "
            f"{raw_bytes / r['mse_bytes']:6.1f}x"
            f" | {r['prod_bytes']:6.1f} {r['prod_idx']:5.1f} {r['prod_signs']:6.1f} "
            f"{r['prod_norms']:5.1f} {r['prod_bytes'] * 8 / d:11.2f} "
            f"{raw_bytes / r['prod_bytes']:6.1f}x"
        )

    make_memory_plot(rows, raw_bytes, d)
    return rows


def make_memory_plot(rows, raw_bytes, d):
    bit_widths = [r["b"] for r in rows]
    fig, (size_ax, rate_ax) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Storage Footprint", fontsize=16)

    size_ax.axhline(raw_bytes, color="black", linestyle=":", label="float32, unquantized")
    size_ax.axhline(8 * (2 * d + 1), color="gray", linestyle=":",
                    label="float64 concat (previous representation)")
    size_ax.plot(bit_widths, [r["mse_bytes"] for r in rows], marker="o",
                 color="C0", label="TurboQuantMSE")
    size_ax.plot(bit_widths, [r["prod_bytes"] for r in rows], marker="o",
                 color="C1", label="TurboQuantProd")
    size_ax.set_yscale("log")
    size_ax.set_xlabel("Bit-widths")
    size_ax.set_ylabel("Bytes per vector")
    size_ax.set_title("Bytes per vector vs Bit-widths")
    size_ax.legend(fontsize=8)

    rate_ax.plot([r["mse_bytes"] * 8 / d for r in rows],
                 [r["mse_distortion"] for r in rows], marker="o", color="C0",
                 label="TurboQuantMSE ($D_{mse}$)")
    rate_ax.plot([r["prod_bytes"] * 8 / d for r in rows],
                 [r["prod_distortion"] * d for r in rows], marker="o", color="C1",
                 label="TurboQuantProd ($D_{prod}\\cdot d$)")
    rate_ax.plot(bit_widths, [1 / 4**b for b in bit_widths], linestyle="--",
                 color="black", label="Lower Bound ($4^{-b}$)")
    rate_ax.set_yscale("log")
    rate_ax.set_xlabel("Actual bits per coordinate (stored)")
    rate_ax.set_ylabel("Distortion")
    rate_ax.set_title("Distortion vs Bits Actually Stored")
    rate_ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("memory_footprint.png", dpi=120)


def load_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_CACHE.exists():
        embeddings = np.load(DATA_CACHE)
        print(f"Loaded cached data: {embeddings.shape}")
    else:
        print("Downloading the dataset...")
        ds = load_dataset("SLU-CSCI4750/glove.6B.100d.txt")
        df = ds["train"].to_pandas()
        tokens = df["text"].str.strip().str.split()
        embeddings = np.vstack(
            tokens.str[1:].apply(lambda values: np.array(list(map(float, values))))
        )
        np.save(DATA_CACHE, embeddings)
        print(f"DONE\nData Shape: {embeddings.shape} \n\n")

    if SAMPLE_CACHE.exists():
        sample_emb = np.load(SAMPLE_CACHE)
        print(f"Loaded cached sample: {sample_emb.shape}")
        return sample_emb

    print("Sampling 5000 points...")
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(embeddings.shape[0], size=5000, replace=False)
    sample_emb = embeddings[sample_indices]
    sample_emb = sample_emb / np.linalg.norm(sample_emb, keepdims=True, axis=-1)
    np.save(SAMPLE_CACHE, sample_emb)

    print(f"DONE!\nData Shape: {sample_emb.shape}")
    return sample_emb

def main():
    sample_emb = load_data()
    bit_widths = np.array([1, 2, 3, 4, 5])

    lowerbound_mse = lambda b, d : 1/ (4**b)
    upperbound_mse = lambda b, d : math.sqrt(3) * math.pi / (2 * (4**b))
    lowerbound_inner_prod = lambda b, d : 1/(d * 4**b)
    upperbound_inner_prod = lambda b, d : math.sqrt(3) * math.pi**2/(d * 4**b)

    metrics_mse = eval_quant_model(
        sample_emb, bit_widths, TurboQuantMSE, EVALUATION_SEEDS
    )
    metrics_prod = eval_quant_model(
        sample_emb, bit_widths, TurboQuantProd, EVALUATION_SEEDS
    )

    make_plots(bit_widths, metrics_mse, metrics_prod, upperbound_mse, lowerbound_mse, upperbound_inner_prod, lowerbound_inner_prod, sample_emb.shape[-1])
    make_error_distribution_plots(sample_emb)
    make_error_vs_inner_product_plots(sample_emb)
    measure_memory(sample_emb, bit_widths, metrics_mse, metrics_prod)

    

if __name__ == "__main__":
    main()
    