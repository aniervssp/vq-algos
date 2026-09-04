import numpy as np
from quantizers import TurboQuantMSE, TurboQuantProd
from datasets import load_dataset
import matplotlib.pyplot as plt
import math 
from pathlib import Path

RANDOM_SEED = 42
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DATA_CACHE = CACHE_DIR / "glove_embeddings.npy"
SAMPLE_CACHE = CACHE_DIR / "glove_sample_5000.npy"

def eval_quant_model(X, bit_widths, model):
    d = X.shape[-1]

    mse_distortions = []
    inner_prod_mult_bias = []
    inner_prod_errors = []

    for b in bit_widths:
        quant_model = model(dim=d, bit_width=b)
        idx = quant_model.quantize(X)

        X_approx = quant_model.dequantize(idx)

        mse = np.mean(np.sum((X - X_approx) ** 2, axis=-1))
        mse_distortions.append(mse)

        inner_prods = X @ X.T 
        inner_prods_approx = X @ X_approx.T

        inner_prod_mean_err = np.mean(np.abs(inner_prods - inner_prods_approx))

        inner_prod_mult_bias.append(
            np.mean(inner_prods_approx) / np.mean(inner_prods)
        )

        inner_prod_errors.append(inner_prod_mean_err)

    return {    
        "mse_distortions": mse_distortions,
        "inner_prod_mult_bias": inner_prod_mult_bias,
        "inner_prod_errors": inner_prod_errors,
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
    fig.savefig("eval_quant_mse.png")



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
    
    metrics_mse = eval_quant_model(sample_emb, bit_widths, TurboQuantMSE)
    metrics_prod = eval_quant_model(sample_emb, bit_widths, TurboQuantProd)

    make_plots(bit_widths, metrics_mse, metrics_prod, upperbound_mse, lowerbound_mse, upperbound_inner_prod, lowerbound_inner_prod, sample_emb.shape[-1])

    

if __name__ == "__main__":
    main()
    