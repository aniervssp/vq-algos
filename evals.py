import numpy as np
from quantizers import TurboQuantMSE
from metrics import compute_mse
from datasets import load_dataset
import matplotlib.pyplot as plt
import math 
from pathlib import Path

RANDOM_SEED = 42
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DATA_CACHE = CACHE_DIR / "glove_embeddings.npy"
SAMPLE_CACHE = CACHE_DIR / "glove_sample_5000.npy"

def eval_quant_model(X, bit_widths, model, lowerbound_mse, upperbound_mse):
    d = X.shape[-1]
    mse_distortions = []
    inner_prod_mult_bias = []

    for b in bit_widths:
        print(f"Evaluating QuantMSE for bit-width={b}...")
        quant_mse = model(dim=d, bit_width=b)
        idx = quant_mse.quantize(X)

        X_approx = quant_mse.dequantize(idx)

        mse = compute_mse(X, X_approx)
        mse_distortions.append(mse)

        inner_prod = np.mean(X @ X_approx.T)
        inner_prod_mult_bias.append(inner_prod / np.mean(X @ X.T))
    

    fig, (mse_ax, bias_ax) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle("TurboQuantMSE Evaluation")

    mse_ax.plot(
        bit_widths,
        mse_distortions,
        label="empirical"
    )
    mse_ax.plot(
        bit_widths,
        upperbound_mse(bit_widths),
        label="theoretical upper bound",
    )
    mse_ax.plot(
        bit_widths,
        lowerbound_mse(bit_widths),
        label="theoretical lower bound",
    )

    bias_ax.plot(
        bit_widths,
        inner_prod_mult_bias,
        label="inner product multiplicative bias",
    )

    mse_ax.legend()
    mse_ax.set_xlabel("Bit-widths")
    mse_ax.set_ylabel("MSE Distortions")
    mse_ax.set_title("MSE Distortions vs Bit-widths")

    bias_ax.legend()
    bias_ax.set_xlabel("Bit-widths")
    bias_ax.set_ylabel("Inner Product Multiplicative Bias")
    bias_ax.set_title("Inner Product Multiplicative Bias vs Bit-widths")

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
    bit_widths = np.array([1, 2, 3, 4, 5, 7])

    lowerbound_mse = lambda b : 1/ (4**b)
    upperbound_mse = lambda b : math.sqrt(3) * math.pi / (2 * (4**b))
    quant_model = TurboQuantMSE
    eval_quant_model(sample_emb, bit_widths, quant_model, lowerbound_mse, upperbound_mse)    

if __name__ == "__main__":
    main()
    