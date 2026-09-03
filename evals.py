import polars as pl
import pandas as pd
import numpy as np
from quantizers import TurboQuantMSE, TurboQuantProd
from metrics import compute_mse
from datasets import load_dataset
import matplotlib.pyplot as plt
import math 

def eval_quant_model(X, bit_widths, model, lowerbound_mse, upperbound_mse):
    d = X.shape[-1]
    mse_distortions = []

    for b in bit_widths:
        print(f"Evaluating QuantMSE for bit-width={b}...")
        quant_mse = model(dim=d, bit_width=b)
        idx = quant_mse.quantize(sample_emb)
        sample_emb_approx = quant_mse.dequantize(idx)
        mse = compute_mse(sample_emb, sample_emb_approx)
        mse_distortions.append(mse)

    plt.plot(bit_widths, mse_distortions, label="empirical mse distortions")
    plt.plot(
        bit_widths,
        upperbound_mse(bit_widths),
        label="theoretical upper bound for the mse distortion",
    )
    plt.plot(
        bit_widths,
        lowerbound_mse(bit_widths),
        label="theoretical lower bound for the mse distortion",
    )
    plt.legend()
    plt.savefig("eval_quant_mse.png")

def load_data():

    print("Downloading the dataset...")
    ds = load_dataset("SLU-CSCI4750/glove.6B.100d.txt")
    df = ds["train"].to_pandas()

    tokens = df["text"].str.strip().str.split()

    df["word"] = tokens.str[0]
    df["embedding"] = tokens.str[1:].apply(lambda x: np.array(list(map(float, x))))
    df.drop(columns=['text'], inplace=True)

    print(f"DONE\nData Shape: {df.shape} \n\n")

    print("Sampling 5000 points...")
    sample = df.sample(5000)
    sample_emb = np.vstack(sample["embedding"])
    sample_emb = sample_emb / np.linalg.norm(sample_emb, keepdims=True, axis=-1)

    print(f"DONE!\nData Shape: {sample_emb.shape}")

    return sample_emb

if __name__ == "__main__":
    
    sample_emb = load_data()
    bit_widths = np.array([1, 2, 3, 4, 8])

    lowerbound_mse = lambda b : 1/ (4**b)
    upperbound_mse = lambda b : math.sqrt(3) * math.pi / (2 * (4**b))
    quant_model = TurboQuantMSE

    eval_quant_model(sample_emb, bit_widths, quant_model, lowerbound_mse, upperbound_mse)

