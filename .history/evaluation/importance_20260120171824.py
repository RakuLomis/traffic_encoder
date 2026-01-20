import numpy as np
import pandas as pd 

def compute_ngi(values: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """
    Compute Normalized Gate Importance (NGI).

    Args:
        values (np.ndarray): Raw gate values in (0,1), e.g., sigmoid outputs.
        tau (float): Temperature for softmax normalization.

    Returns:
        np.ndarray: NGI values summing to 1.
    """
    values = np.asarray(values, dtype=np.float64)
    scaled = values / tau
    exp_vals = np.exp(scaled - np.max(scaled))  # numerical stability
    ngi = exp_vals / np.sum(exp_vals)
    return ngi


def compute_gcs(ngi: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Gate Concentration Score (GCS) as Shannon entropy.

    Args:
        ngi (np.ndarray): Normalized Gate Importance values.
        eps (float): Small constant to avoid log(0).

    Returns:
        float: GCS value.
    """
    ngi = np.asarray(ngi, dtype=np.float64)
    return -np.sum(ngi * np.log(ngi + eps))

def analyze_field_wise_importance(
    csv_path: str,
    tau: float = 0.5
) -> pd.DataFrame:
    """
    Compute NGI and GCS for all protocol fields.

    Returns a DataFrame with NGI values and prints global GCS.
    """
    df = pd.read_csv(csv_path)

    raw_scores = df["importance_score"].values
    ngi = compute_ngi(raw_scores, tau=tau)
    gcs = compute_gcs(ngi)

    df_out = df.copy()
    df_out["NGI"] = ngi

    print(f"[Field-wise] GCS = {gcs:.4f}")

    return df_out.sort_values("NGI", ascending=False)

