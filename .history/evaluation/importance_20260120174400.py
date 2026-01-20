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

def analyze_field_wise_importance_per_expert(
    csv_path: str,
    tau: float = 0.5
):
    """
    Compute field-wise NGI and GCS *within each expert*.

    Args:
        csv_path (str): Path to feature importance CSV.
        tau (float): Temperature for NGI softmax.

    Returns:
        field_ngi_df (pd.DataFrame): field-level NGI (with expert separation)
        expert_gcs_df (pd.DataFrame): per-expert GCS summary
    """
    df = pd.read_csv(csv_path)

    field_level_records = []
    expert_level_records = []

    # === 核心：按 expert 分组 ===
    for expert_name, sub_df in df.groupby("expert_name"):
        raw_scores = sub_df["importance_score"].values

        # --- NGI (within this expert only) ---
        ngi = compute_ngi(raw_scores, tau=tau)

        # --- GCS (entropy of this expert's field usage) ---
        gcs = compute_gcs(ngi)

        # 记录 expert-level GCS
        expert_level_records.append({
            "expert_name": expert_name,
            "num_fields": len(sub_df),
            "GCS": gcs
        })

        # 记录 field-level NGI
        for (_, row), ngi_val in zip(sub_df.iterrows(), ngi):
            field_level_records.append({
                "expert_name": expert_name,
                "feature_name": row["feature_name"],
                "raw_importance": row["importance_score"],
                "NGI": ngi_val
            })

    field_ngi_df = pd.DataFrame(field_level_records)
    expert_gcs_df = pd.DataFrame(expert_level_records).sort_values("GCS")

    return field_ngi_df, expert_gcs_df



