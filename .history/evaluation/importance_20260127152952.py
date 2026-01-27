import numpy as np
import pandas as pd 
import os

def compute_ngi(values: np.ndarray, tau: float = 0.2) -> np.ndarray:
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
    tau: float = 0.2
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
    output_dir: str,
    tau: float = 0.2,
    prefix: str = ""
):
    """
    Compute field-wise NGI and expert-wise GCS (with logN/GCS),
    and export results to CSV files.

    Args:
        csv_path (str): Path to feature importance CSV
                        (must contain: expert_name, feature_name, importance_score).
        output_dir (str): Directory to save output CSVs.
        tau (float): Temperature for NGI.
        prefix (str): Optional prefix for output filenames.

    Outputs:
        - <prefix>field_ngi_tau{tau}.csv
        - <prefix>expert_gcs_tau{tau}.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    field_records = []
    expert_records = []

    for expert_name, sub_df in df.groupby("expert_name"):
        raw_scores = sub_df["importance_score"].values
        ngi = compute_ngi(raw_scores, tau=tau)
        gcs = compute_gcs(ngi)

        num_fields = len(sub_df)
        logN = np.log(num_fields)
        logN_over_GCS = logN / gcs if gcs > 0 else np.nan

        # ---- expert-level record ----
        expert_records.append({
            "expert_name": expert_name,
            "num_fields": num_fields,
            "GCS": gcs,
            "logN": logN,
            "logN_over_GCS": logN_over_GCS
        })

        # ---- field-level records ----
        for (_, row), ngi_val in zip(sub_df.iterrows(), ngi):
            field_records.append({
                "expert_name": expert_name,
                "feature_name": row["feature_name"],
                "raw_importance": row["importance_score"],
                "NGI": ngi_val
            })

    field_ngi_df = pd.DataFrame(field_records)
    expert_gcs_df = (
        pd.DataFrame(expert_records)
        .sort_values("logN_over_GCS", ascending=False)
        .reset_index(drop=True)
    )

    # ---- export ----
    tau_str = str(tau).replace(".", "_")
    field_out = os.path.join(
        output_dir, f"{prefix}_field_ngi_tau_{tau_str}.csv"
    )
    expert_out = os.path.join(
        output_dir, f"{prefix}_expert_gcs_tau_{tau_str}.csv"
    )

    field_ngi_df.to_csv(field_out, index=False)
    expert_gcs_df.to_csv(expert_out, index=False)

    print(f"[Saved] Field-wise NGI  -> {field_out}")
    print(f"[Saved] Expert-wise GCS -> {expert_out}")

    return field_ngi_df, expert_gcs_df

def analyze_expert_importance(
    csv_path: str,
    output_dir: str,
    tau: float = 0.2,
    prefix: str = ""
):
    """
    Analyze expert-level importance:
    - compute expert NGI
    - compute global expert GCS
    - export CSV files

    Args:
        csv_path (str): CSV with columns [expert_name, importance_score]
        output_dir (str): directory to save outputs
        tau (float): temperature for NGI
        prefix (str): optional prefix for output filenames

    Outputs:
        - <prefix>expert_ngi_tau{tau}.csv
        - <prefix>expert_gcs_tau{tau}.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    expert_names = df["expert_name"].values
    raw_scores = df["importance_score"].values

    # ---- NGI over experts ----
    ngi = compute_ngi(raw_scores, tau=tau)

    df_ngi = pd.DataFrame({
        "expert_name": expert_names,
        "raw_importance": raw_scores,
        "NGI": ngi
    }).sort_values("NGI", ascending=False).reset_index(drop=True)

    # ---- GCS over experts ----
    num_experts = len(ngi)
    gcs = compute_gcs(ngi)
    logN = np.log(num_experts)
    logN_over_GCS = logN / gcs if gcs > 0 else np.nan

    df_gcs = pd.DataFrame([{
        "num_experts": num_experts,
        "GCS": gcs,
        "logN": logN,
        "logN_over_GCS": logN_over_GCS
    }])

    # ---- export ----
    tau_str = str(tau).replace(".", "_")
    ngi_out = os.path.join(
        output_dir, f"{prefix}_EXP_ngi_tau_{tau_str}.csv"
    )
    gcs_out = os.path.join(
        output_dir, f"{prefix}_EXP_gcs_tau_{tau_str}.csv"
    )

    df_ngi.to_csv(ngi_out, index=False)
    df_gcs.to_csv(gcs_out, index=False)

    print(f"[Saved] Expert NGI -> {ngi_out}")
    print(f"[Saved] Expert GCS -> {gcs_out}")

    return df_ngi, df_gcs

def analyze_field_wise_importance_per_expert_v2(
    csv_path: str,
    output_dir: str,
    tau: float = 0.2,
    prefix: str = "",
    invalid_field_regex: str = r"(payload|segment_data|reassembled|segments)"
):
    """
    Compute field-wise NGI and expert-wise GCS in a *semantically valid* feature space.

    This version explicitly excludes structural placeholder fields
    (e.g., payload-related fields) from NGI/GCS normalization to avoid
    interpretability bias.

    Args:
        csv_path (str): Path to feature importance CSV
                        (must contain: expert_name, feature_name, importance_score).
        output_dir (str): Directory to save output CSVs.
        tau (float): Temperature for NGI.
        prefix (str): Optional prefix for output filenames.
        invalid_field_regex (str): Regex for fields to exclude before NGI computation.

    Outputs:
        - <prefix>_field_ngi_tau_{tau}.csv
        - <prefix>_expert_gcs_tau_{tau}.csv
    """
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    field_records = []
    expert_records = []

    for expert_name, sub_df in df.groupby("expert_name"):

        # ------------------------------------------------------------
        # 1. Filter out semantically invalid / placeholder fields
        # ------------------------------------------------------------
        valid_mask = ~sub_df["feature_name"].str.contains(
            invalid_field_regex,
            regex=True,
            na=False
        )
        sub_df_valid = sub_df[valid_mask].reset_index(drop=True)

        # Safety check
        if len(sub_df_valid) == 0:
            print(
                f"[Warning] Expert '{expert_name}' has no valid fields "
                f"after filtering. Skipping NGI/GCS computation."
            )
            continue

        # ------------------------------------------------------------
        # 2. Compute NGI and GCS on valid feature subset
        # ------------------------------------------------------------
        raw_scores = sub_df_valid["importance_score"].values
        ngi = compute_ngi(raw_scores, tau=tau)
        gcs = compute_gcs(ngi)

        num_fields = len(sub_df_valid)
        logN = np.log(num_fields)
        logN_over_GCS = logN / gcs if gcs > 0 else np.nan

        # ---- expert-level record ----
        expert_records.append({
            "expert_name": expert_name,
            "num_fields": num_fields,
            "GCS": gcs,
            "logN": logN,
            "logN_over_GCS": logN_over_GCS
        })

        # ------------------------------------------------------------
        # 3. Field-level NGI records (valid fields only)
        # ------------------------------------------------------------
        for (_, row), ngi_val in zip(sub_df_valid.iterrows(), ngi):
            field_records.append({
                "expert_name": expert_name,
                "feature_name": row["feature_name"],
                "raw_importance": row["importance_score"],
                "NGI": ngi_val
            })

    # ------------------------------------------------------------
    # 4. Construct output DataFrames
    # ------------------------------------------------------------
    field_ngi_df = pd.DataFrame(field_records)

    expert_gcs_df = (
        pd.DataFrame(expert_records)
        .sort_values("logN_over_GCS", ascending=False)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # 5. Export CSV files
    # ------------------------------------------------------------
    tau_str = str(tau).replace(".", "_")

    field_out = os.path.join(
        output_dir, f"{prefix}_field_ngi_tau_{tau_str}_v2.csv"
    )
    expert_out = os.path.join(
        output_dir, f"{prefix}_expert_gcs_tau_{tau_str}_v2.csv"
    )

    field_ngi_df.to_csv(field_out, index=False)
    expert_gcs_df.to_csv(expert_out, index=False)

    print(f"[Saved] Field-wise NGI (filtered)  -> {field_out}")
    print(f"[Saved] Expert-wise GCS (filtered) -> {expert_out}")

    return field_ngi_df, expert_gcs_df



