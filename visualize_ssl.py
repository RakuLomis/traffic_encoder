from __future__ import annotations

import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

from utils.data_loader_ptga_le import GNNTrafficDataset
from utils.flow_batch_components import FlowCentricBatchSampler
from models.ProtocolTreeGAttention_le import HierarchicalMoE


# ============================================================
# Configuration
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "cstnet_tls_1.3"
SPLIT_TO_VIS = "validation_set"  # "validation_set" or "test_set"

ROOT_PATH = os.path.join("..", "TrafficData", "datasets_csv_add2")
SPLIT_DIR = os.path.join(ROOT_PATH, "datasets_split", DATASET_NAME)
TRAIN_CSV_PATH = os.path.join(SPLIT_DIR, "train_set.csv")
VAL_CSV_PATH = os.path.join(SPLIT_DIR, "validation_set.csv")
TEST_CSV_PATH = os.path.join(SPLIT_DIR, "test_set.csv")
TARGET_CSV_PATH = os.path.join(SPLIT_DIR, f"{SPLIT_TO_VIS}.csv")

CONFIG_PATH = os.path.join(".", "Data", "fields_embedding_configs_v1.yaml")
VOCAB_PATH = os.path.join(ROOT_PATH, "categorical_vocabs", DATASET_NAME + "_vocabs.yaml")
CKPT_PATH = os.path.join("..", "Res", "ssl_pretrain", DATASET_NAME, "ssl_pretrain_best.pth")
SAVE_DIR = os.path.join("..", "Res", "ssl_pretrain", DATASET_NAME, "tsne")

# Keep feature scope consistent with SSL training.
ENABLED_LAYERS = ["ip", "tcp", "tls"]
USE_IP_ADDRESS = False
USE_MAC_ADDRESS = False
USE_PORT = False
DROP_FIELD_NAMES = ["tls.handshake.extensions_server_name"]
ENABLE_VIRTUAL_SINK = True
VIRTUAL_SINK_NAME = "__VIRTUAL_SINK__"
BACKBONE_MODE = "expert_local"

# Flow-centric extraction
USE_FLOW_CENTRIC_BATCHING = True
FLOW_MICRO_PACKETS_PER_FLOW = 32
FLOW_MACRO_FLOWS_PER_BATCH = 64
BATCH_SIZE_FALLBACK = 1024
NUM_WORKERS = 0

# flow repr pool for visualization
FLOW_POOL_MODE = "max"  # "max" | "mean"

# Optional downsampling for readability (0 = disable)
MAX_FLOWS_PER_CLASS = 300

# Cluster-separation quantitative metrics
ENABLE_CLUSTER_METRICS = True
MIN_SAMPLES_PER_CLASS_FOR_CLUSTER_METRICS = 2
METRICS_USE_PCA = True
METRICS_PCA_COMPONENTS = 50

# t-SNE settings
USE_PCA_BEFORE_TSNE = True
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

# plotting
FIGSIZE = (11, 9)
POINT_SIZE = 10
POINT_ALPHA = 0.8
LEGEND_MAX_CLASSES = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_label_columns(df: pd.DataFrame, label_to_id: Dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns:
        out["label"] = "ssl"
    out["label"] = out["label"].astype(str)
    out["label_id"] = out["label"].map(label_to_id).fillna(0).astype(int)
    return out


def build_label_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    labels = set()
    for d in (train_df, val_df, test_df):
        if "label" in d.columns:
            labels.update(d["label"].astype(str).tolist())
    labels_sorted = sorted(labels)
    return {x: i for i, x in enumerate(labels_sorted)}


def maybe_drop_fields_from_batch(batch_dict: Dict, drop_field_names: List[str]) -> Dict:
    if not drop_field_names:
        return batch_dict
    names = [str(x) for x in drop_field_names if str(x).strip()]
    if not names:
        return batch_dict
    for _, value in batch_dict.items():
        if not hasattr(value, "__dict__"):
            continue
        for n in names:
            if hasattr(value, n):
                try:
                    delattr(value, n)
                except AttributeError:
                    pass
    return batch_dict


def move_batch_to_device(batch_dict: Dict, device: torch.device) -> Dict:
    for key, value in batch_dict.items():
        if hasattr(value, "to"):
            try:
                batch_dict[key] = value.to(device, non_blocking=True)
            except TypeError:
                batch_dict[key] = value.to(device)
    return batch_dict


def aggregate_packet_repr_by_flow(
    packet_repr: torch.Tensor,
    packet_labels: torch.Tensor,
    flow_ids: torch.Tensor,
    pool_mode: str = "max",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
      flow_repr: [F, D]
      flow_labels: [F]
      unique_flow_ids: [F]
    """
    unique_flow_ids, inverse = torch.unique(flow_ids, sorted=False, return_inverse=True)
    n_flows = int(unique_flow_ids.numel())
    if n_flows == 0:
        return (
            torch.empty((0, packet_repr.size(1)), device=packet_repr.device, dtype=packet_repr.dtype),
            torch.empty((0,), device=packet_labels.device, dtype=torch.long),
            torch.empty((0,), device=flow_ids.device, dtype=flow_ids.dtype),
        )

    pool_mode = str(pool_mode).lower()
    if pool_mode == "mean":
        flow_repr = torch.zeros((n_flows, packet_repr.size(1)), dtype=packet_repr.dtype, device=packet_repr.device)
        flow_repr.index_add_(0, inverse, packet_repr)
        cnt = torch.zeros((n_flows,), dtype=packet_repr.dtype, device=packet_repr.device)
        cnt.index_add_(0, inverse, torch.ones_like(inverse, dtype=packet_repr.dtype))
        flow_repr = flow_repr / cnt.clamp_min(1.0).unsqueeze(-1)
    elif pool_mode == "max":
        flow_repr = torch.full((n_flows, packet_repr.size(1)), float("-inf"), dtype=packet_repr.dtype, device=packet_repr.device)
        idx_expand = inverse.unsqueeze(-1).expand(-1, packet_repr.size(1))
        flow_repr.scatter_reduce_(0, idx_expand, packet_repr, reduce="amax", include_self=True)
        empty_rows = torch.isinf(flow_repr).all(dim=1)
        if empty_rows.any():
            flow_repr[empty_rows] = 0.0
    else:
        raise ValueError(f"Unsupported FLOW_POOL_MODE: {pool_mode}")

    # Majority label per flow.
    flow_labels = torch.empty((n_flows,), dtype=torch.long, device=packet_labels.device)
    for i in range(n_flows):
        y = packet_labels[inverse == i]
        binc = torch.bincount(y.long())
        flow_labels[i] = int(torch.argmax(binc).item())

    return flow_repr, flow_labels, unique_flow_ids


def sample_flows_per_class(
    features: np.ndarray,
    labels: np.ndarray,
    flow_ids: np.ndarray,
    max_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_per_class <= 0:
        return features, labels, flow_ids
    keep_indices = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, size=max_per_class, replace=False)
        keep_indices.append(idx)
    if not keep_indices:
        return features, labels, flow_ids
    keep = np.concatenate(keep_indices, axis=0)
    keep = np.sort(keep)
    return features[keep], labels[keep], flow_ids[keep]


def filter_for_cluster_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    min_samples_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only classes with enough samples for stable cluster metrics.
    """
    keep_mask = np.zeros(labels.shape[0], dtype=bool)
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        if int(cnt) >= int(min_samples_per_class):
            keep_mask |= (labels == cls)
    return features[keep_mask], labels[keep_mask]


def main() -> None:
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(TARGET_CSV_PATH):
        raise FileNotFoundError(f"Target split csv not found: {TARGET_CSV_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    train_df_raw = pd.read_csv(TRAIN_CSV_PATH, dtype=str) if os.path.exists(TRAIN_CSV_PATH) else pd.DataFrame()
    val_df_raw = pd.read_csv(VAL_CSV_PATH, dtype=str) if os.path.exists(VAL_CSV_PATH) else pd.DataFrame()
    test_df_raw = pd.read_csv(TEST_CSV_PATH, dtype=str) if os.path.exists(TEST_CSV_PATH) else pd.DataFrame()
    target_df_raw = pd.read_csv(TARGET_CSV_PATH, dtype=str)

    label_to_id = build_label_mapping(train_df_raw, val_df_raw, test_df_raw)
    target_df = ensure_label_columns(target_df_raw, label_to_id)
    id_to_label = {v: k for k, v in label_to_id.items()}

    dataset = GNNTrafficDataset(
        target_df,
        config_path=CONFIG_PATH,
        vocab_path=VOCAB_PATH,
        enabled_layers=ENABLED_LAYERS,
        use_flow_features=False,
        use_ip_address=USE_IP_ADDRESS,
        use_mac_address=USE_MAC_ADDRESS,
        use_port=USE_PORT,
        backbone_mode=BACKBONE_MODE,
        enable_virtual_sink=ENABLE_VIRTUAL_SINK,
        virtual_sink_name=VIRTUAL_SINK_NAME,
        obfuscation_config=None,
    )

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": dataset.collate_from_index,
    }
    if USE_FLOW_CENTRIC_BATCHING:
        batch_sampler = FlowCentricBatchSampler(
            flow_ids=dataset.flow_ids,
            packets_per_flow=FLOW_MICRO_PACKETS_PER_FLOW,
            flows_per_batch=FLOW_MACRO_FLOWS_PER_BATCH,
            shuffle_flows=False,
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE_FALLBACK, shuffle=False, drop_last=False, **loader_kwargs)

    num_classes = int(max(2, len(label_to_id)))
    model = HierarchicalMoE(
        config_path=CONFIG_PATH,
        vocab_path=VOCAB_PATH,
        num_classes=num_classes,
        expert_graph_info=dataset.expert_graphs,
        use_flow_features=False,
        num_flow_features=0,
        hidden_dim=128,
        dropout_rate=0.1,
        expert_gate_noise_std=0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    all_flow_repr = []
    all_flow_labels = []
    all_flow_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            batch = maybe_drop_fields_from_batch(batch, DROP_FIELD_NAMES)

            any_key = next(iter(batch.keys()))
            packet_labels = batch[any_key].y
            flow_ids = batch.get("flow_ids", None)
            if flow_ids is None:
                raise RuntimeError("flow_ids missing in batch. Cannot build flow-level representations.")

            _, _, packet_repr = model(batch, return_packet_repr=True)
            flow_repr, flow_labels, unique_flow_ids = aggregate_packet_repr_by_flow(
                packet_repr=packet_repr,
                packet_labels=packet_labels,
                flow_ids=flow_ids,
                pool_mode=FLOW_POOL_MODE,
            )
            all_flow_repr.append(flow_repr.detach().cpu().numpy())
            all_flow_labels.append(flow_labels.detach().cpu().numpy())
            all_flow_ids.append(unique_flow_ids.detach().cpu().numpy())

    if len(all_flow_repr) == 0:
        raise RuntimeError("No flow representations were extracted.")

    features = np.concatenate(all_flow_repr, axis=0)
    labels = np.concatenate(all_flow_labels, axis=0).astype(np.int64)
    flow_ids_np = np.concatenate(all_flow_ids, axis=0).astype(np.int64)

    features, labels, flow_ids_np = sample_flows_per_class(
        features=features,
        labels=labels,
        flow_ids=flow_ids_np,
        max_per_class=MAX_FLOWS_PER_CLASS,
    )

    n_samples, feat_dim = features.shape
    if n_samples < 3:
        raise RuntimeError(f"Too few flow samples for t-SNE: {n_samples}")

    features_for_tsne = features
    pca_dim_used = None
    if USE_PCA_BEFORE_TSNE:
        pca_dim_used = int(min(PCA_COMPONENTS, feat_dim, n_samples - 1))
        if pca_dim_used >= 2:
            pca = PCA(n_components=pca_dim_used, random_state=SEED)
            features_for_tsne = pca.fit_transform(features_for_tsne)

    perplexity = int(max(5, min(TSNE_PERPLEXITY, n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=TSNE_N_ITER,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(features_for_tsne)

    cluster_metrics = {
        "enabled": bool(ENABLE_CLUSTER_METRICS),
        "min_samples_per_class": int(MIN_SAMPLES_PER_CLASS_FOR_CLUSTER_METRICS),
        "n_flows_before_filter": int(features.shape[0]),
        "n_classes_before_filter": int(len(np.unique(labels))),
        "n_flows_used": 0,
        "n_classes_used": 0,
        "silhouette_raw": None,
        "dbi_raw": None,
        "silhouette_metrics_space": None,
        "dbi_metrics_space": None,
    }
    if ENABLE_CLUSTER_METRICS:
        f_raw, y_raw = filter_for_cluster_metrics(
            features=features,
            labels=labels,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS_FOR_CLUSTER_METRICS,
        )
        n_classes_used = int(len(np.unique(y_raw)))
        cluster_metrics["n_flows_used"] = int(f_raw.shape[0])
        cluster_metrics["n_classes_used"] = n_classes_used

        # Need at least 2 classes and at least 2 samples total.
        if f_raw.shape[0] >= 2 and n_classes_used >= 2:
            try:
                cluster_metrics["silhouette_raw"] = float(silhouette_score(f_raw, y_raw, metric="euclidean"))
            except Exception:
                cluster_metrics["silhouette_raw"] = None
            try:
                cluster_metrics["dbi_raw"] = float(davies_bouldin_score(f_raw, y_raw))
            except Exception:
                cluster_metrics["dbi_raw"] = None

            f_m = f_raw
            if METRICS_USE_PCA:
                pca_dim_metrics = int(min(METRICS_PCA_COMPONENTS, f_m.shape[1], f_m.shape[0] - 1))
                if pca_dim_metrics >= 2:
                    pca_m = PCA(n_components=pca_dim_metrics, random_state=SEED)
                    f_m = pca_m.fit_transform(f_m)
            try:
                cluster_metrics["silhouette_metrics_space"] = float(
                    silhouette_score(f_m, y_raw, metric="euclidean")
                )
            except Exception:
                cluster_metrics["silhouette_metrics_space"] = None
            try:
                cluster_metrics["dbi_metrics_space"] = float(davies_bouldin_score(f_m, y_raw))
            except Exception:
                cluster_metrics["dbi_metrics_space"] = None

    out_df = pd.DataFrame(
        {
            "x": proj[:, 0],
            "y": proj[:, 1],
            "label_id": labels,
            "label_name": [id_to_label.get(int(x), str(int(x))) for x in labels],
            "flow_id": flow_ids_np,
        }
    )
    points_csv = os.path.join(SAVE_DIR, f"{DATASET_NAME}_{SPLIT_TO_VIS}_ssl_tsne_points.csv")
    out_df.to_csv(points_csv, index=False)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(20, len(unique_labels)))
    plt.figure(figsize=FIGSIZE)
    for i, cls in enumerate(unique_labels):
        idx = labels == cls
        cls_name = id_to_label.get(int(cls), str(int(cls)))
        plt.scatter(
            proj[idx, 0],
            proj[idx, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=[cmap(i)],
            label=cls_name,
            linewidths=0,
        )
    plt.title(f"SSL Flow t-SNE ({DATASET_NAME} | {SPLIT_TO_VIS})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    if len(unique_labels) <= LEGEND_MAX_CLASSES:
        plt.legend(loc="best", fontsize=8, markerscale=2, frameon=False)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_{SPLIT_TO_VIS}_ssl_tsne.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    cfg = {
        "dataset_name": DATASET_NAME,
        "split": SPLIT_TO_VIS,
        "target_csv_path": TARGET_CSV_PATH,
        "ckpt_path": CKPT_PATH,
        "save_dir": SAVE_DIR,
        "enabled_layers": ENABLED_LAYERS,
        "use_ip_address": USE_IP_ADDRESS,
        "use_mac_address": USE_MAC_ADDRESS,
        "use_port": USE_PORT,
        "drop_field_names": DROP_FIELD_NAMES,
        "flow_pool_mode": FLOW_POOL_MODE,
        "use_flow_centric_batching": USE_FLOW_CENTRIC_BATCHING,
        "flow_micro_packets_per_flow": FLOW_MICRO_PACKETS_PER_FLOW,
        "flow_macro_flows_per_batch": FLOW_MACRO_FLOWS_PER_BATCH,
        "num_samples": int(n_samples),
        "feature_dim": int(feat_dim),
        "use_pca_before_tsne": USE_PCA_BEFORE_TSNE,
        "pca_dim_used": pca_dim_used,
        "tsne_perplexity_used": perplexity,
        "tsne_n_iter": TSNE_N_ITER,
        "max_flows_per_class": MAX_FLOWS_PER_CLASS,
        "enable_cluster_metrics": ENABLE_CLUSTER_METRICS,
        "metrics_min_samples_per_class": MIN_SAMPLES_PER_CLASS_FOR_CLUSTER_METRICS,
        "metrics_use_pca": METRICS_USE_PCA,
        "metrics_pca_components": METRICS_PCA_COMPONENTS,
        "cluster_metrics": cluster_metrics,
        "points_csv": points_csv,
        "figure_path": fig_path,
    }
    with open(os.path.join(SAVE_DIR, f"{DATASET_NAME}_{SPLIT_TO_VIS}_ssl_tsne_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(
        os.path.join(SAVE_DIR, f"{DATASET_NAME}_{SPLIT_TO_VIS}_ssl_cluster_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cluster_metrics, f, indent=2, ensure_ascii=False)

    print(f"[t-SNE] done. points={points_csv}")
    print(f"[t-SNE] done. figure={fig_path}")
    if ENABLE_CLUSTER_METRICS:
        print(
            "[Cluster Metrics] "
            f"silhouette_raw={cluster_metrics['silhouette_raw']} "
            f"dbi_raw={cluster_metrics['dbi_raw']} "
            f"silhouette_metrics_space={cluster_metrics['silhouette_metrics_space']} "
            f"dbi_metrics_space={cluster_metrics['dbi_metrics_space']}"
        )


if __name__ == "__main__":
    main()
