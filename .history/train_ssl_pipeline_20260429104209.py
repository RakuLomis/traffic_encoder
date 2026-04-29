from __future__ import annotations

from typing import Dict, Tuple, Set, List
import os
import gc
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loader_ptga_le import GNNTrafficDataset
from models.ProtocolTreeGAttention_le import HierarchicalMoE
from utils.metrics import calculate_metrics
from utils.train_eval_flow import compute_dataset_expert_importance
from utils.flow_batch_components import (
    FlowCentricBatchSampler,
    aggregate_logits_by_flow_tensor,
)
from utils.leakage_tools import compute_leakage_report, sanitize_splits_prefer_test
from utils.ssl_augment import (
    apply_node_feature_mask,
    apply_sink_edge_dropout,
    build_sink_metadata,
)


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    strict_deterministic_algorithms: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    if strict_deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operators may not support strict deterministic mode.
            pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 0:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def load_pretrained_weights(
    model: HierarchicalMoE,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
    skip_aggregator_head: bool = False,
    skip_mismatch_shapes: bool = True,
) -> Dict:
    """
    Load checkpoint with optional classifier-head skipping.
    Returns load summary for logging.
    """
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(state_dict)}")

    model_state = model.state_dict()
    filtered = state_dict
    skipped_keys: List[str] = []
    agg_skipped_keys: List[str] = []

    def _numel_of(sd: Dict) -> int:
        total = 0
        for _, vv in sd.items():
            try:
                total += int(vv.numel())
            except Exception:
                pass
        return total

    total_tensors_ckpt = len(state_dict)
    total_dims_ckpt = _numel_of(state_dict)

    if skip_aggregator_head:
        agg_skipped_keys = [k for k in filtered.keys() if str(k).startswith("aggregator.")]
        filtered = {
            k: v for k, v in filtered.items()
            if not str(k).startswith("aggregator.")
        }
    candidate_tensors_after_agg = len(filtered)
    candidate_dims_after_agg = _numel_of(filtered)

    if skip_mismatch_shapes:
        shape_ok = {}
        for k, v in filtered.items():
            if k not in model_state:
                # keep unexpected keys behavior handled by load_state_dict
                shape_ok[k] = v
                continue
            try:
                if tuple(model_state[k].shape) == tuple(v.shape):
                    shape_ok[k] = v
                else:
                    skipped_keys.append(k)
            except Exception:
                # conservative fallback: skip problematic entries
                skipped_keys.append(k)
        filtered = shape_ok

    # Use strict=False here because we may intentionally skip keys.
    missing, unexpected = model.load_state_dict(filtered, strict=False if (skip_aggregator_head or skip_mismatch_shapes) else strict)
    skipped_mismatch_dims = 0
    for k in skipped_keys:
        if k in state_dict:
            try:
                skipped_mismatch_dims += int(state_dict[k].numel())
            except Exception:
                pass
    skipped_agg_dims = 0
    for k in agg_skipped_keys:
        if k in state_dict:
            try:
                skipped_agg_dims += int(state_dict[k].numel())
            except Exception:
                pass
    loaded_dims = _numel_of(filtered)
    return {
        "ckpt_path": ckpt_path,
        "strict": bool(strict),
        "skip_aggregator_head": bool(skip_aggregator_head),
        "skip_mismatch_shapes": bool(skip_mismatch_shapes),
        "loaded_keys": len(filtered),
        "loaded_dims": int(loaded_dims),
        "total_ckpt_tensors": int(total_tensors_ckpt),
        "total_ckpt_dims": int(total_dims_ckpt),
        "candidate_tensors_after_agg_skip": int(candidate_tensors_after_agg),
        "candidate_dims_after_agg_skip": int(candidate_dims_after_agg),
        "skipped_aggregator_keys_count": int(len(agg_skipped_keys)),
        "skipped_aggregator_dims": int(skipped_agg_dims),
        "skipped_mismatch_keys_count": len(skipped_keys),
        "skipped_mismatch_dims": int(skipped_mismatch_dims),
        "skipped_mismatch_keys_preview": skipped_keys[:20],
        "skipped_aggregator_keys_preview": agg_skipped_keys[:20],
        "missing_keys": list(missing) if hasattr(missing, "__iter__") else [],
        "unexpected_keys": list(unexpected) if hasattr(unexpected, "__iter__") else [],
    }


def move_batch_to_device(batch_dict: Dict, device: torch.device) -> Dict:
    for key, value in batch_dict.items():
        if hasattr(value, "to"):
            try:
                batch_dict[key] = value.to(device, non_blocking=True)
            except TypeError:
                batch_dict[key] = value.to(device)
    return batch_dict


def maybe_drop_fields_from_batch(
    batch_dict: Dict,
    drop_field_names: List[str],
) -> Dict:
    """
    Optionally remove specific field attributes from all expert Data objects.
    This is more flexible than a hardcoded SNI-only drop.
    """
    if not drop_field_names:
        return batch_dict

    names = [str(x) for x in drop_field_names if str(x).strip()]
    if not names:
        return batch_dict

    for _, value in batch_dict.items():
        # Only expert Data-like objects support attribute deletion.
        if not hasattr(value, "__dict__"):
            continue
        for field_name in names:
            if hasattr(value, field_name):
                try:
                    delattr(value, field_name)
                except AttributeError:
                    pass
    return batch_dict


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard symmetric NT-Xent over two aligned views.
    """
    if z1.size(0) != z2.size(0):
        n = min(z1.size(0), z2.size(0))
        z1 = z1[:n]
        z2 = z2[:n]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)
    if n <= 1:
        return torch.zeros((), device=z1.device, dtype=z1.dtype)

    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.mm(z, z.t()) / max(float(temperature), 1e-6)  # [2N, 2N]
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets], dim=0)  # positives index
    return F.cross_entropy(sim, targets)


def layer_consistency_loss(
    expert_dict_v1: Dict[str, torch.Tensor],
    expert_dict_v2: Dict[str, torch.Tensor],
    pair: Tuple[str, str] = ("tcp_core", "tls_record"),
    temperature: float = 0.2,
) -> torch.Tensor:
    a, b = pair
    if a not in expert_dict_v1 or b not in expert_dict_v1:
        return torch.zeros((), device=next(iter(expert_dict_v1.values())).device)
    if a not in expert_dict_v2 or b not in expert_dict_v2:
        return torch.zeros((), device=next(iter(expert_dict_v2.values())).device)
    # Cross-layer consistency on each view, then average.
    l1 = nt_xent_loss(expert_dict_v1[a], expert_dict_v1[b], temperature=temperature)
    l2 = nt_xent_loss(expert_dict_v2[a], expert_dict_v2[b], temperature=temperature)
    return 0.5 * (l1 + l2)


def aggregate_packet_repr_by_flow(
    packet_repr: torch.Tensor,
    inverse_flow_index: torch.Tensor,
    num_flows: int,
    pool_mode: str = "max",
) -> torch.Tensor:
    """
    Aggregate packet representations into flow representations.
    """
    if num_flows <= 0:
        return torch.empty((0, packet_repr.size(1)), device=packet_repr.device, dtype=packet_repr.dtype)

    pool_mode = str(pool_mode).lower()
    if pool_mode == "mean":
        out = torch.zeros((num_flows, packet_repr.size(1)), dtype=packet_repr.dtype, device=packet_repr.device)
        out.index_add_(0, inverse_flow_index, packet_repr)
        cnt = torch.zeros((num_flows,), dtype=packet_repr.dtype, device=packet_repr.device)
        cnt.index_add_(
            0,
            inverse_flow_index,
            torch.ones((inverse_flow_index.size(0),), dtype=packet_repr.dtype, device=packet_repr.device),
        )
        return out / cnt.clamp_min(1.0).unsqueeze(-1)
    if pool_mode == "max":
        out = torch.full(
            (num_flows, packet_repr.size(1)),
            float("-inf"),
            dtype=packet_repr.dtype,
            device=packet_repr.device,
        )
        idx_expand = inverse_flow_index.unsqueeze(-1).expand(-1, packet_repr.size(1))
        out.scatter_reduce_(0, idx_expand, packet_repr, reduce="amax", include_self=True)
        zero_rows = torch.isinf(out).all(dim=1)
        if zero_rows.any():
            out[zero_rows] = 0.0
        return out
    raise ValueError(f"Unsupported pool_mode: {pool_mode}")


@torch.no_grad()
def evaluate_split_supervised_flow(
    model: HierarchicalMoE,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    drop_field_names: List[str] | None = None,
) -> Tuple[Dict[str, float], torch.Tensor]:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    running_loss = 0.0
    running_items = 0

    for batch in tqdm(dataloader, desc="[SSL-EVAL] Evaluating(flow)", leave=False):
        batch = move_batch_to_device(batch, device)
        batch = maybe_drop_fields_from_batch(batch_dict=batch, drop_field_names=drop_field_names or [])
        any_key = next(iter(batch.keys()))
        labels = batch[any_key].y
        flow_ids = batch.get("flow_ids", None)
        if flow_ids is None:
            raise RuntimeError("[SSL-EVAL] flow_ids missing in batch_dict for flow-level evaluation.")

        logits, _ = model(batch)
        flow_logits, flow_labels, _ = aggregate_logits_by_flow_tensor(
            packet_logits=logits,
            packet_labels=labels,
            flow_ids=flow_ids,
            num_classes=num_classes,
            method="mean_logits",
        )
        if flow_logits.size(0) == 0:
            continue
        loss = F.cross_entropy(flow_logits, flow_labels)
        running_loss += float(loss.item()) * int(flow_labels.size(0))
        running_items += int(flow_labels.size(0))

        pred = torch.argmax(flow_logits, dim=1).detach().cpu()
        gt = flow_labels.detach().cpu()
        for t, p in zip(gt.view(-1), pred.view(-1)):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                cm[int(t), int(p)] += 1

    metrics = calculate_metrics(cm)
    metrics["loss"] = running_loss / max(1, running_items)
    return metrics, cm


def set_requires_grad_for_stage_b(model: HierarchicalMoE, mode: str) -> None:
    mode = str(mode).lower()
    if mode not in ("linear_probe", "fine_tune"):
        raise ValueError(f"Unsupported stage-B mode: {mode}")
    if mode == "fine_tune":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.aggregator.parameters():
        p.requires_grad = True


def get_trainable_parameters(model: HierarchicalMoE):
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch_downstream_flow(
    model: HierarchicalMoE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    drop_field_names: List[str] | None = None,
) -> Tuple[Dict[str, float], torch.Tensor]:
    model.train()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    running_loss = 0.0
    running_items = 0

    for batch in tqdm(dataloader, desc="[Stage-B] Training", leave=False):
        batch = move_batch_to_device(batch, device)
        batch = maybe_drop_fields_from_batch(batch_dict=batch, drop_field_names=drop_field_names or [])
        any_key = next(iter(batch.keys()))
        labels = batch[any_key].y
        flow_ids = batch.get("flow_ids", None)
        if flow_ids is None:
            raise RuntimeError("[Stage-B] flow_ids missing in batch_dict for flow-level training.")

        logits, _ = model(batch)
        flow_logits, flow_labels, _ = aggregate_logits_by_flow_tensor(
            packet_logits=logits,
            packet_labels=labels,
            flow_ids=flow_ids,
            num_classes=num_classes,
            method="mean_logits",
        )
        if flow_logits.size(0) == 0:
            continue

        loss = F.cross_entropy(flow_logits, flow_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(get_trainable_parameters(model), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item()) * int(flow_labels.size(0))
        running_items += int(flow_labels.size(0))

        pred = torch.argmax(flow_logits, dim=1).detach().cpu()
        gt = flow_labels.detach().cpu()
        for t, p in zip(gt.view(-1), pred.view(-1)):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                cm[int(t), int(p)] += 1

    metrics = calculate_metrics(cm)
    metrics["loss"] = running_loss / max(1, running_items)
    return metrics, cm


def build_label_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    all_labels = set()
    for d in (train_df, val_df, test_df):
        if "label" in d.columns:
            all_labels.update(d["label"].astype(str).tolist())
    labels_sorted = sorted(all_labels)
    return {name: i for i, name in enumerate(labels_sorted)}


def ensure_label_columns(df: pd.DataFrame, label_to_id: Dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns:
        out["label"] = "ssl"
    out["label"] = out["label"].astype(str)
    out["label_id"] = out["label"].map(label_to_id).fillna(0).astype(int)
    return out


if __name__ == "__main__":
    # --------------------------
    # Stage-A SSL defaults
    # --------------------------
    SEED = 42
    DETERMINISTIC_TRAINING = True
    STRICT_DETERMINISTIC_ALGORITHMS = False
    set_seed(
        SEED,
        deterministic=DETERMINISTIC_TRAINING,
        strict_deterministic_algorithms=STRICT_DETERMINISTIC_ALGORITHMS,
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    NUM_WORKERS = 2
    PERSISTENT_WORKERS = True
    PREFETCH = 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.2

    # SSL loss weights
    LAMBDA_INST = 1.0
    LAMBDA_LAYER = 0.2
    ENABLE_LAYER_CONSISTENCY = True

    # Augmentation controls
    MASK_RATIO = 0.3 # initial: 0.15
    SINK_DROP_PROB = 0.10
    ENABLE_SINK_EDGE_DROP = True
    STRICT_NO_LEAKAGE_CHECK = True
    ENABLE_LEAKAGE_AUTOFIX = True
    LEAKAGE_AUTOFIX_SAVE_CSV = False # True
    LEAKAGE_AUTOFIX_DIRNAME = "leakage_sanitized"
    ENABLE_SUPERVISED_EVAL = True
    SSL_EVAL_EVERY_N_EPOCHS = 5
    RUN_STAGE_A_SSL = True
    # RUN_STAGE_A_SSL = False
    RUN_STAGE_B_DOWNSTREAM = True
    # Optional: dedicated pretrained checkpoint for Stage-B initialization.
    # If provided and file exists, it will be loaded before Stage-B training.
    SSL_PARA_EXP_PATH = os.path.join('CipherSpectrum', '0417', 'ssl_pretrain_best.pth')
    # PRETRAIN_CKPT_PATH =  os.path.join("..", "Res", "ssl_pretrain", SSL_PARA_EXP_PATH)
    PRETRAIN_CKPT_PATH = None
    # PRETRAIN_LOAD_STRICT = True
    PRETRAIN_LOAD_STRICT = False
    PRETRAIN_SKIP_AGGREGATOR_HEAD = True
    PRETRAIN_SKIP_MISMATCH_SHAPES = True
    SAVE_PRETRAIN_LOAD_REPORT = True
    PRETRAIN_LOAD_REPORT_FILENAME = "pretrain_load_report.json"
    ENABLE_FLOW_CENTRIC_BATCHING = True
    FLOW_MICRO_PACKETS_PER_FLOW = 32
    FLOW_MACRO_FLOWS_PER_BATCH = 32
    FLOW_POOL_MODE = "max"  # 'max' | 'mean'
    DROP_LAST_TRAIN_FLOW_BATCH = False
    # Feature control (must stay consistent across train/val/test)
    USE_IP_ADDRESS = False
    USE_MAC_ADDRESS = False
    USE_PORT = False
    # Optional distilled PTG spec generated by build_distilled_ptg.py
    # Set to None to use original full PTG construction.
    DISTILLED_GRAPH_SPEC_PATH = os.path.join(
        "..", "Res", "distilled_ptg", "CipherSpectrum",
        "CipherSpectrum_ptgd_topk5_plusk_noroot_sink.json"
    )
    # Example:
    # DISTILLED_GRAPH_SPEC_PATH = os.path.join(
    #     "..", "Res", "distilled_ptg", "cstnet_tls_1.3",
    #     "cstnet_tls_1.3_ptgd_topk5_plusk_noroot_sink.json"
    # )
    DROP_FIELD_NAMES = [
        "tls.handshake.extensions_server_name",
    ]
    STAGE_B_MODE = "fine_tune"  # 'linear_probe' | 'fine_tune'
    STAGE_B_EPOCHS = 100
    STAGE_B_LR = 5e-4
    STAGE_B_WEIGHT_DECAY = 1e-4
    STAGE_B_PATIENCE = 15

    # LR scheduler (for both Stage-A and Stage-B)
    ENABLE_LR_SCHEDULER = True
    LR_SCHED_FACTOR = 0.5
    LR_SCHED_MIN_LR = 1e-6
    LR_SCHED_PATIENCE_STAGE_A = 5
    LR_SCHED_PATIENCE_STAGE_B = 5
    LR_SCHED_MONITOR_STAGE_A = "inst_loss"  # "inst_loss" | "layer_loss" | "ssl_loss" | "val_f1"
    LR_SCHED_MONITOR_STAGE_B = "val_f1"    # "val_f1" | "val_loss"

    # --------------------------
    # Paths (edit for your setup)
    # --------------------------
    # dataset_name = "cstnet_tls_1.3"
    dataset_name = "CipherSpectrum"
    root_path = os.path.join("..", "TrafficData", "datasets_csv_add2")
    split_dir = os.path.join(root_path, "datasets_split", dataset_name)
    train_csv_path = os.path.join(split_dir, "train_set.csv")
    val_csv_path = os.path.join(split_dir, "validation_set.csv")
    test_csv_path = os.path.join(split_dir, "test_set.csv")
    config_path = os.path.join(".", "Data", "fields_embedding_configs_v1.yaml")
    vocab_path = os.path.join(root_path, "categorical_vocabs", dataset_name + "_vocabs.yaml")
    save_dir = os.path.join("..", "Res", "ssl_pretrain", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config snapshot
    config_snapshot = {
        "dataset_name": dataset_name,
        "seed": SEED,
        "deterministic_training": DETERMINISTIC_TRAINING,
        "strict_deterministic_algorithms": STRICT_DETERMINISTIC_ALGORITHMS,
        "train_csv": train_csv_path,
        "val_csv": val_csv_path,
        "test_csv": test_csv_path,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "lambda_inst": LAMBDA_INST,
        "lambda_layer": LAMBDA_LAYER,
        "enable_layer_consistency": ENABLE_LAYER_CONSISTENCY,
        "mask_ratio": MASK_RATIO,
        "sink_drop_prob": SINK_DROP_PROB,
        "enable_sink_edge_drop": ENABLE_SINK_EDGE_DROP,
        "enable_flow_centric_batching": ENABLE_FLOW_CENTRIC_BATCHING,
        "flow_micro_packets_per_flow": FLOW_MICRO_PACKETS_PER_FLOW,
        "flow_macro_flows_per_batch": FLOW_MACRO_FLOWS_PER_BATCH,
        "flow_pool_mode": FLOW_POOL_MODE,
        "ssl_eval_every_n_epochs": SSL_EVAL_EVERY_N_EPOCHS,
        "use_ip_address": USE_IP_ADDRESS,
        "use_mac_address": USE_MAC_ADDRESS,
        "use_port": USE_PORT,
        "distilled_graph_spec_path": DISTILLED_GRAPH_SPEC_PATH,
        "drop_field_names": DROP_FIELD_NAMES,
        "strict_no_leakage_check": STRICT_NO_LEAKAGE_CHECK,
        "enable_leakage_autofix": ENABLE_LEAKAGE_AUTOFIX,
        "leakage_autofix_save_csv": LEAKAGE_AUTOFIX_SAVE_CSV,
        "leakage_autofix_dirname": LEAKAGE_AUTOFIX_DIRNAME,
        "run_stage_a_ssl": RUN_STAGE_A_SSL,
        "run_stage_b_downstream": RUN_STAGE_B_DOWNSTREAM,
        "pretrain_ckpt_path": PRETRAIN_CKPT_PATH,
        "pretrain_load_strict": PRETRAIN_LOAD_STRICT,
        "pretrain_skip_aggregator_head": PRETRAIN_SKIP_AGGREGATOR_HEAD,
        "pretrain_skip_mismatch_shapes": PRETRAIN_SKIP_MISMATCH_SHAPES,
        "save_pretrain_load_report": SAVE_PRETRAIN_LOAD_REPORT,
        "pretrain_load_report_filename": PRETRAIN_LOAD_REPORT_FILENAME,
        "stage_b_mode": STAGE_B_MODE,
        "stage_b_epochs": STAGE_B_EPOCHS,
        "stage_b_lr": STAGE_B_LR,
        "stage_b_weight_decay": STAGE_B_WEIGHT_DECAY,
        "stage_b_patience": STAGE_B_PATIENCE,
        "enable_lr_scheduler": ENABLE_LR_SCHEDULER,
        "lr_sched_factor": LR_SCHED_FACTOR,
        "lr_sched_min_lr": LR_SCHED_MIN_LR,
        "lr_sched_patience_stage_a": LR_SCHED_PATIENCE_STAGE_A,
        "lr_sched_patience_stage_b": LR_SCHED_PATIENCE_STAGE_B,
        "lr_sched_monitor_stage_a": LR_SCHED_MONITOR_STAGE_A,
        "lr_sched_monitor_stage_b": LR_SCHED_MONITOR_STAGE_B,
    }
    with open(os.path.join(save_dir, "ssl_config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    print(f"[SSL] device={DEVICE}")
    print(f"[SSL] reading {train_csv_path}")

    train_df_raw = pd.read_csv(train_csv_path, dtype=str)
    val_df_raw = pd.read_csv(val_csv_path, dtype=str) if os.path.exists(val_csv_path) else pd.DataFrame()
    test_df_raw = pd.read_csv(test_csv_path, dtype=str) if os.path.exists(test_csv_path) else pd.DataFrame()

    if STRICT_NO_LEAKAGE_CHECK:
        report_before = compute_leakage_report(train_df_raw, val_df_raw, test_df_raw)
        with open(os.path.join(save_dir, "leakage_check_before_fix.json"), "w", encoding="utf-8") as f:
            json.dump(report_before, f, indent=2, ensure_ascii=False)

        final_report = report_before
        if ENABLE_LEAKAGE_AUTOFIX:
            train_df_raw, val_df_raw, test_df_raw, fix_report = sanitize_splits_prefer_test(
                train_df=train_df_raw,
                val_df=val_df_raw,
                test_df=test_df_raw,
            )
            with open(os.path.join(save_dir, "leakage_fix_report.json"), "w", encoding="utf-8") as f:
                json.dump(fix_report, f, indent=2, ensure_ascii=False)
            report_after = compute_leakage_report(train_df_raw, val_df_raw, test_df_raw)
            with open(os.path.join(save_dir, "leakage_check_after_fix.json"), "w", encoding="utf-8") as f:
                json.dump(report_after, f, indent=2, ensure_ascii=False)
            final_report = report_after

            if LEAKAGE_AUTOFIX_SAVE_CSV:
                fixed_dir = os.path.join(save_dir, LEAKAGE_AUTOFIX_DIRNAME)
                os.makedirs(fixed_dir, exist_ok=True)
                train_df_raw.to_csv(os.path.join(fixed_dir, "train_set.csv"), index=False)
                val_df_raw.to_csv(os.path.join(fixed_dir, "validation_set.csv"), index=False)
                test_df_raw.to_csv(os.path.join(fixed_dir, "test_set.csv"), index=False)

        with open(os.path.join(save_dir, "leakage_check.json"), "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        print(f"[SSL] leakage check: {final_report['intersections']}")
        if not final_report["no_leakage_passed"]:
            raise RuntimeError("[SSL] No-leakage check failed. Abort pretraining.")
    label_to_id = build_label_mapping(train_df_raw, val_df_raw, test_df_raw)
    df = ensure_label_columns(train_df_raw, label_to_id)
    val_df = ensure_label_columns(val_df_raw, label_to_id) if len(val_df_raw) else pd.DataFrame()
    test_df = ensure_label_columns(test_df_raw, label_to_id) if len(test_df_raw) else pd.DataFrame()

    ssl_dataset = GNNTrafficDataset(
        df,
        config_path=config_path,
        vocab_path=vocab_path,
        enabled_layers=["ip", "tcp", "tls"],
        use_flow_features=False,
        use_ip_address=USE_IP_ADDRESS,
        use_mac_address=USE_MAC_ADDRESS,
        use_port=USE_PORT,
        backbone_mode="expert_local",
        enable_virtual_sink=True,
        virtual_sink_name="__VIRTUAL_SINK__",
        obfuscation_config=None,
        distilled_graph_spec_path=DISTILLED_GRAPH_SPEC_PATH,
    )
    sink_meta = build_sink_metadata(ssl_dataset.expert_graphs, sink_name="__VIRTUAL_SINK__")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    common_loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "collate_fn": ssl_dataset.collate_from_index,
        "persistent_workers": (PERSISTENT_WORKERS and NUM_WORKERS > 0),
        "generator": loader_generator,
    }
    if NUM_WORKERS > 0:
        common_loader_kwargs["prefetch_factor"] = PREFETCH

    if ENABLE_FLOW_CENTRIC_BATCHING:
        train_batch_sampler = FlowCentricBatchSampler(
            flow_ids=ssl_dataset.flow_ids,
            packets_per_flow=FLOW_MICRO_PACKETS_PER_FLOW,
            flows_per_batch=FLOW_MACRO_FLOWS_PER_BATCH,
            shuffle_flows=True,
            drop_last=DROP_LAST_TRAIN_FLOW_BATCH,
        )
        loader = DataLoader(
            ssl_dataset,
            batch_sampler=train_batch_sampler,
            **common_loader_kwargs,
        )
    else:
        loader = DataLoader(
            ssl_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )

    val_loader = None
    if ENABLE_SUPERVISED_EVAL and len(val_df) > 0:
        val_dataset = GNNTrafficDataset(
            val_df,
            config_path=config_path,
            vocab_path=vocab_path,
            enabled_layers=["ip", "tcp", "tls"],
            use_flow_features=False,
            use_ip_address=USE_IP_ADDRESS,
            use_mac_address=USE_MAC_ADDRESS,
            use_port=USE_PORT,
            backbone_mode="expert_local",
            enable_virtual_sink=True,
            virtual_sink_name="__VIRTUAL_SINK__",
            obfuscation_config=None,
            distilled_graph_spec_path=DISTILLED_GRAPH_SPEC_PATH,
        )
        val_kwargs = dict(common_loader_kwargs)
        val_kwargs["collate_fn"] = val_dataset.collate_from_index
        if ENABLE_FLOW_CENTRIC_BATCHING:
            val_batch_sampler = FlowCentricBatchSampler(
                flow_ids=val_dataset.flow_ids,
                packets_per_flow=FLOW_MICRO_PACKETS_PER_FLOW,
                flows_per_batch=FLOW_MACRO_FLOWS_PER_BATCH,
                shuffle_flows=False,
                drop_last=False,
            )
            val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **val_kwargs)
        else:
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **val_kwargs)

    test_loader = None
    if ENABLE_SUPERVISED_EVAL and len(test_df) > 0:
        test_dataset = GNNTrafficDataset(
            test_df,
            config_path=config_path,
            vocab_path=vocab_path,
            enabled_layers=["ip", "tcp", "tls"],
            use_flow_features=False,
            use_ip_address=USE_IP_ADDRESS,
            use_mac_address=USE_MAC_ADDRESS,
            use_port=USE_PORT,
            backbone_mode="expert_local",
            enable_virtual_sink=True,
            virtual_sink_name="__VIRTUAL_SINK__",
            obfuscation_config=None,
            distilled_graph_spec_path=DISTILLED_GRAPH_SPEC_PATH,
        )
        test_kwargs = dict(common_loader_kwargs)
        test_kwargs["collate_fn"] = test_dataset.collate_from_index
        if ENABLE_FLOW_CENTRIC_BATCHING:
            test_batch_sampler = FlowCentricBatchSampler(
                flow_ids=test_dataset.flow_ids,
                packets_per_flow=FLOW_MICRO_PACKETS_PER_FLOW,
                flows_per_batch=FLOW_MACRO_FLOWS_PER_BATCH,
                shuffle_flows=False,
                drop_last=False,
            )
            test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **test_kwargs)
        else:
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **test_kwargs)

    num_classes = int(max(2, len(label_to_id)))
    model = HierarchicalMoE(
        config_path=config_path,
        vocab_path=vocab_path,
        num_classes=num_classes,
        expert_graph_info=ssl_dataset.expert_graphs,
        use_flow_features=False,
        num_flow_features=0,
        hidden_dim=128,
        dropout_rate=0.1,
        expert_gate_noise_std=0.0,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_a = None
    if ENABLE_LR_SCHEDULER:
        mode_a = "max" if LR_SCHED_MONITOR_STAGE_A == "val_f1" else "min"
        scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode_a,
            factor=LR_SCHED_FACTOR,
            patience=LR_SCHED_PATIENCE_STAGE_A,
            min_lr=LR_SCHED_MIN_LR,
        )
    best_loss = float("inf")
    best_val_f1 = -1.0
    history = []

    for epoch in range(NUM_EPOCHS if RUN_STAGE_A_SSL else 0):
        model.train()
        running_inst = 0.0
        running_layer = 0.0
        running_total = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"[SSL] Epoch {epoch+1}/{NUM_EPOCHS}"):
            batch = move_batch_to_device(batch, DEVICE)
            batch = maybe_drop_fields_from_batch(
                batch_dict=batch,
                drop_field_names=DROP_FIELD_NAMES,
            )
            if "flow_ids" not in batch:
                raise RuntimeError("[SSL] flow_ids missing in batch. Flow-centric SSL requires flow_ids.")
            flow_ids = batch["flow_ids"]
            unique_flow_ids, inverse_flow_index = torch.unique(
                flow_ids, sorted=False, return_inverse=True
            )
            num_flows_in_batch = int(unique_flow_ids.numel())
            if num_flows_in_batch <= 1:
                continue

            view1, _ = apply_node_feature_mask(batch, mask_ratio=MASK_RATIO, sink_name="__VIRTUAL_SINK__")
            view2, _ = apply_node_feature_mask(batch, mask_ratio=MASK_RATIO, sink_name="__VIRTUAL_SINK__")
            if ENABLE_SINK_EDGE_DROP:
                view1 = apply_sink_edge_dropout(view1, sink_meta=sink_meta, p_drop=SINK_DROP_PROB)
                view2 = apply_sink_edge_dropout(view2, sink_meta=sink_meta, p_drop=SINK_DROP_PROB)

            out1 = model(view1, return_packet_repr=True, return_expert_embeddings=True)
            out2 = model(view2, return_packet_repr=True, return_expert_embeddings=True)
            _, _, z1_pkt, exp1 = out1
            _, _, z2_pkt, exp2 = out2

            z1_flow = aggregate_packet_repr_by_flow(
                packet_repr=z1_pkt,
                inverse_flow_index=inverse_flow_index,
                num_flows=num_flows_in_batch,
                pool_mode=FLOW_POOL_MODE,
            )
            z2_flow = aggregate_packet_repr_by_flow(
                packet_repr=z2_pkt,
                inverse_flow_index=inverse_flow_index,
                num_flows=num_flows_in_batch,
                pool_mode=FLOW_POOL_MODE,
            )

            loss_inst = nt_xent_loss(z1_flow, z2_flow, temperature=TEMPERATURE)
            loss_layer = (
                layer_consistency_loss(exp1, exp2, pair=("tcp_core", "tls_record"), temperature=TEMPERATURE)
                if ENABLE_LAYER_CONSISTENCY
                else torch.zeros_like(loss_inst)
            )
            loss = LAMBDA_INST * loss_inst + LAMBDA_LAYER * loss_layer

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_inst += float(loss_inst.item())
            running_layer += float(loss_layer.item())
            running_total += float(loss.item())
            steps += 1

            del view1, view2, out1, out2, z1_pkt, z2_pkt, z1_flow, z2_flow, exp1, exp2, loss_inst, loss_layer, loss

        epoch_inst = running_inst / max(1, steps)
        epoch_layer = running_layer / max(1, steps)
        epoch_total = running_total / max(1, steps)
        history.append(
            {
                "epoch": epoch + 1,
                "ssl_loss": epoch_total,
                "inst_loss": epoch_inst,
                "layer_loss": epoch_layer,
                "lr": get_current_lr(optimizer),
            }
        )
        msg = (
            f"[SSL] epoch={epoch+1} total={epoch_total:.4f} "
            f"inst={epoch_inst:.4f} layer={epoch_layer:.4f} "
            f"lr={get_current_lr(optimizer):.2e}"
        )

        should_run_val = (
            val_loader is not None
            and (
                ((epoch + 1) % max(1, int(SSL_EVAL_EVERY_N_EPOCHS)) == 0)
                or (epoch + 1 == NUM_EPOCHS)
            )
        )
        if should_run_val:
            val_metrics, _ = evaluate_split_supervised_flow(
                model=model,
                dataloader=val_loader,
                device=DEVICE,
                num_classes=num_classes,
                drop_field_names=DROP_FIELD_NAMES,
            )
            history[-1].update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["accuracy"]),
                    "val_f1_macro": float(val_metrics["f1_macro"]),
                    "val_precision_macro": float(val_metrics["precision_macro"]),
                    "val_recall_macro": float(val_metrics["recall_macro"]),
                }
            )
            msg += (
                f" | val_loss={val_metrics['loss']:.4f}"
                f" val_acc={val_metrics['accuracy']:.4f}"
                f" val_f1={val_metrics['f1_macro']:.4f}"
            )
        elif val_loader is not None:
            msg += " | val_skipped"
        print(msg)

        if epoch_total < best_loss:
            best_loss = epoch_total
            torch.save(model.state_dict(), os.path.join(save_dir, "ssl_pretrain_best.pth"))
        if should_run_val:
            current_val_f1 = float(history[-1].get("val_f1_macro", -1.0))
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                torch.save(model.state_dict(), os.path.join(save_dir, "ssl_best_by_val_f1.pth"))

        if scheduler_a is not None:
            if LR_SCHED_MONITOR_STAGE_A == "val_f1":
                # If val is skipped this epoch, fallback to ssl_loss.
                metric_a = (
                    float(history[-1].get("val_f1_macro"))
                    if ("val_f1_macro" in history[-1])
                    else float(epoch_total)
                )
            elif LR_SCHED_MONITOR_STAGE_A == "inst_loss":
                metric_a = float(epoch_inst)
            elif LR_SCHED_MONITOR_STAGE_A == "layer_loss":
                metric_a = float(epoch_layer)
            else:
                metric_a = float(epoch_total)
            scheduler_a.step(metric_a)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "ssl_training_log.csv"), index=False)

    # Stage-B initialization checkpoint resolution:
    # priority 1) explicit PRETRAIN_CKPT_PATH
    # priority 2) save_dir/ssl_best_by_val_f1.pth
    # priority 3) save_dir/ssl_pretrain_best.pth
    loaded_ckpt = None
    load_info = None
    if PRETRAIN_CKPT_PATH and os.path.exists(PRETRAIN_CKPT_PATH):
        load_info = load_pretrained_weights(
            model=model,
            ckpt_path=PRETRAIN_CKPT_PATH,
            device=DEVICE,
            strict=PRETRAIN_LOAD_STRICT,
            skip_aggregator_head=PRETRAIN_SKIP_AGGREGATOR_HEAD,
            skip_mismatch_shapes=PRETRAIN_SKIP_MISMATCH_SHAPES,
        )
        loaded_ckpt = PRETRAIN_CKPT_PATH
        if load_info.get("skip_aggregator_head", False):
            print("[Stage-B] loaded pretrained backbone with aggregator head skipped.")
        if load_info.get("skipped_mismatch_keys_count", 0) > 0:
            print(
                f"[Stage-B] skipped {load_info['skipped_mismatch_keys_count']} mismatched keys "
                f"(preview: {load_info.get('skipped_mismatch_keys_preview', [])[:5]})."
            )
    else:
        best_ckpt_path = os.path.join(save_dir, "ssl_best_by_val_f1.pth")
        if os.path.exists(best_ckpt_path):
            load_info = load_pretrained_weights(
                model=model,
                ckpt_path=best_ckpt_path,
                device=DEVICE,
                strict=True,
                skip_aggregator_head=False,
            )
            loaded_ckpt = best_ckpt_path
        elif os.path.exists(os.path.join(save_dir, "ssl_pretrain_best.pth")):
            fallback_ckpt = os.path.join(save_dir, "ssl_pretrain_best.pth")
            load_info = load_pretrained_weights(
                model=model,
                ckpt_path=fallback_ckpt,
                device=DEVICE,
                strict=True,
                skip_aggregator_head=False,
            )
            loaded_ckpt = fallback_ckpt

    if load_info is not None:
        print(
            "[Stage-B] pretrain load stats: "
            f"loaded_dims={load_info.get('loaded_dims', 'NA')} / "
            f"total_ckpt_dims={load_info.get('total_ckpt_dims', 'NA')} | "
            f"skipped_agg_dims={load_info.get('skipped_aggregator_dims', 0)} | "
            f"skipped_mismatch_dims={load_info.get('skipped_mismatch_dims', 0)}"
        )
        if SAVE_PRETRAIN_LOAD_REPORT:
            report_path = os.path.join(save_dir, PRETRAIN_LOAD_REPORT_FILENAME)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(load_info, f, indent=2, ensure_ascii=False)

    if RUN_STAGE_B_DOWNSTREAM:
        if loaded_ckpt is not None:
            print(f"[Stage-B] initialized from checkpoint: {loaded_ckpt}")
        else:
            print("[Stage-B] no pretrained checkpoint found; training from current model state.")

    if RUN_STAGE_B_DOWNSTREAM:
        print(
            f"[Stage-B] start downstream training: mode={STAGE_B_MODE}, "
            f"epochs={STAGE_B_EPOCHS}, lr={STAGE_B_LR}"
        )
        set_requires_grad_for_stage_b(model, STAGE_B_MODE)
        trainable_params = get_trainable_parameters(model)
        if len(trainable_params) == 0:
            raise RuntimeError("[Stage-B] no trainable parameters found.")

        optimizer_b = optim.AdamW(
            trainable_params,
            lr=STAGE_B_LR,
            weight_decay=STAGE_B_WEIGHT_DECAY,
        )
        scheduler_b = None
        if ENABLE_LR_SCHEDULER:
            mode_b = "max" if LR_SCHED_MONITOR_STAGE_B == "val_f1" else "min"
            scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_b,
                mode=mode_b,
                factor=LR_SCHED_FACTOR,
                patience=LR_SCHED_PATIENCE_STAGE_B,
                min_lr=LR_SCHED_MIN_LR,
            )
        best_b_val_f1 = -1.0
        no_improve = 0
        stage_b_history = []

        for epoch_b in range(STAGE_B_EPOCHS):
            train_b_metrics, _ = train_one_epoch_downstream_flow(
                model=model,
                dataloader=loader,
                optimizer=optimizer_b,
                device=DEVICE,
                num_classes=num_classes,
                drop_field_names=DROP_FIELD_NAMES,
            )
            row = {
                "epoch": epoch_b + 1,
                "train_loss": float(train_b_metrics["loss"]),
                "train_acc": float(train_b_metrics["accuracy"]),
                "train_f1_macro": float(train_b_metrics["f1_macro"]),
                "lr": get_current_lr(optimizer_b),
            }
            msg = (
                f"[Stage-B] epoch={epoch_b+1}/{STAGE_B_EPOCHS} "
                f"train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_acc']:.4f} "
                f"train_f1={row['train_f1_macro']:.4f} "
                f"lr={row['lr']:.2e}"
            )

            if val_loader is not None:
                val_b_metrics, _ = evaluate_split_supervised_flow(
                    model=model,
                    dataloader=val_loader,
                    device=DEVICE,
                    num_classes=num_classes,
                    drop_field_names=DROP_FIELD_NAMES,
                )
                row.update(
                    {
                        "val_loss": float(val_b_metrics["loss"]),
                        "val_acc": float(val_b_metrics["accuracy"]),
                        "val_f1_macro": float(val_b_metrics["f1_macro"]),
                    }
                )
                msg += (
                    f" | val_loss={row['val_loss']:.4f} "
                    f"val_acc={row['val_acc']:.4f} "
                    f"val_f1={row['val_f1_macro']:.4f}"
                )

                if row["val_f1_macro"] > best_b_val_f1:
                    best_b_val_f1 = row["val_f1_macro"]
                    no_improve = 0
                    torch.save(model.state_dict(), os.path.join(save_dir, "stage_b_best_by_val_f1.pth"))
                else:
                    no_improve += 1
                    if no_improve >= max(1, int(STAGE_B_PATIENCE)):
                        stage_b_history.append(row)
                        print(msg + f" | early_stop(patience={STAGE_B_PATIENCE})")
                        break

                if scheduler_b is not None:
                    metric_b = (
                        float(row["val_f1_macro"])
                        if LR_SCHED_MONITOR_STAGE_B == "val_f1"
                        else float(row["val_loss"])
                    )
                    scheduler_b.step(metric_b)
            elif scheduler_b is not None:
                # No val loader: fallback to train loss monitoring.
                scheduler_b.step(float(row["train_loss"]))

            print(msg)
            stage_b_history.append(row)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if stage_b_history:
            pd.DataFrame(stage_b_history).to_csv(
                os.path.join(save_dir, "stage_b_training_log.csv"),
                index=False,
            )

        stage_b_ckpt = os.path.join(save_dir, "stage_b_best_by_val_f1.pth")
        if os.path.exists(stage_b_ckpt):
            model.load_state_dict(torch.load(stage_b_ckpt, map_location=DEVICE))

    if test_loader is not None:
        test_metrics, test_cm = evaluate_split_supervised_flow(
            model=model,
            dataloader=test_loader,
            device=DEVICE,
            num_classes=num_classes,
            drop_field_names=DROP_FIELD_NAMES,
        )
        print(
            "[SSL] Final Test Performance: "
            f"loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"f1_macro={test_metrics['f1_macro']:.4f}"
        )

        id_to_label = {v: k for k, v in label_to_id.items()}
        class_names: List[str] = [id_to_label.get(i, str(i)) for i in range(num_classes)]
        cm_df = pd.DataFrame(test_cm.numpy(), index=class_names, columns=class_names)
        cm_path = os.path.join(save_dir, f"{dataset_name}_ssl_final_test_confusion_matrix.csv")
        cm_df.to_csv(cm_path)

        test_summary = {
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["accuracy"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "test_precision_macro": float(test_metrics["precision_macro"]),
            "test_recall_macro": float(test_metrics["recall_macro"]),
            "num_classes": int(num_classes),
            "num_test_samples": int(test_cm.sum().item()),
        }
        with open(os.path.join(save_dir, "ssl_final_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)

        # Also append final test metrics into stage_b_training_log.csv for one-file tracking.
        if RUN_STAGE_B_DOWNSTREAM:
            stage_b_log_path = os.path.join(save_dir, "stage_b_training_log.csv")
            final_test_row = {
                "epoch": "final_test",
                "lr": get_current_lr(optimizer_b) if RUN_STAGE_B_DOWNSTREAM else None,
                "train_loss": None,
                "train_acc": None,
                "train_f1_macro": None,
                "val_loss": None,
                "val_acc": None,
                "val_f1_macro": None,
                "test_loss": float(test_metrics["loss"]),
                "test_acc": float(test_metrics["accuracy"]),
                "test_f1_macro": float(test_metrics["f1_macro"]),
                "test_precision_macro": float(test_metrics["precision_macro"]),
                "test_recall_macro": float(test_metrics["recall_macro"]),
            }
            if os.path.exists(stage_b_log_path):
                df_stage_b = pd.read_csv(stage_b_log_path)
                # Ensure columns exist before append.
                for k in final_test_row.keys():
                    if k not in df_stage_b.columns:
                        df_stage_b[k] = np.nan
                df_stage_b = pd.concat([df_stage_b, pd.DataFrame([final_test_row])], ignore_index=True)
                df_stage_b.to_csv(stage_b_log_path, index=False)
            else:
                pd.DataFrame([final_test_row]).to_csv(stage_b_log_path, index=False)

        # NGI-like feature importance report
        importance_reports_dict = model.get_feature_importance()
        rows = []
        for expert_name, expert_df in importance_reports_dict.items():
            tmp = expert_df.copy()
            tmp["expert_name"] = expert_name
            rows.append(tmp[["expert_name", "feature_name", "importance_score"]])
        if rows:
            ngi_df = pd.concat(rows, axis=0, ignore_index=True)
            ngi_df.to_csv(
                os.path.join(save_dir, f"{dataset_name}_ssl_feature_importance_report.csv"),
                index=False,
            )

        # GCS-like expert importance reports
        try:
            lastbatch_expert_df = model.get_expert_importance()
            lastbatch_expert_df.to_csv(
                os.path.join(save_dir, f"{dataset_name}_ssl_lastbatch_expert_layer_importance.csv"),
                index=False,
            )
        except Exception as e:
            print(f"[SSL] warning: cannot export last-batch expert importance: {e}")

        try:
            expected_expert_weights = compute_dataset_expert_importance(model, test_loader, DEVICE)
            expert_names = list(model.gnn_expert_names)
            if model.use_flow_features:
                expert_names.append("Flow_Features_Block")
            gcs_df = pd.DataFrame(
                {"expert_name": expert_names, "importance_score": expected_expert_weights.numpy()}
            ).sort_values("importance_score", ascending=False)
            gcs_df.to_csv(
                os.path.join(save_dir, f"{dataset_name}_ssl_expert_layer_importance.csv"),
                index=False,
            )
        except Exception as e:
            print(f"[SSL] warning: cannot export dataset-level expert importance: {e}")

    print(f"[SSL] done. best_loss={best_loss:.6f}")
