from __future__ import annotations

from typing import Dict, Any, Tuple
import copy
import torch


_PYG_META_KEYS = {"edge_index", "batch", "ptr", "y", "num_nodes"}


def clone_batch_dict(batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a safe clone for augmentation without mutating the source batch.
    """
    out: Dict[str, Any] = {}
    for k, v in batch_dict.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
            continue
        if hasattr(v, "clone"):
            try:
                out[k] = v.clone()
                continue
            except Exception:
                pass
        if hasattr(v, "to_dict"):  # likely a PyG Data object
            new_v = copy.copy(v)
            for attr, val in v:
                if torch.is_tensor(val):
                    setattr(new_v, attr, val.clone())
            out[k] = new_v
            continue
        out[k] = copy.deepcopy(v)
    return out


def _is_physical_node_attr(attr_name: str, sink_name: str) -> bool:
    if attr_name in _PYG_META_KEYS:
        return False
    if attr_name == sink_name:
        return False
    # Heuristic: physical protocol fields are dot-notated.
    return "." in attr_name


def apply_node_feature_mask(
    batch_dict: Dict[str, Any],
    mask_ratio: float = 0.15,
    sink_name: str = "__VIRTUAL_SINK__",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Randomly mask physical node features in each expert batch.
    Returns augmented batch and a simple mask-info dict.
    """
    mask_ratio = float(max(0.0, min(1.0, mask_ratio)))
    aug = clone_batch_dict(batch_dict)
    mask_info: Dict[str, Any] = {}

    for expert_name, data in aug.items():
        if expert_name in {"flow_ids", "flow_stats"}:
            continue
        if not hasattr(data, "__iter__"):
            continue
        expert_masked = []
        for attr_name, val in data:
            if not torch.is_tensor(val):
                continue
            if not _is_physical_node_attr(attr_name, sink_name):
                continue

            # Apply entry-wise Bernoulli mask.
            keep = torch.rand_like(val.float()) > mask_ratio
            if val.dtype.is_floating_point:
                masked = val * keep.to(val.dtype)
            else:
                masked = torch.where(keep, val, torch.zeros_like(val))
            setattr(data, attr_name, masked)
            expert_masked.append(attr_name)
        mask_info[expert_name] = expert_masked

    return aug, mask_info


def build_sink_metadata(
    expert_graph_info: Dict[str, Dict[str, Any]],
    sink_name: str = "__VIRTUAL_SINK__",
) -> Dict[str, Dict[str, int]]:
    """
    Build per-expert sink metadata from dataset graph info.
    """
    meta: Dict[str, Dict[str, int]] = {}
    for expert_name, info in expert_graph_info.items():
        all_nodes = info.get("all_nodes", [])
        if sink_name in all_nodes:
            sink_idx = int(all_nodes.index(sink_name))
            meta[expert_name] = {
                "sink_idx": sink_idx,
                "num_nodes_per_graph": int(len(all_nodes)),
            }
    return meta


def apply_sink_edge_dropout(
    batch_dict: Dict[str, Any],
    sink_meta: Dict[str, Dict[str, int]],
    p_drop: float = 0.10,
) -> Dict[str, Any]:
    """
    Randomly drop only sink-related edges in batched graphs.
    """
    p_drop = float(max(0.0, min(1.0, p_drop)))
    if p_drop <= 0.0:
        return clone_batch_dict(batch_dict)

    aug = clone_batch_dict(batch_dict)
    for expert_name, cfg in sink_meta.items():
        if expert_name not in aug:
            continue
        data = aug[expert_name]
        if not hasattr(data, "edge_index"):
            continue
        edge_index = data.edge_index
        if not torch.is_tensor(edge_index) or edge_index.numel() == 0:
            continue

        n = int(cfg["num_nodes_per_graph"])
        sink_idx = int(cfg["sink_idx"])
        src_local = edge_index[0] % n
        dst_local = edge_index[1] % n
        sink_edge = (src_local == sink_idx) | (dst_local == sink_idx)

        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        if sink_edge.any():
            rand = torch.rand(int(sink_edge.sum().item()), device=edge_index.device)
            keep_sink = rand > p_drop
            keep[sink_edge] = keep_sink
        data.edge_index = edge_index[:, keep]

    return aug
