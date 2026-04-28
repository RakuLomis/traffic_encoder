from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import torch
import yaml

from utils.dataframe_tools import protocol_tree, add_root_layer


# ============================================================
# Config
# ============================================================
DATASET_NAME = "cstnet_tls_1.3"
ROOT_PATH = os.path.join("..", "TrafficData", "datasets_csv_add2")
SPLIT_DIR = os.path.join(ROOT_PATH, "datasets_split", DATASET_NAME)
CSV_PATH = os.path.join(SPLIT_DIR, "train_set.csv")
YAML_CONFIG_PATH = os.path.join(".", "Data", "fields_embedding_configs_v1.yaml")
NGI_CSV_PATH = os.path.join("..", "Res", "ssl_pretrain", DATASET_NAME, f"{DATASET_NAME}_ssl_feature_importance_report.csv")
OUTPUT_DIR = os.path.join("..", "Res", "distilled_ptg", DATASET_NAME)
OUTPUT_TAG = "topk5_plusk_noroot_sink"

TOPK_PER_EXPERT = 5
ENABLED_LAYERS = ["ip", "tcp", "tls"]  # e.g., ['eth','ip','tcp','tls'] or subset
BACKBONE_MODE = "expert_local"  # 'expert_local' | 'global'
KEEP_SINK = True
SINK_NAME = "__VIRTUAL_SINK__"
REMOVE_ROOT = True
USE_IP_ADDRESS = False
USE_MAC_ADDRESS = False
USE_PORT = False


def build_edge_index_from_tree(ptree: Dict[str, List[str]], field_to_node_idx: Dict[str, int]) -> torch.Tensor:
    edge_list = []
    for parent, children in ptree.items():
        if parent not in field_to_node_idx:
            continue
        p = field_to_node_idx[parent]
        for c in children:
            if c not in field_to_node_idx:
                continue
            ci = field_to_node_idx[c]
            edge_list.append([ci, p])
            edge_list.append([p, ci])
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def build_parent_map(ptree: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    parent_map: Dict[str, Set[str]] = {}
    for p, children in ptree.items():
        for c in children:
            parent_map.setdefault(c, set()).add(p)
    return parent_map


def collect_ancestors(node: str, parent_map: Dict[str, Set[str]]) -> Set[str]:
    out = set()
    stack = [node]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for par in parent_map.get(cur, set()):
            if par not in out:
                out.add(par)
                stack.append(par)
    return out


def build_expert_definitions(
    csv_fields: Set[str],
    physical_fields: Set[str],
    enabled_layers: Optional[List[str]],
    use_ip_address: bool,
    use_mac_address: bool,
    use_port: bool,
) -> Dict[str, Set[str]]:
    eligible_fields = set(physical_fields)
    eth_fields = {f for f in eligible_fields if f.startswith("eth.")}
    ip_fields = {f for f in eligible_fields if f.startswith("ip.")}
    tcp_core_fields = {f for f in eligible_fields if f.startswith("tcp.") and "options" not in f}
    port_fields = {"tcp.srcport", "tcp.dstport"}

    if not use_mac_address:
        eth_fields = eth_fields - {"eth.src", "eth.dst"}
    if not use_ip_address:
        ip_fields = ip_fields - {"ip.src", "ip.dst"}
    if not use_port:
        tcp_core_fields = tcp_core_fields - port_fields

    expert_defs = {
        "eth": eth_fields,
        "ip": ip_fields,
        "tcp_core": tcp_core_fields,
        "tcp_options": {f for f in eligible_fields if f.startswith("tcp.options.")},
        "tls_record": {f for f in eligible_fields if f.startswith("tls.record.")},
        "tls_handshake": {f for f in eligible_fields if f.startswith("tls.handshake.")},
        "tls_x509": {f for f in eligible_fields if f.startswith("tls.x509")},
    }

    layer_to_experts = {
        "eth": ["eth"],
        "ip": ["ip"],
        "tcp": ["tcp_core", "tcp_options"],
        "tls": ["tls_record", "tls_handshake", "tls_x509"],
    }
    if enabled_layers is not None:
        enabled = set()
        for l in enabled_layers:
            enabled.update(layer_to_experts.get(l, []))
        expert_defs = {k: v for k, v in expert_defs.items() if k in enabled}

    # Keep only fields actually present in CSV.
    expert_defs = {k: set(v).intersection(csv_fields) for k, v in expert_defs.items()}
    return expert_defs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    if not os.path.exists(YAML_CONFIG_PATH):
        raise FileNotFoundError(YAML_CONFIG_PATH)
    if not os.path.exists(NGI_CSV_PATH):
        raise FileNotFoundError(NGI_CSV_PATH)

    df = pd.read_csv(CSV_PATH, dtype=str, nrows=1)
    csv_fields = set(df.columns.tolist())
    with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["field_embedding_config"]
    physical_fields = set(cfg.keys()).intersection(csv_fields)

    expert_defs = build_expert_definitions(
        csv_fields=csv_fields,
        physical_fields=physical_fields,
        enabled_layers=ENABLED_LAYERS,
        use_ip_address=USE_IP_ADDRESS,
        use_mac_address=USE_MAC_ADDRESS,
        use_port=USE_PORT,
    )

    ngi = pd.read_csv(NGI_CSV_PATH)
    required_cols = {"expert_name", "feature_name", "importance_score"}
    if not required_cols.issubset(set(ngi.columns)):
        raise RuntimeError(f"NGI csv missing required columns: {required_cols}")

    out_graphs = {}
    summary_rows = []

    for expert_name, expert_fields in expert_defs.items():
        real_nodes_full = sorted(list(set(expert_fields).intersection(physical_fields)))
        if len(real_nodes_full) == 0:
            continue

        # Full PTG from current expert fields.
        if BACKBONE_MODE == "expert_local":
            ptree_full = protocol_tree(real_nodes_full, list_layers=None, logical_tree=True)
        else:
            ptree_full = protocol_tree(real_nodes_full, list_layers=["eth", "ip", "tcp", "tls"], logical_tree=True)
        add_root_layer(ptree_full, mode="protocol_only")

        # TopK physical fields from NGI.
        sub = ngi[ngi["expert_name"] == expert_name].copy()
        sub = sub[sub["feature_name"].isin(real_nodes_full)]
        sub = sub.sort_values("importance_score", ascending=False)
        topk_fields = sub["feature_name"].head(TOPK_PER_EXPERT).astype(str).tolist()
        topk_fields = [f for f in topk_fields if f in real_nodes_full]
        if len(topk_fields) == 0:
            # fallback: keep a tiny valid graph from first field
            topk_fields = real_nodes_full[: min(TOPK_PER_EXPERT, len(real_nodes_full))]

        parent_map = build_parent_map(ptree_full)
        keep_nodes = set(topk_fields)
        for f in topk_fields:
            keep_nodes.update(collect_ancestors(f, parent_map))

        # remove helper bucket
        keep_nodes.discard("statistics")
        if REMOVE_ROOT:
            keep_nodes.discard("root")
        if KEEP_SINK:
            keep_nodes.add(SINK_NAME)

        # rebuild filtered edges from full tree
        edges = []
        for p, children in ptree_full.items():
            if p not in keep_nodes:
                continue
            for c in children:
                if c not in keep_nodes:
                    continue
                edges.append((p, c))
                edges.append((c, p))

        # ensure sink connectivity
        if KEEP_SINK:
            non_sink_nodes = [n for n in keep_nodes if n != SINK_NAME]
            for n in non_sink_nodes:
                edges.append((SINK_NAME, n))
                edges.append((n, SINK_NAME))

        all_nodes_distilled = sorted(list(keep_nodes))
        field_to_node_idx = {n: i for i, n in enumerate(all_nodes_distilled)}
        edge_pairs = []
        for u, v in edges:
            if u in field_to_node_idx and v in field_to_node_idx:
                edge_pairs.append([field_to_node_idx[u], field_to_node_idx[v]])
        if len(edge_pairs) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

        real_nodes_distilled = sorted([n for n in all_nodes_distilled if n in physical_fields])
        removed_nodes = sorted(list(set(ptree_full.keys()).union(*[set(v) for v in ptree_full.values()]) - set(all_nodes_distilled)))
        added_backbone_nodes = sorted(list(set(real_nodes_distilled).difference(set(topk_fields))))

        # simple quality checks
        assert all(f in all_nodes_distilled for f in topk_fields), f"TopK missing in {expert_name}"
        if REMOVE_ROOT:
            assert "root" not in all_nodes_distilled, f"root still exists in {expert_name}"
        if KEEP_SINK:
            assert SINK_NAME in all_nodes_distilled, f"sink missing in {expert_name}"

        out_graphs[expert_name] = {
            "real_nodes_distilled": real_nodes_distilled,
            "all_nodes_distilled": all_nodes_distilled,
            "field_to_node_idx_distilled": field_to_node_idx,
            "edge_index_distilled": edge_index.tolist(),
            "meta": {
                "topk_fields": topk_fields,
                "added_backbone_nodes": added_backbone_nodes,
                "removed_nodes": removed_nodes,
            },
        }

        # compression stats
        full_nodes = set(ptree_full.keys())
        for vv in ptree_full.values():
            full_nodes.update(vv)
        full_nodes.discard("statistics")
        full_edge = build_edge_index_from_tree(ptree_full, {n: i for i, n in enumerate(sorted(list(full_nodes)))})

        summary_rows.append({
            "expert_name": expert_name,
            "topk": TOPK_PER_EXPERT,
            "full_nodes": len(full_nodes),
            "distilled_nodes": len(all_nodes_distilled),
            "full_edges": int(full_edge.shape[1]),
            "distilled_edges": int(edge_index.shape[1]),
            "node_compression_ratio": len(all_nodes_distilled) / max(len(full_nodes), 1),
            "edge_compression_ratio": int(edge_index.shape[1]) / max(int(full_edge.shape[1]), 1),
        })

    out_obj = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": DATASET_NAME,
        "topk_per_expert": TOPK_PER_EXPERT,
        "keep_sink": KEEP_SINK,
        "sink_name": SINK_NAME,
        "remove_root": REMOVE_ROOT,
        "backbone_mode": BACKBONE_MODE,
        "enabled_layers": ENABLED_LAYERS,
        "use_ip_address": USE_IP_ADDRESS,
        "use_mac_address": USE_MAC_ADDRESS,
        "use_port": USE_PORT,
        "expert_graphs": out_graphs,
    }

    out_json = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_ptgd_{OUTPUT_TAG}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_ptgd_{OUTPUT_TAG}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"[PTG-d] saved spec: {out_json}")
    print(f"[PTG-d] saved summary: {summary_csv}")


if __name__ == "__main__":
    main()

