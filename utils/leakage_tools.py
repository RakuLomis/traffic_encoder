from __future__ import annotations

from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd


def _pick_first_existing(columns: List[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def resolve_five_tuple_columns(df: pd.DataFrame) -> Dict[str, str | None]:
    cols = df.columns.tolist()
    return {
        "src_ip": _pick_first_existing(cols, ["ip.src", "ipv6.src", "src_ip"]),
        "dst_ip": _pick_first_existing(cols, ["ip.dst", "ipv6.dst", "dst_ip"]),
        "src_port": _pick_first_existing(cols, ["tcp.srcport", "udp.srcport", "src_port", "sport"]),
        "dst_port": _pick_first_existing(cols, ["tcp.dstport", "udp.dstport", "dst_port", "dport"]),
        "proto": _pick_first_existing(cols, ["ip.proto", "ip.protocol", "protocol"]),
    }


def build_five_tuple_keys(df: pd.DataFrame) -> pd.Series:
    """
    Build five-tuple key:
      src_ip|dst_ip|src_port|dst_port|protocol
    """
    cols = resolve_five_tuple_columns(df)
    required = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing = [k for k in required if cols.get(k) is None]
    if missing:
        raise RuntimeError(
            "Cannot build five-tuple keys. Missing required columns for: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    src_ip = df[cols["src_ip"]].fillna("__NA__").astype(str)
    dst_ip = df[cols["dst_ip"]].fillna("__NA__").astype(str)
    src_port = df[cols["src_port"]].fillna("__NA__").astype(str)
    dst_port = df[cols["dst_port"]].fillna("__NA__").astype(str)
    if cols["proto"] is not None:
        proto = df[cols["proto"]].fillna("__NA__").astype(str)
    else:
        proto = pd.Series(["__NA_PROTO__"] * len(df), dtype=str)

    return src_ip + "|" + dst_ip + "|" + src_port + "|" + dst_port + "|" + proto


def _summary_from_key_sets(train_keys: set[str], val_keys: set[str], test_keys: set[str]) -> Dict:
    inter_train_val = len(train_keys.intersection(val_keys))
    inter_train_test = len(train_keys.intersection(test_keys))
    inter_val_test = len(val_keys.intersection(test_keys))
    return {
        "counts": {
            "train_tuples": len(train_keys),
            "val_tuples": len(val_keys),
            "test_tuples": len(test_keys),
        },
        "intersections": {
            "train_val": inter_train_val,
            "train_test": inter_train_test,
            "val_test": inter_val_test,
        },
        "no_leakage_passed": (inter_train_val == 0 and inter_train_test == 0 and inter_val_test == 0),
    }


def compute_leakage_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    train_keys = set(build_five_tuple_keys(train_df).tolist())
    val_keys = set(build_five_tuple_keys(val_df).tolist())
    test_keys = set(build_five_tuple_keys(test_df).tolist())

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "check_type": "five_tuple",
        "five_tuple_format": "(src_ip,dst_ip,src_port,dst_port,protocol)",
    }
    report.update(_summary_from_key_sets(train_keys, val_keys, test_keys))
    return report


def sanitize_splits_prefer_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Sanitization policy:
      1) If train intersects val or test: delete those rows from train.
      2) If val intersects test: delete those rows from val.
      3) Keep test untouched as much as possible.
    """
    train_keys_series = build_five_tuple_keys(train_df)
    val_keys_series = build_five_tuple_keys(val_df)
    test_keys_series = build_five_tuple_keys(test_df)

    train_keys = set(train_keys_series.tolist())
    val_keys = set(val_keys_series.tolist())
    test_keys = set(test_keys_series.tolist())
    before = _summary_from_key_sets(train_keys, val_keys, test_keys)

    # Rule 1: remove from train if intersects with val or test
    bad_train_keys = val_keys.union(test_keys)
    keep_train_mask = ~train_keys_series.isin(bad_train_keys)
    train_new = train_df.loc[keep_train_mask].reset_index(drop=True)

    # Rule 2: remove from val if intersects with test
    bad_val_keys = test_keys
    keep_val_mask = ~val_keys_series.isin(bad_val_keys)
    val_new = val_df.loc[keep_val_mask].reset_index(drop=True)

    # test unchanged
    test_new = test_df.copy().reset_index(drop=True)

    train_keys_after = set(build_five_tuple_keys(train_new).tolist())
    val_keys_after = set(build_five_tuple_keys(val_new).tolist())
    test_keys_after = set(build_five_tuple_keys(test_new).tolist())
    after = _summary_from_key_sets(train_keys_after, val_keys_after, test_keys_after)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "check_type": "five_tuple",
        "policy": {
            "train_vs_val_or_test": "drop_from_train",
            "val_vs_test": "drop_from_val",
            "test": "keep_unchanged",
        },
        "before": before,
        "removed_rows": {
            "train": int((~keep_train_mask).sum()),
            "val": int((~keep_val_mask).sum()),
            "test": 0,
        },
        "after": after,
    }
    return train_new, val_new, test_new, report

