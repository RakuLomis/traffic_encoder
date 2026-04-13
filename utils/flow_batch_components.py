from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class FlowCentricBatchSampler(Sampler[List[int]]):
    """
    Build packet-index batches from flow groups.
    Each yielded batch contains K flows, each flow truncated to first-N packets.
    """
    def __init__(
        self,
        flow_ids: torch.Tensor,
        packets_per_flow: int,
        flows_per_batch: int,
        shuffle_flows: bool = True,
        drop_last: bool = False,
    ):
        if flow_ids.ndim != 1:
            raise ValueError("flow_ids must be a 1D tensor")
        self.packets_per_flow = max(1, int(packets_per_flow))
        self.flows_per_batch = max(1, int(flows_per_batch))
        self.shuffle_flows = bool(shuffle_flows)
        self.drop_last = bool(drop_last)

        flow_to_indices: Dict[int, List[int]] = {}
        for idx, fid in enumerate(flow_ids.tolist()):
            k = int(fid)
            if k not in flow_to_indices:
                flow_to_indices[k] = []
            flow_to_indices[k].append(idx)

        self._flow_chunks: List[List[int]] = []
        for _, idxs in flow_to_indices.items():
            if len(idxs) > 0:
                self._flow_chunks.append(idxs[: self.packets_per_flow])

    def __iter__(self):
        order = list(range(len(self._flow_chunks)))
        if self.shuffle_flows:
            random.shuffle(order)
        step = self.flows_per_batch
        for start in range(0, len(order), step):
            end = min(start + step, len(order))
            if self.drop_last and (end - start) < step:
                continue
            flat: List[int] = []
            for i in order[start:end]:
                flat.extend(self._flow_chunks[i])
            if len(flat) > 0:
                yield flat

    def __len__(self):
        n = len(self._flow_chunks)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.flows_per_batch
        return (n + self.flows_per_batch - 1) // self.flows_per_batch


class FlowLogitAttentionPool(nn.Module):
    """
    Learn packet importance from packet logits for flow-level aggregation.
    Works with variable number of packets per flow.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, packet_logits: torch.Tensor) -> torch.Tensor:
        # packet_logits: [N_flow_packets, C] -> scores: [N_flow_packets]
        # Keep dtype consistent with module weights to avoid AMP dtype mismatch
        # when logits are fp16/bf16 while this MLP runs in fp32.
        target_dtype = self.score_mlp[0].weight.dtype
        x = packet_logits.to(dtype=target_dtype)
        return self.score_mlp(x).squeeze(-1)


class FlowReprLogitAggregator(nn.Module):
    """
    Learnable flow-level aggregator over variable-length packet sets.
    Input per packet: fused packet representation + packet logits.
    """
    def __init__(
        self,
        repr_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_prob_input: bool = True,
        use_logit_branch: bool = True,
        attn_temperature: float = 1.0,
        use_residual_mean: bool = True,
        use_logit_residual_baseline: bool = True,
        logit_residual_init: float = 0.0,
        first_n_packets: int = 0,
        pool_mode: str = 'attn',  # 'attn' | 'max' | 'mean'
    ):
        super().__init__()
        self.use_prob_input = bool(use_prob_input)
        self.use_logit_branch = bool(use_logit_branch)
        self.attn_temperature = max(float(attn_temperature), 1e-3)
        self.use_residual_mean = bool(use_residual_mean)
        self.use_logit_residual_baseline = bool(use_logit_residual_baseline)
        self.first_n_packets = max(0, int(first_n_packets))
        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in ('attn', 'max', 'mean'):
            raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")
        self.repr_ln = nn.LayerNorm(repr_dim)
        self.repr_proj = nn.Linear(repr_dim, hidden_dim)
        if self.use_logit_branch:
            self.logit_ln = nn.LayerNorm(num_classes)
            self.logit_proj = nn.Linear(num_classes, hidden_dim)
            fuse_in_dim = hidden_dim * 2
        else:
            self.logit_ln = None
            self.logit_proj = None
            fuse_in_dim = hidden_dim
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(fuse_in_dim),
            nn.Linear(fuse_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_head = nn.Linear(hidden_dim, 1)
        if self.use_residual_mean:
            # Residual gate starts from 0: baseline behaves like mean-pooling at init.
            self.residual_gamma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('residual_gamma', None)
        self.flow_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if self.use_logit_residual_baseline:
            self.logit_residual_gamma = nn.Parameter(torch.tensor(float(logit_residual_init)))
        else:
            self.register_parameter('logit_residual_gamma', None)

    def forward(self, packet_repr: torch.Tensor, packet_logits: torch.Tensor) -> torch.Tensor:
        if self.first_n_packets > 0 and packet_repr.size(0) > self.first_n_packets:
            packet_repr = packet_repr[: self.first_n_packets]
            packet_logits = packet_logits[: self.first_n_packets]
        target_dtype = self.repr_proj.weight.dtype
        repr_in = packet_repr.to(dtype=target_dtype)
        r = self.repr_proj(self.repr_ln(repr_in))
        if self.use_logit_branch:
            logit_in = packet_logits.to(dtype=target_dtype)
            if self.use_prob_input:
                logit_in = torch.softmax(logit_in, dim=-1)
            l = self.logit_proj(self.logit_ln(logit_in))
            h_in = torch.cat([r, l], dim=-1)
        else:
            h_in = r
        h = self.fuse_mlp(h_in)
        if self.pool_mode == 'attn':
            alpha = torch.softmax(self.attn_head(h).squeeze(-1) / self.attn_temperature, dim=0)
            attn_repr = torch.sum(alpha.unsqueeze(-1) * h, dim=0)
        elif self.pool_mode == 'max':
            attn_repr = torch.max(h, dim=0).values
        else:  # mean
            attn_repr = h.mean(dim=0)
        if self.use_residual_mean:
            mean_repr = r.mean(dim=0)
            flow_repr = mean_repr + self.residual_gamma.to(dtype=attn_repr.dtype) * attn_repr
        else:
            flow_repr = attn_repr
        flow_attn_logits = self.flow_head(flow_repr)
        if self.use_logit_residual_baseline:
            mean_packet_logits = packet_logits.to(dtype=flow_attn_logits.dtype).mean(dim=0)
            return mean_packet_logits + self.logit_residual_gamma.to(dtype=flow_attn_logits.dtype) * flow_attn_logits
        return flow_attn_logits

    def forward_grouped(
        self,
        packet_repr: torch.Tensor,
        packet_logits: torch.Tensor,
        inverse_flow_index: torch.Tensor,
        num_flows: int,
    ) -> torch.Tensor:
        """
        Vectorized flow aggregation on GPU.
        packet_repr: [N, D]
        packet_logits: [N, C]
        inverse_flow_index: [N], values in [0, num_flows-1]
        return: [num_flows, C]
        """
        target_dtype = self.repr_proj.weight.dtype
        inverse_flow_index = inverse_flow_index.long()
        if self.first_n_packets > 0 and inverse_flow_index.numel() > 0:
            keep = torch.zeros_like(inverse_flow_index, dtype=torch.bool)
            for fid in range(num_flows):
                idx = torch.nonzero(inverse_flow_index == fid, as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                keep[idx[: self.first_n_packets]] = True
            packet_repr = packet_repr[keep]
            packet_logits = packet_logits[keep]
            inverse_flow_index = inverse_flow_index[keep]
        repr_in = packet_repr.to(dtype=target_dtype)
        r = self.repr_proj(self.repr_ln(repr_in))
        if self.use_logit_branch:
            logit_in = packet_logits.to(dtype=target_dtype)
            if self.use_prob_input:
                logit_in = torch.softmax(logit_in, dim=-1)
            l = self.logit_proj(self.logit_ln(logit_in))
            h_in = torch.cat([r, l], dim=-1)
        else:
            h_in = r
        h = self.fuse_mlp(h_in)  # [N, H]

        if self.pool_mode == 'attn':
            scores = self.attn_head(h).squeeze(-1) / self.attn_temperature  # [N]
            # Segment-softmax per flow id via scatter-reduce + scatter-add.
            max_per_flow = torch.full(
                (num_flows,),
                float('-inf'),
                dtype=scores.dtype,
                device=scores.device,
            )
            max_per_flow.scatter_reduce_(
                0, inverse_flow_index, scores, reduce='amax', include_self=True
            )
            stable = scores - max_per_flow.index_select(0, inverse_flow_index)
            exp_scores = torch.exp(stable)
            denom = torch.zeros((num_flows,), dtype=exp_scores.dtype, device=exp_scores.device)
            denom.scatter_add_(0, inverse_flow_index, exp_scores)
            eps = torch.tensor(1e-12, dtype=denom.dtype, device=denom.device)
            alpha = exp_scores / (denom.index_select(0, inverse_flow_index) + eps)  # [N]
            alpha = alpha.to(dtype=h.dtype)
            attn_repr = torch.zeros((num_flows, h.size(1)), dtype=h.dtype, device=h.device)
            attn_repr.index_add_(0, inverse_flow_index, h * alpha.unsqueeze(-1))
        elif self.pool_mode == 'max':
            attn_repr = torch.full((num_flows, h.size(1)), float('-inf'), dtype=h.dtype, device=h.device)
            idx_expand = inverse_flow_index.unsqueeze(-1).expand(-1, h.size(1))
            attn_repr.scatter_reduce_(0, idx_expand, h, reduce='amax', include_self=True)
            zero_rows = torch.isinf(attn_repr).all(dim=1)
            if zero_rows.any():
                attn_repr[zero_rows] = 0.0
        else:  # mean
            attn_repr = torch.zeros((num_flows, h.size(1)), dtype=h.dtype, device=h.device)
            attn_repr.index_add_(0, inverse_flow_index, h)
            cnt_h = torch.zeros((num_flows,), dtype=h.dtype, device=h.device)
            ones_h = torch.ones((inverse_flow_index.size(0),), dtype=h.dtype, device=h.device)
            cnt_h.index_add_(0, inverse_flow_index, ones_h)
            attn_repr = attn_repr / cnt_h.clamp_min(1.0).unsqueeze(-1)
        if self.use_residual_mean:
            mean_repr = torch.zeros((num_flows, r.size(1)), dtype=r.dtype, device=r.device)
            mean_repr.index_add_(0, inverse_flow_index, r)
            counts = torch.zeros((num_flows,), dtype=r.dtype, device=r.device)
            ones = torch.ones((inverse_flow_index.size(0),), dtype=r.dtype, device=r.device)
            counts.index_add_(0, inverse_flow_index, ones)
            mean_repr = mean_repr / (counts.clamp_min(1.0).unsqueeze(-1))
            flow_repr = mean_repr + self.residual_gamma.to(dtype=attn_repr.dtype) * attn_repr
        else:
            flow_repr = attn_repr
        flow_attn_logits = self.flow_head(flow_repr)
        if self.use_logit_residual_baseline:
            mean_packet_logits = torch.zeros(
                (num_flows, packet_logits.size(1)),
                dtype=flow_attn_logits.dtype,
                device=flow_attn_logits.device,
            )
            mean_packet_logits.index_add_(0, inverse_flow_index, packet_logits.to(dtype=flow_attn_logits.dtype))
            cnt = torch.zeros((num_flows,), dtype=flow_attn_logits.dtype, device=flow_attn_logits.device)
            ones = torch.ones((inverse_flow_index.size(0),), dtype=flow_attn_logits.dtype, device=flow_attn_logits.device)
            cnt.index_add_(0, inverse_flow_index, ones)
            mean_packet_logits = mean_packet_logits / cnt.clamp_min(1.0).unsqueeze(-1)
            return mean_packet_logits + self.logit_residual_gamma.to(dtype=flow_attn_logits.dtype) * flow_attn_logits
        return flow_attn_logits



def aggregate_logits_by_flow_tensor(
    packet_logits: torch.Tensor,
    packet_labels: torch.Tensor,
    flow_ids: torch.Tensor,
    num_classes: int,
    method: str = 'mean_logits',
    topk: int = 8,
    soft_temp: float = 1.0,
    packet_repr: Optional[torch.Tensor] = None,
    packet_weighter: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Aggregate packet logits to flow logits using flow_ids.
    Returns: (flow_logits, flow_labels, inconsistent_flows)
    """
    if packet_logits.ndim != 2:
        raise ValueError("packet_logits must be [N, C].")
    if packet_labels.ndim != 1 or flow_ids.ndim != 1:
        raise ValueError("packet_labels/flow_ids must be [N].")
    if packet_logits.size(0) != packet_labels.size(0) or packet_logits.size(0) != flow_ids.size(0):
        raise ValueError("packet_logits, packet_labels, flow_ids must share same N.")
    if method == 'repr_logits_attn':
        if packet_weighter is None:
            raise RuntimeError("packet_weighter (FlowReprLogitAggregator) is required for repr_logits_attn.")
        if packet_repr is None:
            raise RuntimeError("packet_repr is required for repr_logits_attn.")
        if packet_repr.size(0) != packet_logits.size(0):
            raise ValueError("packet_repr and packet_logits must share same N.")

        unique_flow_ids, inverse = torch.unique(flow_ids, sorted=False, return_inverse=True)
        n_flows = int(unique_flow_ids.numel())
        if n_flows == 0:
            return (
                torch.empty((0, num_classes), device=packet_logits.device, dtype=packet_logits.dtype),
                torch.empty((0,), device=packet_labels.device, dtype=packet_labels.dtype),
                0,
            )

        if hasattr(packet_weighter, "forward_grouped"):
            flow_logits = packet_weighter.forward_grouped(
                packet_repr=packet_repr,
                packet_logits=packet_logits,
                inverse_flow_index=inverse,
                num_flows=n_flows,
            )
        else:
            # Fallback for compatibility.
            agg_logits: List[torch.Tensor] = []
            for i in range(n_flows):
                mask = (inverse == i)
                agg_logits.append(packet_weighter(packet_repr[mask], packet_logits[mask]))
            flow_logits = torch.stack(agg_logits, dim=0)

        # Keep label/inconsistency semantics compatible with previous logic:
        # flow label = min class id appearing in that flow (same as torch.unique(...)[0]).
        label_min = torch.full(
            (n_flows,),
            fill_value=int(num_classes),
            dtype=torch.long,
            device=packet_labels.device,
        )
        label_min.scatter_reduce_(0, inverse, packet_labels.long(), reduce='amin', include_self=True)
        label_max = torch.full(
            (n_flows,),
            fill_value=-1,
            dtype=torch.long,
            device=packet_labels.device,
        )
        label_max.scatter_reduce_(0, inverse, packet_labels.long(), reduce='amax', include_self=True)
        flow_labels = label_min.long()
        inconsistent = int((label_min != label_max).sum().item())
        return flow_logits, flow_labels, inconsistent

    # Fast vectorized path for the common methods to reduce Python-loop overhead.
    if method in ('mean_logits', 'mean_probs'):
        values = packet_logits if method == 'mean_logits' else torch.softmax(packet_logits, dim=1)
        unique_flow_ids, inverse = torch.unique(flow_ids, sorted=False, return_inverse=True)
        n_flows = unique_flow_ids.numel()

        if n_flows == 0:
            return (
                torch.empty((0, num_classes), device=packet_logits.device, dtype=packet_logits.dtype),
                torch.empty((0,), device=packet_labels.device, dtype=packet_labels.dtype),
                0,
            )

        flow_logits = torch.zeros(
            (n_flows, num_classes), device=values.device, dtype=values.dtype
        )
        flow_logits.index_add_(0, inverse, values)
        flow_counts = torch.bincount(inverse, minlength=n_flows).to(values.dtype).unsqueeze(1).clamp_min(1.0)
        flow_logits = flow_logits / flow_counts

        label_onehot = F.one_hot(packet_labels, num_classes=num_classes).to(torch.long)
        label_counts = torch.zeros(
            (n_flows, num_classes), device=packet_labels.device, dtype=torch.long
        )
        label_counts.index_add_(0, inverse, label_onehot)
        flow_labels = torch.argmax(label_counts, dim=1).long()
        inconsistent = int((label_counts.gt(0).sum(dim=1) > 1).sum().item())
        return flow_logits, flow_labels, inconsistent

    unique_flow_ids = torch.unique(flow_ids)
    agg_logits = []
    agg_labels = []
    inconsistent = 0

    for fid in unique_flow_ids:
        mask = (flow_ids == fid)
        logits_f = packet_logits[mask]
        labels_f = packet_labels[mask]

        u = torch.unique(labels_f)
        if u.numel() == 1:
            y = u[0]
        else:
            binc = torch.bincount(labels_f, minlength=num_classes)
            y = torch.argmax(binc)
            inconsistent += 1

        if method == 'learned_attn_logits':
            if packet_weighter is None:
                raise ValueError("packet_weighter is required when method='learned_attn_logits'.")
            scores = packet_weighter(logits_f)
            alpha = torch.softmax(scores, dim=0)
            z = torch.sum(alpha.unsqueeze(-1) * logits_f, dim=0)
        elif method == 'soft_weighted_logits':
            temp = max(float(soft_temp), 1e-4)
            confidence = logits_f.max(dim=1).values
            alpha = torch.softmax(confidence / temp, dim=0)
            z = torch.sum(alpha.unsqueeze(-1) * logits_f, dim=0)
        elif method == 'topk_mean_logits':
            k = max(1, min(int(topk), logits_f.size(0)))
            confidence = logits_f.max(dim=1).values
            topk_idx = torch.topk(confidence, k=k, largest=True).indices
            z = logits_f.index_select(0, topk_idx).mean(dim=0)
        elif method == 'mean_probs':
            z = torch.softmax(logits_f, dim=1).mean(dim=0)
        elif method == 'logsumexp':
            z = torch.logsumexp(logits_f, dim=0) - torch.log(
                torch.tensor(float(logits_f.size(0)), device=logits_f.device)
            )
        else:  # mean_logits
            z = logits_f.mean(dim=0)

        agg_logits.append(z)
        agg_labels.append(y)

    if len(agg_logits) == 0:
        return (
            torch.empty((0, num_classes), device=packet_logits.device, dtype=packet_logits.dtype),
            torch.empty((0,), device=packet_labels.device, dtype=packet_labels.dtype),
            0,
        )

    flow_logits = torch.stack(agg_logits, dim=0)
    flow_labels = torch.stack(agg_labels, dim=0).long()
    return flow_logits, flow_labels, inconsistent
