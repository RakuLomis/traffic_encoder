from typing import Dict, Tuple, List, Optional
import torch
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm 
from utils.data_loader import TrafficDataset
from torch.utils.data import Dataset, DataLoader
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention 
# from models.PTA_rebuild import ProtocolTreeAttention
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from torch.profiler import profile, record_function, ProfilerActivity
from utils.data_loader import custom_collate_fn
from models.MoEPTA import MoEPTA
# from utils.data_loader_gnn import GNNTrafficDataset, gnn_collate_fn
# from utils.data_loader_ptga import GNNTrafficDataset
from utils.data_loader_ptga_le import GNNTrafficDataset
from models.ProtocolTreeGAttention import ProtocolTreeGAttention
from models.ProtocolTreeGAttention_le import HierarchicalMoE
from utils.metrics import calculate_metrics
from utils.model_utils import diagnose_gate_weights_for_class
import sys
# from transformers import get_linear_schedule_with_warmup
from utils.loss_functions import FocalLoss
import numpy as np
import random 
from torch.optim import RAdam
import copy
import gc
from utils.dataframe_tools import stratified_sample_dataframe, stratified_hybrid_sample_dataframe_optimized, stratified_aggressive_balancing
from utils.dataframe_tools import stratified_hybrid_sample_from_csv_stream
from utils.dataframe_tools import stratified_flow_sample_from_csv_stream

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

TORCH_LONG_MAX = torch.iinfo(torch.long).max
def robust_hex_to_int(x):
    """Safely convert a hex-like value to int; return 0 on invalid input."""
    if not pd.notna(x): return 0
    try:
         
        val_str = str(x).split('.')[0]
         
        val_str = val_str.lower().replace('0x','')
        if not val_str: return 0   
        return min(int(val_str, 16), TORCH_LONG_MAX)
    except ValueError:
        return 0   
    
def robust_timestamp_to_tsval(x):
    """
    Extract TSval from tcp.options.timestamp (format: 080a + TSval + TSecr).
    Return 0 when format is invalid.
    """
    if not pd.notna(x): return 0
    try:
         
        s = str(x).lower().replace('0x', '')
        
         
         
        if len(s) != 20 or not s.startswith('080a'):
            return 0   
            
         
        tsval_hex = s[4:12]
        
         
        return int(tsval_hex, 16)
    except (ValueError, TypeError):
        return 0   

def set_seed(seed_value: int, deterministic: bool = True):
    """
    Set all relevant random seeds for reproducibility.
    """
    print(f"Setting global seed to {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)   
        
        # Deterministic mode is reproducible but often slower.
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

def seed_worker(worker_id):
    """
    Initialize worker-level RNG state for DataLoader workers.
    """
     
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


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


def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    num_classes: int, 
     
    loss_fn: nn.Module, # for focal_loss
    alpha: float = 1e-5,
    train_target: str = 'packet',
    flow_agg_method: str = 'mean_logits',
    flow_topk: int = 8,
    flow_soft_temp: float = 1.0,
    flow_packet_weighter: Optional[nn.Module] = None,
    flow_loss_use_packet_aux: bool = True,
    flow_packet_aux_weight: float = 0.1,
    current_epoch: int = 0,
    repr_detach_warmup_epochs: int = 0,
    repr_permanent_detach: bool = False,
    flow_repr_gamma_clamp_max: Optional[float] = None,
    collect_dual_level_metrics: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: torch.cuda.amp.GradScaler = None,
) -> (Dict, torch.Tensor): # type: ignore 
    """
    Run one training epoch for the HierarchicalMoE model.
    The dataloader yields a dictionary of PyG batches.
    Returns aggregated metrics and the confusion matrix.
    """
    model.train()
    if flow_packet_weighter is not None:
        flow_packet_weighter.train()
    running_loss = 0.0
    running_items = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    packet_confusion_matrix = (
        torch.zeros(num_classes, num_classes, dtype=torch.long)
        if collect_dual_level_metrics else None
    )
    flow_confusion_matrix = (
        torch.zeros(num_classes, num_classes, dtype=torch.long)
        if collect_dual_level_metrics else None
    )
    
     
     
    # base_loss_fn = nn.CrossEntropyLoss(reduction='none')

    for i, batch_dict in enumerate(tqdm(dataloader, desc="Training")):
         
        # batch_dict = batch_dict.to(device)

         
        # Move each item in batch_dict to device when supported.
         
        # This avoids assumptions about a specific batch container type.
        try:
            for key, value in batch_dict.items():
                if hasattr(value, 'to'):   
                    try:
                        batch_dict[key] = value.to(device, non_blocking=True)
                    except TypeError:
                        batch_dict[key] = value.to(device)
        except Exception as e:
              
             print(f"Warning: failed to move batch item '{key}' to device. Error: {e}")
        
         
         
        # labels = batch_dict['eth'].y 
        any_key = next(iter(batch_dict.keys()))
        labels = batch_dict[any_key].y
        flow_ids = batch_dict.get('flow_ids', None)

         
         
        FIELD_TO_IGNORE = 'tls.handshake.extensions_server_name'
        
         
        if 'tls_handshake' in batch_dict and hasattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE):
            try:
                 
                delattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE)
            except AttributeError:
                pass   
        # =====================================================================
        
         
        #    outputs = logits
        #    gates_dict = {'eth': gate_tensor, 'ip': gate_tensor, ...}
        with torch.autocast(
            device_type='cuda',
            dtype=amp_dtype,
            enabled=(use_amp and device.type == 'cuda')
        ):
            need_packet_repr = (train_target == 'flow' and flow_agg_method == 'repr_logits_attn')
            model_out = model(batch_dict, return_packet_repr=need_packet_repr)
            if need_packet_repr:
                outputs, gates_dict, packet_repr = model_out
            else:
                outputs, gates_dict = model_out
                packet_repr = None
        
         
        # classification_loss_per_sample = base_loss_fn(outputs, labels)

         
         
        # sample_weights = dynamic_weights[labels]
         
        # classification_loss = (classification_loss_per_sample * sample_weights).mean()
            packet_loss = loss_fn(outputs, labels)
            if train_target == 'flow':
                if flow_ids is None:
                    raise RuntimeError("flow_ids missing in batch_dict while train_target='flow'.")
                packet_repr_for_agg = packet_repr
                if (
                    flow_agg_method == 'repr_logits_attn'
                    and packet_repr_for_agg is not None
                    and (
                        repr_permanent_detach
                        or current_epoch < max(0, int(repr_detach_warmup_epochs))
                    )
                ):
                    packet_repr_for_agg = packet_repr_for_agg.detach()
                flow_logits, flow_labels, _ = aggregate_logits_by_flow_tensor(
                    packet_logits=outputs,
                    packet_labels=labels,
                    flow_ids=flow_ids,
                    num_classes=num_classes,
                    method=flow_agg_method,
                    topk=flow_topk,
                    soft_temp=flow_soft_temp,
                    packet_repr=packet_repr_for_agg,
                    packet_weighter=flow_packet_weighter,
                )
                classification_loss = loss_fn(flow_logits, flow_labels)
                if flow_loss_use_packet_aux:
                    classification_loss = classification_loss + flow_packet_aux_weight * packet_loss
            else:
                classification_loss = packet_loss
            
            total_mask_entropy_loss = 0.0
            num_experts_with_gate = len(gates_dict)   
            
            if num_experts_with_gate > 0:
                for name, gate in gates_dict.items():
                    total_mask_entropy_loss += -(gate * torch.log(gate + 1e-8) + 
                                                (1 - gate) * torch.log(1 - gate + 1e-8)).mean()
                
                total_mask_entropy_loss = total_mask_entropy_loss / num_experts_with_gate
            
            total_loss = classification_loss + alpha * total_mask_entropy_loss
        
         
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            params_to_clip = list(model.parameters())
            if flow_packet_weighter is not None:
                params_to_clip += list(flow_packet_weighter.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            params_to_clip = list(model.parameters())
            if flow_packet_weighter is not None:
                params_to_clip += list(flow_packet_weighter.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            optimizer.step()

        if (
            flow_packet_weighter is not None
            and flow_repr_gamma_clamp_max is not None
            and hasattr(flow_packet_weighter, "logit_residual_gamma")
            and getattr(flow_packet_weighter, "logit_residual_gamma") is not None
        ):
            with torch.no_grad():
                flow_packet_weighter.logit_residual_gamma.clamp_(
                    min=0.0, max=float(flow_repr_gamma_clamp_max)
                )
        
         
         
         

         
         
        if collect_dual_level_metrics:
            # Optional dual-level metrics (adds overhead).
            _, packet_pred = torch.max(outputs.data, 1)
            packet_labels_cpu = labels.detach().cpu()
            packet_pred_cpu = packet_pred.detach().cpu()
            for t, p in zip(packet_labels_cpu.view(-1), packet_pred_cpu.view(-1)):
                if t < num_classes and p < num_classes:
                    packet_confusion_matrix[t.long(), p.long()] += 1

            if flow_ids is not None:
                flow_logits_m, flow_labels_m, _ = aggregate_logits_by_flow_tensor(
                    packet_logits=outputs.detach(),
                    packet_labels=labels.detach(),
                    flow_ids=flow_ids.detach(),
                    num_classes=num_classes,
                    method=flow_agg_method,
                    topk=flow_topk,
                    soft_temp=flow_soft_temp,
                    packet_repr=(packet_repr.detach() if packet_repr is not None else None),
                    packet_weighter=flow_packet_weighter,
                )
                if flow_logits_m.size(0) > 0:
                    _, flow_pred_m = torch.max(flow_logits_m.data, 1)
                    flow_labels_cpu_m = flow_labels_m.detach().cpu()
                    flow_pred_cpu_m = flow_pred_m.detach().cpu()
                    for t, p in zip(flow_labels_cpu_m.view(-1), flow_pred_cpu_m.view(-1)):
                        if t < num_classes and p < num_classes:
                            flow_confusion_matrix[t.long(), p.long()] += 1

        if train_target == 'flow':
            running_loss += classification_loss.item() * flow_labels.size(0)
            running_items += flow_labels.size(0)
            _, predicted = torch.max(flow_logits.data, 1)
            labels_cpu = flow_labels.detach().cpu()
            predicted_cpu = predicted.detach().cpu()
        else:
            running_loss += classification_loss.item() * labels.size(0)
            running_items += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            labels_cpu = labels.detach().cpu()
            predicted_cpu = predicted.detach().cpu()
        for t, p in zip(labels_cpu.view(-1), predicted_cpu.view(-1)):
            if t < num_classes and p < num_classes:
                confusion_matrix[t.long(), p.long()] += 1

     
    epoch_loss = running_loss / running_items if running_items > 0 else 0
    
    epoch_metrics = calculate_metrics(confusion_matrix)
    epoch_metrics['loss'] = epoch_loss
    if collect_dual_level_metrics:
        packet_metrics = calculate_metrics(packet_confusion_matrix)
        epoch_metrics['packet_accuracy'] = packet_metrics['accuracy']
        epoch_metrics['packet_f1_macro'] = packet_metrics['f1_macro']
        epoch_metrics['packet_f1_weighted'] = packet_metrics['f1_weighted']
        epoch_metrics['packet_samples'] = int(packet_confusion_matrix.sum().item())

        if flow_confusion_matrix.sum().item() > 0:
            flow_metrics = calculate_metrics(flow_confusion_matrix)
            epoch_metrics['flow_accuracy'] = flow_metrics['accuracy']
            epoch_metrics['flow_f1_macro'] = flow_metrics['f1_macro']
            epoch_metrics['flow_f1_weighted'] = flow_metrics['f1_weighted']
            epoch_metrics['flow_samples'] = int(flow_confusion_matrix.sum().item())
        else:
            epoch_metrics['flow_accuracy'] = float('nan')
            epoch_metrics['flow_f1_macro'] = float('nan')
            epoch_metrics['flow_f1_weighted'] = float('nan')
            epoch_metrics['flow_samples'] = 0
    else:
        epoch_metrics['packet_accuracy'] = float('nan')
        epoch_metrics['packet_f1_macro'] = float('nan')
        epoch_metrics['packet_f1_weighted'] = float('nan')
        epoch_metrics['packet_samples'] = 0
        epoch_metrics['flow_accuracy'] = float('nan')
        epoch_metrics['flow_f1_macro'] = float('nan')
        epoch_metrics['flow_f1_weighted'] = float('nan')
        epoch_metrics['flow_samples'] = 0
    
    return epoch_metrics, confusion_matrix


@torch.no_grad()   
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    num_classes: int, 
    loss_fn: nn.Module,
    eval_target: str = 'packet',
    flow_agg_method: str = 'mean_logits',
    flow_topk: int = 8,
    flow_soft_temp: float = 1.0,
    flow_packet_weighter: Optional[nn.Module] = None,
    collect_dual_level_metrics: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[Dict, torch.Tensor]:  #, torch.Tensor]:
    """
    Evaluate one epoch for the HierarchicalMoE model.
    The dataloader yields a dictionary of PyG batches.
    Returns metrics and confusion matrix.
    """
    model.eval()   
    if flow_packet_weighter is not None:
        flow_packet_weighter.eval()
    
    running_loss = 0.0
    running_items = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    packet_confusion_matrix = (
        torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
        if collect_dual_level_metrics else None
    )
    # Memory-safe accumulators for flow-level evaluation on large datasets.
    flow_num_dict: Dict[int, torch.Tensor] = {}
    flow_den_dict: Dict[int, float] = {}
    flow_topk_dict: Dict[int, List[Tuple[float, torch.Tensor]]] = {}
    flow_label_count_dict: Dict[int, Dict[int, int]] = {}
    batch_level_flow_eval = (eval_target == 'flow' and flow_agg_method == 'repr_logits_attn')
    
     
    # base_loss_fn = nn.CrossEntropyLoss()

    for batch_dict in tqdm(dataloader, desc="Evaluating"):
         
        # Move each item in batch_dict to device when supported.
         
        # This avoids assumptions about a specific batch container type.
        try:
            for key, value in batch_dict.items():
                if hasattr(value, 'to'):   
                    try:
                        batch_dict[key] = value.to(device, non_blocking=True)
                    except TypeError:
                        batch_dict[key] = value.to(device)
        except Exception as e:
              
             print(f"Warning: failed to move batch item '{key}' to device. Error: {e}")
        
         
        # labels = batch_dict['eth'].y 
        any_key = next(iter(batch_dict.keys()))
        labels = batch_dict[any_key].y
        flow_ids = batch_dict.get('flow_ids', None)

         
         
        FIELD_TO_IGNORE = 'tls.handshake.extensions_server_name'
        
         
        if 'tls_handshake' in batch_dict and hasattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE):
            try:
                 
                delattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE)
            except AttributeError:
                pass   
        # =====================================================================
        
         
        with torch.autocast(
            device_type='cuda',
            dtype=amp_dtype,
            enabled=(use_amp and device.type == 'cuda')
        ):
            need_packet_repr = (eval_target == 'flow' and flow_agg_method == 'repr_logits_attn')
            model_out = model(batch_dict, return_packet_repr=need_packet_repr)
            if need_packet_repr:
                outputs, _, packet_repr = model_out
            else:
                outputs, _ = model_out
                packet_repr = None

        if collect_dual_level_metrics:
            _, packet_pred = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), packet_pred.view(-1)):
                if t < num_classes and p < num_classes:
                    packet_confusion_matrix[t, p] += 1
        
         
        # loss = base_loss_fn(outputs, labels)
         
         
        if eval_target == 'flow':
            if flow_ids is None:
                raise RuntimeError("flow_ids missing in batch_dict while eval_target='flow'.")
            if batch_level_flow_eval:
                flow_logits_b, flow_labels_b, _ = aggregate_logits_by_flow_tensor(
                    packet_logits=outputs,
                    packet_labels=labels,
                    flow_ids=flow_ids,
                    num_classes=num_classes,
                    method=flow_agg_method,
                    topk=flow_topk,
                    soft_temp=flow_soft_temp,
                    packet_repr=packet_repr,
                    packet_weighter=flow_packet_weighter,
                )
                if flow_logits_b.size(0) > 0:
                    flow_loss_b = loss_fn(flow_logits_b, flow_labels_b)
                    running_loss += flow_loss_b.item() * flow_labels_b.size(0)
                    running_items += flow_labels_b.size(0)
                    _, flow_pred_b = torch.max(flow_logits_b.data, 1)
                    for t, p in zip(flow_labels_b.view(-1), flow_pred_b.view(-1)):
                        if t < num_classes and p < num_classes:
                            confusion_matrix[t, p] += 1
                continue

            logits_cpu = outputs.detach().float().cpu()
            labels_cpu = labels.detach().cpu()
            flow_ids_cpu = flow_ids.detach().cpu()

            if flow_agg_method == 'topk_mean_logits':
                confidences = logits_cpu.max(dim=1).values
                k = max(1, int(flow_topk))
                for i in range(logits_cpu.size(0)):
                    fid = int(flow_ids_cpu[i].item())
                    y = int(labels_cpu[i].item())
                    conf = float(confidences[i].item())
                    if fid not in flow_topk_dict:
                        flow_topk_dict[fid] = []
                        flow_label_count_dict[fid] = {}
                    flow_topk_dict[fid].append((conf, logits_cpu[i]))
                    flow_label_count_dict[fid][y] = flow_label_count_dict[fid].get(y, 0) + 1
                # Trim per-flow candidates to top-k incrementally.
                for fid in list(flow_topk_dict.keys()):
                    cand = flow_topk_dict[fid]
                    if len(cand) > k * 2:
                        cand.sort(key=lambda x: x[0], reverse=True)
                        flow_topk_dict[fid] = cand[:k]
            else:
                if flow_agg_method == 'mean_logits':
                    values_cpu = logits_cpu
                    weights_cpu = torch.ones(logits_cpu.size(0), dtype=torch.float32)
                elif flow_agg_method == 'mean_probs':
                    values_cpu = torch.softmax(logits_cpu, dim=1)
                    weights_cpu = torch.ones(logits_cpu.size(0), dtype=torch.float32)
                elif flow_agg_method == 'soft_weighted_logits':
                    temp = max(float(flow_soft_temp), 1e-4)
                    conf = logits_cpu.max(dim=1).values
                    weights_cpu = torch.exp(conf / temp).float()
                    values_cpu = logits_cpu
                elif flow_agg_method == 'learned_attn_logits':
                    if flow_packet_weighter is None:
                        raise RuntimeError("flow_packet_weighter is required for learned_attn_logits.")
                    scores = flow_packet_weighter(outputs.detach()).detach().float().cpu()
                    weights_cpu = torch.exp(scores).float()
                    values_cpu = logits_cpu
                elif flow_agg_method == 'logsumexp':
                    values_cpu = logits_cpu
                    weights_cpu = torch.ones(logits_cpu.size(0), dtype=torch.float32)
                else:
                    values_cpu = logits_cpu
                    weights_cpu = torch.ones(logits_cpu.size(0), dtype=torch.float32)

                for i in range(values_cpu.size(0)):
                    fid = int(flow_ids_cpu[i].item())
                    y = int(labels_cpu[i].item())
                    w = float(weights_cpu[i].item())
                    v = values_cpu[i]
                    if fid not in flow_num_dict:
                        if flow_agg_method == 'logsumexp':
                            flow_num_dict[fid] = v.clone()
                            flow_den_dict[fid] = 1.0
                        else:
                            flow_num_dict[fid] = v * w
                            flow_den_dict[fid] = w
                        flow_label_count_dict[fid] = {y: 1}
                    else:
                        if flow_agg_method == 'logsumexp':
                            flow_num_dict[fid] = torch.logaddexp(flow_num_dict[fid], v)
                            flow_den_dict[fid] += 1.0
                        else:
                            flow_num_dict[fid] += v * w
                            flow_den_dict[fid] += w
                        flow_label_count_dict[fid][y] = flow_label_count_dict[fid].get(y, 0) + 1
        else:
            loss = loss_fn(outputs, labels)   
            running_loss += loss.item() * labels.size(0)
            running_items += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                if t < num_classes and p < num_classes:
                    confusion_matrix[t, p] += 1

     
    
     
    if eval_target == 'flow' and not batch_level_flow_eval:
        flow_logits_list: List[torch.Tensor] = []
        flow_labels_list: List[int] = []

        if flow_agg_method == 'topk_mean_logits':
            k = max(1, int(flow_topk))
            for fid, cand in flow_topk_dict.items():
                if len(cand) == 0:
                    continue
                cand.sort(key=lambda x: x[0], reverse=True)
                chosen = cand[:k]
                logits_stack = torch.stack([x[1] for x in chosen], dim=0)
                flow_logits_list.append(logits_stack.mean(dim=0))
                label_count = flow_label_count_dict.get(fid, {})
                y = max(label_count.items(), key=lambda kv: kv[1])[0]
                flow_labels_list.append(int(y))
        elif flow_agg_method == 'logsumexp':
            for fid, lse in flow_num_dict.items():
                n = max(flow_den_dict.get(fid, 1.0), 1.0)
                flow_logits_list.append(lse - torch.log(torch.tensor(n, dtype=lse.dtype)))
                label_count = flow_label_count_dict.get(fid, {})
                y = max(label_count.items(), key=lambda kv: kv[1])[0]
                flow_labels_list.append(int(y))
        else:
            for fid, num in flow_num_dict.items():
                den = max(flow_den_dict.get(fid, 0.0), 1e-12)
                flow_logits_list.append(num / den)
                label_count = flow_label_count_dict.get(fid, {})
                y = max(label_count.items(), key=lambda kv: kv[1])[0]
                flow_labels_list.append(int(y))

        if len(flow_logits_list) > 0:
            flow_logits = torch.stack(flow_logits_list, dim=0).to(device)
            flow_labels = torch.tensor(flow_labels_list, dtype=torch.long, device=device)
            flow_loss = loss_fn(flow_logits, flow_labels)
            running_loss = flow_loss.item() * flow_labels.size(0)
            running_items = flow_labels.size(0)
            _, flow_pred = torch.max(flow_logits.data, 1)
            for t, p in zip(flow_labels.view(-1), flow_pred.view(-1)):
                if t < num_classes and p < num_classes:
                    confusion_matrix[t, p] += 1
        epoch_loss = running_loss / running_items if running_items > 0 else 0
    else:
        epoch_loss = running_loss / running_items if running_items > 0 else 0
    
     
     
    cm_cpu = confusion_matrix.cpu()
    epoch_metrics = calculate_metrics(cm_cpu)   
    epoch_metrics['loss'] = epoch_loss
    if collect_dual_level_metrics:
        packet_cm_cpu = packet_confusion_matrix.cpu()
        packet_metrics = calculate_metrics(packet_cm_cpu)
        epoch_metrics['packet_accuracy'] = packet_metrics['accuracy']
        epoch_metrics['packet_f1_macro'] = packet_metrics['f1_macro']
        epoch_metrics['packet_f1_weighted'] = packet_metrics['f1_weighted']
        epoch_metrics['packet_samples'] = int(packet_cm_cpu.sum().item())

        if eval_target == 'flow':
            # Keep identical global flow definition as main metric.
            epoch_metrics['flow_accuracy'] = epoch_metrics['accuracy']
            epoch_metrics['flow_f1_macro'] = epoch_metrics['f1_macro']
            epoch_metrics['flow_f1_weighted'] = epoch_metrics['f1_weighted']
            epoch_metrics['flow_samples'] = int(cm_cpu.sum().item())
        else:
            epoch_metrics['flow_accuracy'] = float('nan')
            epoch_metrics['flow_f1_macro'] = float('nan')
            epoch_metrics['flow_f1_weighted'] = float('nan')
            epoch_metrics['flow_samples'] = 0
    else:
        epoch_metrics['packet_accuracy'] = float('nan')
        epoch_metrics['packet_f1_macro'] = float('nan')
        epoch_metrics['packet_f1_weighted'] = float('nan')
        epoch_metrics['packet_samples'] = 0
        epoch_metrics['flow_accuracy'] = float('nan')
        epoch_metrics['flow_f1_macro'] = float('nan')
        epoch_metrics['flow_f1_weighted'] = float('nan')
        epoch_metrics['flow_samples'] = 0
    
     
    # tp = confusion_matrix.diag()
    # fp = confusion_matrix.sum(dim=0) - tp
    # fn = confusion_matrix.sum(dim=1) - tp
     
    
    # precision = tp / (tp + fp + epsilon)
    # recall = tp / (tp + fn + epsilon)
    
     
    # per_class_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
     
    return epoch_metrics, cm_cpu, # per_class_f1.cpu()


@torch.no_grad()
def collect_packet_logits_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect packet-level logits and labels in dataloader iteration order.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch_dict in tqdm(dataloader, desc="Collecting packet logits"):
        for key, value in batch_dict.items():
            if hasattr(value, 'to'):
                batch_dict[key] = value.to(device)

        any_key = next(iter(batch_dict.keys()))
        labels = batch_dict[any_key].y

        field_to_ignore = 'tls.handshake.extensions_server_name'
        if 'tls_handshake' in batch_dict and hasattr(batch_dict['tls_handshake'], field_to_ignore):
            try:
                delattr(batch_dict['tls_handshake'], field_to_ignore)
            except AttributeError:
                pass

        outputs, _ = model(batch_dict)
        all_logits.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if len(all_logits) == 0:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def evaluate_flow_aggregation_from_packet_logits(
    packet_logits: torch.Tensor,
    packet_labels: torch.Tensor,
    flow_ids: List[str],
    num_classes: int,
    use_prob_mean: bool = True,
) -> Tuple[Dict, torch.Tensor]:
    """
    Build flow-level predictions by aggregating packet-level outputs:
      - default: mean of packet probabilities per flow.
      - optional: mean of packet logits per flow.
    """
    if packet_logits.ndim != 2:
        raise ValueError("packet_logits must be a 2D tensor [N, C].")
    if packet_labels.ndim != 1:
        raise ValueError("packet_labels must be a 1D tensor [N].")
    if len(flow_ids) != packet_logits.size(0):
        raise ValueError("Length of flow_ids must equal number of packet logits.")

    flow_to_indices: Dict[str, List[int]] = {}
    for idx, fid in enumerate(flow_ids):
        k = str(fid)
        if k not in flow_to_indices:
            flow_to_indices[k] = []
        flow_to_indices[k].append(idx)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    inconsistent_flows = 0

    for fid, indices in flow_to_indices.items():
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        flow_logits = packet_logits.index_select(0, idx_tensor)
        flow_labels = packet_labels.index_select(0, idx_tensor)

        unique_labels = torch.unique(flow_labels)
        if unique_labels.numel() == 1:
            true_label = int(unique_labels.item())
        else:
            # Safety fallback if a flow has mixed labels.
            binc = torch.bincount(flow_labels, minlength=num_classes)
            true_label = int(torch.argmax(binc).item())
            inconsistent_flows += 1

        if use_prob_mean:
            agg_score = torch.softmax(flow_logits, dim=1).mean(dim=0)
        else:
            agg_score = flow_logits.mean(dim=0)
        pred_label = int(torch.argmax(agg_score).item())

        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1

    metrics = calculate_metrics(cm)
    metrics['num_flows'] = len(flow_to_indices)
    metrics['inconsistent_flows'] = inconsistent_flows
    return metrics, cm


def compute_dataset_expert_importance(model, dataloader, device):
    """
    Compute dataset-level expected expert importance:
    \bar{omega}_k = E_x[ omega_k(x) ]

    Args:
        model: trained HierarchicalMoE model
        dataloader: validation or test loader
        device: torch device

    Returns:
        torch.Tensor of shape [num_experts]
    """

    model.eval()

    total_weights = None
    total_samples = 0

    with torch.no_grad():
        for batch_dict in dataloader:

            # ================================
            # Move batch to device (same as evaluate)
            # ================================
            for key, value in batch_dict.items():
                if hasattr(value, 'to'):
                    batch_dict[key] = value.to(device)

            # Forward pass
            logits, _ = model(batch_dict)

             
             
            weights = model._latest_expert_weights  # shape [B, K]

            if weights is None:
                raise RuntimeError("Expert weights not found. "
                                   "Ensure forward() stores _latest_expert_weights.")

             
            if total_weights is None:
                total_weights = weights.sum(dim=0)
            else:
                total_weights += weights.sum(dim=0)

            total_samples += weights.shape[0]

     
    expected_weights = total_weights / total_samples

    return expected_weights.cpu()


# =====================================================================
if __name__ == '__main__':
    SEED = 42
    DETERMINISTIC_TRAINING = False
    set_seed(SEED, deterministic=DETERMINISTIC_TRAINING)

     
    NUM_EPOCHS = 100
    SCALE_FACTOR = 1
    BATCH_SIZE = 1024 // SCALE_FACTOR
    # BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_LEARNING_RATE = 1e-3
    DROPOUT_RATE = 0.1
    NUM_WORKERS = 4
    USE_AMP = True
    AMP_DTYPE_STR = 'fp16'  # 'fp16' or 'bf16'
    THROUGHPUT_PROFILE = 'cuda_low_ram'  # 'base' | 'high' | 'cuda_low_ram'
    DATALOADER_PREFETCH = 2
    PERSISTENT_WORKERS = True
    GNN_INPUT_DIM = 32 
    GNN_HIDDEN_DIM = 128
    PATIENCE = 5
    DIAGNOSE = False
    # DIAGNOSE = True
    stop_training = False

    # USE_FLOW_FEATURES_THIS_RUN = True
    USE_FLOW_FEATURES_THIS_RUN = False
    # USE_MAC_ADDRESS_THIS_RUN = True
    USE_MAC_ADDRESS_THIS_RUN = False
    # USE_IP_ADDRESS_THIS_RUN = True
    USE_IP_ADDRESS_THIS_RUN = False
    # USE_PORT_THIS_RUN = True
    USE_PORT_THIS_RUN = False
    ENABLE_FLOW_AGG_EVAL = True
    FLOW_AGG_USE_PROB_MEAN = True
    ENABLE_DIRECTIONAL_FLOW_CONTEXT = True
    # Lightweight mode: disable dual packet/flow metrics during epoch to reduce overhead.
    ENABLE_DUAL_LEVEL_METRICS = False
    TRAIN_TARGET = 'flow'  # 'packet' or 'flow' 
    FLOW_AGG_METHOD = 'repr_logits_attn'  # 'mean_logits' | 'learned_attn_logits' | 'repr_logits_attn' | 'soft_weighted_logits' | 'topk_mean_logits' | 'mean_probs' | 'logsumexp'
    FLOW_TOPK = 8
    FLOW_SOFT_TEMP = 1.2
    FLOW_ATTN_HIDDEN_DIM = 64
    FLOW_ATTN_DROPOUT = 0.1
    FLOW_REPR_LOGIT_USE_PROBS = True
    FLOW_REPR_USE_LOGIT_BRANCH = False
    FLOW_REPR_ATTN_TAU = 1.5
    FLOW_REPR_USE_RESIDUAL_MEAN = False
    FLOW_REPR_USE_LOGIT_RESIDUAL_BASELINE = False
    FLOW_REPR_LOGIT_RESIDUAL_INIT = 0.0
    FLOW_REPR_FIRST_N_PACKETS = 64
    FLOW_REPR_POOL_MODE = 'max'  # 'attn' | 'max' | 'mean'
    FLOW_REPR_DETACH_WARMUP_EPOCHS = 3
    FLOW_REPR_PERMANENT_DETACH = True
    FLOW_REPR_TRAIN_AGG_ONLY_EPOCHS = 5
    FLOW_REPR_GAMMA_CLAMP_MAX = 0.2
    FLOW_LOSS_USE_PACKET_AUX = True
    FLOW_PACKET_AUX_WEIGHT = 0.3
    STRATIFIED_TRAIN_SET = False
    STRATIFIED_VAL_TEST_SET = False
    # STRATIFIED_TRAIN_SET = True
    # STRATIFIED_VAL_TEST_SET = True
    SAMPLING_PROPORTION = 0.6
    SAMPLING_GRANULARITY = 'flow'  # 'flow' or 'packet'
    MIN_FLOWS_PER_CLASS = 5
    MAX_PACKETS_PER_FLOW_TRAIN = 256
    # MAX_PACKETS_PER_FLOW_EVAL = 256
    MAX_PACKETS_PER_FLOW_EVAL = MAX_PACKETS_PER_FLOW_TRAIN
    PREFER_LONG_FLOWS = True
    LONG_FLOW_PRIORITY_RATIO = 0.5
    # ABLATION_LAYERS = ['eth', 'ip', 'tcp', 'tls']
    ABLATION_LAYERS = ['ip', 'tcp', 'tls']
    BACKBONE_MODE = 'expert_local'  # 'expert_local' | 'global'

    FILTER_SHORT_ENTRIES = False

    if THROUGHPUT_PROFILE == 'high':
        BATCH_SIZE = 2048 // SCALE_FACTOR
        NUM_WORKERS = max(NUM_WORKERS, min(8, (os.cpu_count() or 8)))
        print("[Throughput] High profile enabled.")
    elif THROUGHPUT_PROFILE == 'cuda_low_ram':
        # Keep memory footprint moderate while maintaining GPU-friendly throughput.
        BATCH_SIZE = 1024 // SCALE_FACTOR
        NUM_WORKERS = min(NUM_WORKERS, max(2, min(6, (os.cpu_count() or 6))))
        DATALOADER_PREFETCH = 2
        print("[Throughput] CUDA low-RAM profile enabled.")

    OBFUSCATION_CONFIG = {
        "len_noise": 0.1,
        "iat_noise": 0.005,
    }
     
    FOCAL_GAMMA = 2.0
    # Stabilize focal alpha on long-tail data:
    # 'none' keeps inverse-frequency as-is;
    # 'sqrt' applies sqrt smoothing; 'log' applies log1p smoothing.
    FOCAL_ALPHA_SMOOTHING = 'sqrt'  # 'none' | 'sqrt' | 'log'
    # Clamp alpha to avoid extreme gradient spikes on ultra-rare classes.
    FOCAL_ALPHA_CLAMP_MIN = 0.1
    FOCAL_ALPHA_CLAMP_MAX = 10.0
    MASK_ENTROPY_REG_ALPHA = 1e-5
    EXPERT_GATE_NOISE_STD = 0.01
    EPSILON = 1e-6 

    ROLLBACK_PATIENCE = NUM_EPOCHS // 10
    EARLY_STOP_PATIENCE = ROLLBACK_PATIENCE + 5
    MIN_LR_FOR_TRAINING = 1e-6
    LOG_FLUSH_EVERY_EPOCH = 1
    print(f"Batch size: {BATCH_SIZE}; Learning rate: {LEARNING_RATE}")
    print(f"Backbone mode: {BACKBONE_MODE}")
    print(
        f"Throughput profile: {THROUGHPUT_PROFILE}; workers: {NUM_WORKERS}; "
        f"prefetch: {DATALOADER_PREFETCH}; persistent_workers: {PERSISTENT_WORKERS}"
    )
    print(f"Flow aggregation: {FLOW_AGG_METHOD} (topk={FLOW_TOPK}, soft_temp={FLOW_SOFT_TEMP})")
    print(f"Flow attn config: hidden={FLOW_ATTN_HIDDEN_DIM}, dropout={FLOW_ATTN_DROPOUT}")
    print(
        f"repr_logits_attn config: use_probs={FLOW_REPR_LOGIT_USE_PROBS}, "
        f"use_logit_branch={FLOW_REPR_USE_LOGIT_BRANCH}, "
        f"residual_mean={FLOW_REPR_USE_RESIDUAL_MEAN}, "
        f"logit_residual_baseline={FLOW_REPR_USE_LOGIT_RESIDUAL_BASELINE}, "
        f"logit_residual_init={FLOW_REPR_LOGIT_RESIDUAL_INIT}, "
        f"first_n={FLOW_REPR_FIRST_N_PACKETS}, pool_mode={FLOW_REPR_POOL_MODE}, "
        f"tau={FLOW_REPR_ATTN_TAU}, detach_warmup_epochs={FLOW_REPR_DETACH_WARMUP_EPOCHS}, "
        f"permanent_detach={FLOW_REPR_PERMANENT_DETACH}, "
        f"agg_only_epochs={FLOW_REPR_TRAIN_AGG_ONLY_EPOCHS}, "
        f"gamma_clamp_max={FLOW_REPR_GAMMA_CLAMP_MAX}"
    )
    print(
        f"Focal alpha config: smoothing={FOCAL_ALPHA_SMOOTHING}, "
        f"clamp=[{FOCAL_ALPHA_CLAMP_MIN}, {FOCAL_ALPHA_CLAMP_MAX}]"
    )
    print(
        f"Mask entropy alpha: {MASK_ENTROPY_REG_ALPHA}; "
        f"expert gate noise std: {EXPERT_GATE_NOISE_STD}"
    )
     
     
    # dataset_name = 'ISCX-VPN'
    # dataset_name = 'ISCX-TOR-Acctivity'
    # dataset_name = 'ISCX-TOR-Application'
    # dataset_name = 'USTC-TFC2016-Benign'
    # dataset_name = 'dataset_29_d1' 
    # dataset_name = 'dataset_20_d2'
    # dataset_name = 'USTC-TFC2016-Malware'
    dataset_name = 'cstnet_tls_1.3'
    # dataset_name = 'CipherSpectrum'
    root_path = os.path.join('..', 'TrafficData', 'datasets_csv_add2')
    val_test_dir = os.path.join(root_path, 'datasets_split', dataset_name) 
    train_dir = os.path.join(root_path, 'datasets_final')
    vocab_dir = os.path.join(root_path, 'categorical_vocabs')
    config_path = os.path.join('.', 'Data', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join(vocab_dir, dataset_name + '_vocabs.yaml') 
    res_path = os.path.join('..', 'Res')
    # train_set_name = dataset_name + '_chief_block_augmented'
    # train_set_name = dataset_name + '_chief_block_topk_augmented'
    train_set_name = 'train_set'
    val_set_name = 'validation_set' 
    test_set_name = 'test_set'
    # chief_directory = train_dir
    chief_directory = val_test_dir
    val_test_directory = val_test_dir

    upperlayer_dir = os.path.join(root_path, 'datasets_upperlayer', dataset_name) 

    if not FILTER_SHORT_ENTRIES: 
        train_df_path = os.path.join(chief_directory, train_set_name + '.csv') 
        val_df_path = os.path.join(val_test_directory, val_set_name + '.csv')
        test_df_path = os.path.join(val_test_directory, test_set_name + '.csv')
        SOURCE_CSV_PATH = os.path.join(root_path, 'datasets_consolidate', dataset_name + '.csv')
    else: 
        print("USE UPPER FILTER!!!")
        train_df_path = os.path.join(upperlayer_dir, train_set_name + '.csv') 
        val_df_path = os.path.join(upperlayer_dir, val_set_name + '.csv')
        test_df_path = os.path.join(upperlayer_dir, test_set_name + '.csv')

    GLOBAL_CHIEF_SCHEMA = None

    def print_sampling_summary(df: pd.DataFrame, split_name: str):
        print(f"\n[Sampling Summary] {split_name}")
        if df is None or df.empty:
            print(" - Empty dataframe.")
            return
        if 'label' not in df.columns:
            print(" - Missing column: label")
            return
        if 'stream_id' not in df.columns:
            print(" - Missing column: stream_id")
            return

        total_packets = len(df)
        total_flows = df['stream_id'].nunique(dropna=True)
        print(f" - Total packets: {total_packets}")
        print(f" - Total flows: {total_flows}")
        print(f" - Total classes: {df['label'].nunique(dropna=True)}")

        rows = []
        grouped = df.groupby('label', dropna=False)
        for label, g in grouped:
            rows.append({
                'label': label,
                'flows': g['stream_id'].nunique(dropna=True),
                'packets': len(g),
            })

        if not rows:
            print(" - No class rows to display.")
            return

        stat_df = pd.DataFrame(rows).sort_values(by='packets', ascending=False).reset_index(drop=True)
        stat_df['flow_ratio'] = (stat_df['flows'] / max(total_flows, 1)).map(lambda x: f"{x:.4f}")
        stat_df['packet_ratio'] = (stat_df['packets'] / max(total_packets, 1)).map(lambda x: f"{x:.4f}")
        print(stat_df.to_string(index=False))

    def cap_packets_per_flow(df: pd.DataFrame, split_name: str, cap_per_flow: int = None) -> pd.DataFrame:
        if cap_per_flow is None or cap_per_flow <= 0:
            return df
        if df is None or df.empty:
            return df
        if 'stream_id' not in df.columns:
            print(f"[Packet Cap] {split_name}: skip (missing 'stream_id').")
            return df

        before_packets = len(df)
        before_flows = df['stream_id'].nunique(dropna=True)
        shuffled = df.sample(frac=1.0, random_state=SEED).copy()
        shuffled['_flow_pkt_rank'] = shuffled.groupby('stream_id').cumcount()
        capped = shuffled[shuffled['_flow_pkt_rank'] < cap_per_flow].drop(columns=['_flow_pkt_rank'])
        capped = capped.reset_index(drop=True)

        after_packets = len(capped)
        after_flows = capped['stream_id'].nunique(dropna=True)
        print(
            f"[Packet Cap] {split_name}: cap={cap_per_flow}, "
            f"packets {before_packets}->{after_packets}, flows {before_flows}->{after_flows}"
        )
        return capped

    def load_sampled_split(csv_path: str) -> pd.DataFrame:
        if SAMPLING_GRANULARITY == 'flow':
            return stratified_flow_sample_from_csv_stream(
                csv_path=csv_path,
                label_column='label',
                flow_id_column='stream_id',
                proportion=SAMPLING_PROPORTION,
                chunksize=200000,
                random_state=SEED,
                min_flows_per_class=MIN_FLOWS_PER_CLASS,
                prefer_long_flows=PREFER_LONG_FLOWS,
                long_flow_priority_ratio=LONG_FLOW_PRIORITY_RATIO,
                read_csv_kwargs={
                    "dtype": str,
                    "low_memory": False,
                },
            )
        return stratified_hybrid_sample_from_csv_stream(
            csv_path=csv_path,
            label_column='label',
            proportion=SAMPLING_PROPORTION,
            chunksize=200000,
            random_state=SEED,
            read_csv_kwargs={
                "dtype": str,
                "low_memory": False,
            },
        )

    if STRATIFIED_TRAIN_SET: 
        train_df = load_sampled_split(train_df_path)

    else: 
         
        print("\n[1/4] Loading datasets...")
        try:
            train_df = pd.read_csv(train_df_path, dtype=str)
            # val_df = pd.read_csv(val_df_path, dtype=str)
            # test_df = pd.read_csv(test_df_path, dtype=str)
        except FileNotFoundError as e:
            print(f"Error: dataset file not found. Please finish preprocessing first. {e}")
            exit()

    if STRATIFIED_VAL_TEST_SET: 
        val_df = load_sampled_split(val_df_path)

        test_df = load_sampled_split(test_df_path)
    else: 
        try:
            val_df = pd.read_csv(val_df_path, dtype=str)
            test_df = pd.read_csv(test_df_path, dtype=str)
        except FileNotFoundError as e:
            print(f"Error: dataset file not found. Please finish preprocessing first. {e}")
            exit()
        
    print(f" - Train set: {len(train_df)} rows")
    print(f" - Validation set: {len(val_df)} rows")
    print(f" - Test set: {len(test_df)} rows")
    print_sampling_summary(train_df, "Train")
    print_sampling_summary(val_df, "Validation")
    print_sampling_summary(test_df, "Test")

    train_df_full_for_flow_stats = train_df.copy(deep=True) if USE_FLOW_FEATURES_THIS_RUN else None
    val_df_full_for_flow_stats = val_df.copy(deep=True) if USE_FLOW_FEATURES_THIS_RUN else None
    test_df_full_for_flow_stats = test_df.copy(deep=True) if USE_FLOW_FEATURES_THIS_RUN else None

    train_df = cap_packets_per_flow(train_df, "Train", MAX_PACKETS_PER_FLOW_TRAIN)
    val_df = cap_packets_per_flow(val_df, "Validation", MAX_PACKETS_PER_FLOW_EVAL)
    test_df = cap_packets_per_flow(test_df, "Test", MAX_PACKETS_PER_FLOW_EVAL)

    print_sampling_summary(train_df, "Train (after packet cap)")
    print_sampling_summary(val_df, "Validation (after packet cap)")
    print_sampling_summary(test_df, "Test (after packet cap)")

    
    print("\n[3/4] Creating label mapping...")
    labels = train_df['label'].unique() # ?
    label_to_int = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    train_df['label_id'] = train_df['label'].map(label_to_int)


     
    print("\n[2/4] Aligning feature space for validation and test sets...")
    chief_schema = [col for col in train_df.columns if col not in ['label', 'label_id']]

    GLOBAL_CHIEF_SCHEMA = chief_schema
    
     
    val_df_aligned = val_df.reindex(columns=chief_schema, fill_value='0')
    val_df_aligned['label'] = val_df['label']
    
    test_df_aligned = test_df.reindex(columns=chief_schema, fill_value='0')
    test_df_aligned['label'] = test_df['label']
    
    print(" - Feature alignment complete.")
    # ==============================================================
    del val_df, test_df

    val_df_aligned['label_id'] = val_df_aligned['label'].map(label_to_int).fillna(-1).astype(int)   
    test_df_aligned['label_id'] = test_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) 
    test_stream_ids_for_eval = test_df_aligned['stream_id'].astype(str).tolist()
    
    if USE_FLOW_FEATURES_THIS_RUN:
        print("\n[2.5/4] Performing Feature Engineering 5.0 (Correct OPEN_WORLD branching)...")

         
        # flow_feature_names = [
        #     'flow_avg_len', 'flow_std_len', 'flow_pkt_count',
        #     'flow_avg_iat', 'flow_std_iat', 'flow_max_iat',
         
         
        # ]
        def calculate_flow_stats(df: pd.DataFrame, is_train_set: bool = False):
            """
            ?DataFrame ?*? ?

            -  is_train_set=True,  (maps, defaults)
            -  is_train_set=False, ?maps
            - ?flow 
            - 
            """

            print(f"   -> Calculating stats for DataFrame (size: {len(df)})...")

            # ======================
             
            # ======================
            has_ip_len = 'ip.len' in df.columns
            has_stream_id = 'stream_id' in df.columns

            if not (has_ip_len and has_stream_id):
                print("   [Warning] Missing 'ip.len' or 'stream_id'. Skip flow feature stats.")
                 
                if is_train_set:
                    return {}, {}
                else:
                    return {}

            # ======================
             
            # ======================
            if 'ip.len_temp_dec' not in df.columns:
                df['ip.len_temp_dec'] = df['ip.len'].apply(robust_hex_to_int).astype(np.float32)

            if 'stream_id_temp_dec' not in df.columns:
                df['stream_id_temp_dec'] = pd.to_numeric(df['stream_id'], errors='coerce').fillna(-1).astype(np.int64)

            # ======================
             
            # ======================
            has_time = True

            if 'time_temp_dec' in df.columns:
                time_col_used = 'time_temp_dec'
            else:
                candidate_time_cols = [
                    'tcp.options.timestamp',   
                    'frame.time_relative',   
                    'frame.time_epoch',   
                ]
                time_col = None
                for cand in candidate_time_cols:
                    if cand in df.columns:
                        time_col = cand
                        break
                    
                if time_col is None:
                    has_time = False
                    time_col_used = None
                    print("     [Warning] No time column found. Skip IAT/duration features.")
                else:
                    print(f"     -> Using time column '{time_col}' to compute IAT/duration features...")
                    if time_col == 'tcp.options.timestamp':
                        df['time_temp_dec'] = df[time_col].apply(robust_timestamp_to_tsval).astype(np.int64)
                    else:
                        df['time_temp_dec'] = (
                            pd.to_numeric(df[time_col], errors='coerce')
                              .fillna(0)
                              .astype(np.float64)
                        )
                    time_col_used = 'time_temp_dec'

            # ======================
             
            # ======================
            maps = {}

            print("     -> Calculating base length/count features...")
            grouped_len = df.groupby('stream_id_temp_dec')

             
            maps['flow_avg_len']   = grouped_len['ip.len_temp_dec'].mean()
            maps['flow_std_len']   = grouped_len['ip.len_temp_dec'].std().fillna(0)
            maps['flow_pkt_count'] = grouped_len['ip.len_temp_dec'].count()

             
            LARGE_PKT_THRESHOLD = 1400
            large_pkts = df[df['ip.len_temp_dec'] > LARGE_PKT_THRESHOLD]
            base_large_pkt_count = (
                large_pkts
                .groupby('stream_id_temp_dec')
                .size()
                .reindex(maps['flow_pkt_count'].index, fill_value=0)
            )
            print("     -> Calculating large packet ratio...")
            maps['flow_large_pkt_ratio'] = base_large_pkt_count / (maps['flow_pkt_count'] + EPSILON)

            # ======================
             
            # ======================
            if has_time and time_col_used is not None:
                print("     -> Calculating IAT & duration based features...")

                 
                df_sorted = df.sort_values(by=['stream_id_temp_dec', time_col_used]).copy()
                df_sorted['iat_temp'] = (
                    df_sorted
                    .groupby('stream_id_temp_dec')[time_col_used]
                    .diff()
                    .fillna(0)
                )

                 
                grouped_time = df_sorted.groupby('stream_id_temp_dec')

                maps['flow_avg_iat'] = grouped_time['iat_temp'].mean()
                maps['flow_std_iat'] = grouped_time['iat_temp'].std().fillna(0)
                maps['flow_max_iat'] = grouped_time['iat_temp'].max()

                # 4.3 duration_per_pkt
                flow_min_time = grouped_time[time_col_used].min()
                flow_max_time = grouped_time[time_col_used].max()
                base_duration = (flow_max_time - flow_min_time)

                maps['flow_duration_per_pkt'] = base_duration / (maps['flow_pkt_count'] + EPSILON)
            else:
                print("     [Info] Time-based flow features were not computed.")

            # ======================
             
            # ======================
            if is_train_set:
                print("     -> Calculating global defaults (using MEDIAN where appropriate)...")
                defaults = {}

                for f_name, stat_series in maps.items():
                    if ('count' in f_name) or ('ratio' in f_name):
                        default_val = float(stat_series.mean())
                    else:
                        default_val = float(stat_series.median())

                    if np.isnan(default_val) or np.isinf(default_val):
                        default_val = 0.0

                    defaults[f_name] = default_val

                return maps, defaults
            else:
                return maps


         
        
         
        print(" -> [FLOW ENHANCE MODE] Computing flow stats from full sampled flows per split...")

        if train_df_full_for_flow_stats is None:
            train_df_full_for_flow_stats = train_df.copy(deep=True)
        if val_df_full_for_flow_stats is None:
            val_df_full_for_flow_stats = val_df_aligned.copy(deep=True)
        if test_df_full_for_flow_stats is None:
            test_df_full_for_flow_stats = test_df_aligned.copy(deep=True)

        train_maps, train_defaults = calculate_flow_stats(train_df_full_for_flow_stats, is_train_set=True)
        if len(train_maps) == 0:
            raise RuntimeError(
                "[Fatal] Cannot compute flow features. Required columns include "
                "'ip.len' and 'stream_id' (with optional time columns)."
            )

        flow_feature_names = list(train_maps.keys())
        print(" -> Available flow features:", flow_feature_names)

        def build_defaults_from_maps(maps: Dict[str, pd.Series]) -> Dict[str, float]:
            defaults = {}
            for f_name, stat_series in maps.items():
                if ('count' in f_name) or ('ratio' in f_name):
                    default_val = float(stat_series.mean())
                else:
                    default_val = float(stat_series.median())
                if np.isnan(default_val) or np.isinf(default_val):
                    default_val = 0.0
                defaults[f_name] = default_val
            return defaults

        def add_flow_directional_context_features(
            df: pd.DataFrame,
            split_name: str,
        ) -> List[str]:
            """
            Add per-packet flow-context features:
            - direction feature (forward/reverse sign)
            - local iat feature
            """
            if df is None or df.empty:
                return []
            if 'stream_id' not in df.columns or 'ip.len' not in df.columns:
                print(f"   -> [{split_name}] Skip directional context: missing stream_id/ip.len.")
                return []

            if 'stream_id_temp_dec' not in df.columns:
                df['stream_id_temp_dec'] = pd.to_numeric(df['stream_id'], errors='coerce').fillna(-1).astype(np.int64)
            if 'ip.len_temp_dec' not in df.columns:
                df['ip.len_temp_dec'] = df['ip.len'].apply(robust_hex_to_int).astype(np.float32)

            if 'time_temp_dec' in df.columns:
                time_col_used = 'time_temp_dec'
            else:
                time_col_used = None
                for cand in ['tcp.options.timestamp', 'frame.time_relative', 'frame.time_epoch']:
                    if cand in df.columns:
                        if cand == 'tcp.options.timestamp':
                            df['time_temp_dec'] = df[cand].apply(robust_timestamp_to_tsval).astype(np.int64)
                        else:
                            df['time_temp_dec'] = pd.to_numeric(df[cand], errors='coerce').fillna(0).astype(np.float64)
                        time_col_used = 'time_temp_dec'
                        break

            sort_cols = ['stream_id_temp_dec']
            if time_col_used is not None:
                sort_cols.append(time_col_used)

            base_cols = ['stream_id_temp_dec', 'ip.len_temp_dec']
            if time_col_used is not None:
                base_cols.append(time_col_used)
            if 'ip.src' in df.columns and 'ip.dst' in df.columns:
                base_cols.extend(['ip.src', 'ip.dst'])
            elif 'tcp.srcport' in df.columns and 'tcp.dstport' in df.columns:
                base_cols.extend(['tcp.srcport', 'tcp.dstport'])

            work = df[base_cols].copy()
            work['_orig_idx'] = np.arange(len(work), dtype=np.int64)
            work = work.sort_values(sort_cols, kind='mergesort')
            grouped = work.groupby('stream_id_temp_dec', sort=False)

            # Direction sign: +1 means same direction as first packet of this flow.
            if 'ip.src' in work.columns and 'ip.dst' in work.columns:
                src = work['ip.src'].fillna('').astype(str)
                dst = work['ip.dst'].fillna('').astype(str)
                first_src = grouped['ip.src'].transform('first').fillna('').astype(str)
                first_dst = grouped['ip.dst'].transform('first').fillna('').astype(str)
                work['flow_dir_sign'] = np.where((src == first_src) & (dst == first_dst), 1.0, -1.0).astype(np.float32)
            elif 'tcp.srcport' in work.columns and 'tcp.dstport' in work.columns:
                src = work['tcp.srcport'].fillna('').astype(str)
                dst = work['tcp.dstport'].fillna('').astype(str)
                first_src = grouped['tcp.srcport'].transform('first').fillna('').astype(str)
                first_dst = grouped['tcp.dstport'].transform('first').fillna('').astype(str)
                work['flow_dir_sign'] = np.where((src == first_src) & (dst == first_dst), 1.0, -1.0).astype(np.float32)
            else:
                work['flow_dir_sign'] = np.zeros(len(work), dtype=np.float32)

            feature_names = ['flow_dir_sign']

            if time_col_used is not None:
                work['flow_iat_local'] = grouped[time_col_used].diff().fillna(0).astype(np.float32)
                feature_names.append('flow_iat_local')
            else:
                work['flow_iat_local'] = np.zeros(len(work), dtype=np.float32)
                feature_names.append('flow_iat_local')

            work = work.sort_values('_orig_idx')
            for f_name in feature_names:
                df[f_name] = work[f_name].to_numpy()

            print(f"   -> [{split_name}] Added directional context features: {feature_names}")
            return feature_names

        split_full_and_target = [
            ('Train', train_df_full_for_flow_stats, train_df, train_maps, train_defaults),
            ('Validation', val_df_full_for_flow_stats, val_df_aligned, None, None),
            ('Test', test_df_full_for_flow_stats, test_df_aligned, None, None),
        ]

        context_feature_names: List[str] = []

        for split_name, full_df, target_df, known_maps, known_defaults in split_full_and_target:
            if target_df is None or target_df.empty:
                continue

            print(f"   -> Processing {split_name} set...")
            split_maps = known_maps if known_maps is not None else calculate_flow_stats(full_df, is_train_set=False)
            split_defaults = known_defaults if known_defaults is not None else build_defaults_from_maps(split_maps)

            if 'stream_id_temp_dec' not in target_df.columns:
                target_df['stream_id_temp_dec'] = pd.to_numeric(
                    target_df['stream_id'], errors='coerce'
                ).fillna(-1).astype(np.int64)

            for f_name in flow_feature_names:
                feature_map = split_maps.get(f_name)
                if feature_map is None:
                    target_df[f_name] = split_defaults.get(f_name, 0.0)
                else:
                    target_df[f_name] = target_df['stream_id_temp_dec'].map(feature_map).fillna(
                        split_defaults.get(f_name, 0.0)
                    )

            if ENABLE_DIRECTIONAL_FLOW_CONTEXT:
                added_features = add_flow_directional_context_features(
                    target_df,
                    split_name=split_name,
                )
                if added_features:
                    context_feature_names = list(dict.fromkeys(context_feature_names + added_features))

        if ENABLE_DIRECTIONAL_FLOW_CONTEXT and context_feature_names:
            flow_feature_names = list(dict.fromkeys(flow_feature_names + context_feature_names))

        del train_df_full_for_flow_stats, val_df_full_for_flow_stats, test_df_full_for_flow_stats
        gc.collect()

        # d)
        print(" -> Cleaning up temporary decimal columns...")
        for df in [train_df, val_df_aligned, test_df_aligned]:
            if df is None or df.empty:
                continue
            for tmp_col in ['ip.len_temp_dec', 'stream_id_temp_dec', 'time_temp_dec', 'iat_temp']:
                if tmp_col in df.columns: 
                    df.drop(columns=[tmp_col], inplace=True)
        
         
        # print(" -> Final cleanup of *new* flow features (nan/inf/neginf -> 0.0)...")
        # for df in [train_df, val_df_aligned, test_df_aligned]:
        #      if df.empty: continue
        #      for col in flow_feature_names: 
         
        #          col_data_numeric = pd.to_numeric(df[col], errors='coerce')
        #          col_data_np = col_data_numeric.values
        #          col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0) 
        #          if 'count' in col: df[col] = col_data_cleaned.astype(np.int32)
        #          else: df[col] = col_data_cleaned.astype(np.float32)
                     
        # print(f" -> Feature Engineering 5.0 complete. {len(flow_feature_names)} flow features created.")
        print(" -> Final cleanup of *new* flow features (nan/inf/neginf -> 0.0)...")

        dfs = [
            (train_df,       "train_df"),
            (val_df_aligned, "val_df_aligned"),
            (test_df_aligned,"test_df_aligned"),
        ]

        for df, df_name in dfs:
            if df is None or df.empty:
                continue
            
            for col in flow_feature_names:
                if col not in df.columns:
                     
                    print(f"    [Info] Missing flow feature '{col}' in {df_name}, skipped.")
                    continue
                
                 
                col_data_numeric = pd.to_numeric(df[col], errors='coerce')
                col_data_np = col_data_numeric.values
                col_data_cleaned = np.nan_to_num(
                    col_data_np,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                if 'count' in col:
                    df[col] = col_data_cleaned.astype(np.int32)
                else:
                    df[col] = col_data_cleaned.astype(np.float32)

        print(f" -> Feature Engineering 5.0 complete. {len(flow_feature_names)} flow features defined for this run.")


     
     
    # print("\n[3/4] Creating label mapping...")
    # labels = train_df['label'].unique() # ?
    # label_to_int = {label: i for i, label in enumerate(labels)}
    # num_classes = len(labels)

    # train_df['label_id'] = train_df['label'].map(label_to_int)
     
    # test_df_aligned['label_id'] = test_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) 
    

     
    print("\n[4/4] Creating GNN Datasets and DataLoaders...")
    
     
    train_dataset = GNNTrafficDataset(train_df, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN, enabled_layers=ABLATION_LAYERS, 
                                      backbone_mode=BACKBONE_MODE,
                                      use_ip_address=USE_IP_ADDRESS_THIS_RUN, use_mac_address=USE_MAC_ADDRESS_THIS_RUN, use_port=USE_PORT_THIS_RUN, obfuscation_config=None)
    val_dataset = GNNTrafficDataset(val_df_aligned, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN, enabled_layers=ABLATION_LAYERS, 
                                    backbone_mode=BACKBONE_MODE,
                                    use_ip_address=USE_IP_ADDRESS_THIS_RUN, use_mac_address=USE_MAC_ADDRESS_THIS_RUN, use_port=USE_PORT_THIS_RUN, obfuscation_config=None)
    test_dataset = GNNTrafficDataset(test_df_aligned, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN, enabled_layers=ABLATION_LAYERS, 
                                     backbone_mode=BACKBONE_MODE,
                                     use_ip_address=USE_IP_ADDRESS_THIS_RUN, use_mac_address=USE_MAC_ADDRESS_THIS_RUN, use_port=USE_PORT_THIS_RUN, obfuscation_config=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    amp_dtype = torch.float16 if AMP_DTYPE_STR.lower() == 'fp16' else torch.bfloat16
    use_amp_this_run = bool(USE_AMP and device.type == 'cuda')
    print(f"AMP enabled: {use_amp_this_run} (dtype={AMP_DTYPE_STR})")

     
    
    print("Calculating static class weights (alpha) for FocalLoss...")
    if TRAIN_TARGET == 'flow' and 'stream_id' in train_df.columns:
        # Align class weights with flow-level supervision target.
        flow_label_df = train_df[['stream_id', 'label_id']].drop_duplicates(subset=['stream_id'])
        class_counts_series = flow_label_df['label_id'].value_counts().sort_index()
        alpha_count_level = "flow"
    else:
        # Packet-level fallback.
        class_counts_series = train_df['label_id'].value_counts().sort_index()
        alpha_count_level = "packet"

    class_counts_series = class_counts_series.reindex(range(num_classes), fill_value=0)
    class_counts_np = class_counts_series.to_numpy(dtype=np.int64)
    class_counts_np[class_counts_np <= 0] = 1

    class_weights = torch.tensor(class_counts_np, dtype=torch.float)
    total_samples = class_weights.sum()
    raw_alpha = total_samples / (num_classes * class_weights)

    if FOCAL_ALPHA_SMOOTHING == 'sqrt':
        class_weights = torch.sqrt(raw_alpha)
    elif FOCAL_ALPHA_SMOOTHING == 'log':
        class_weights = torch.log1p(raw_alpha)
    elif FOCAL_ALPHA_SMOOTHING == 'none':
        class_weights = raw_alpha
    else:
        raise ValueError(f"Unsupported FOCAL_ALPHA_SMOOTHING: {FOCAL_ALPHA_SMOOTHING}")

    if FOCAL_ALPHA_CLAMP_MIN is not None or FOCAL_ALPHA_CLAMP_MAX is not None:
        cmin = float(FOCAL_ALPHA_CLAMP_MIN) if FOCAL_ALPHA_CLAMP_MIN is not None else None
        cmax = float(FOCAL_ALPHA_CLAMP_MAX) if FOCAL_ALPHA_CLAMP_MAX is not None else None
        if cmin is not None and cmax is not None:
            class_weights = torch.clamp(class_weights, min=cmin, max=cmax)
        elif cmin is not None:
            class_weights = torch.clamp(class_weights, min=cmin)
        elif cmax is not None:
            class_weights = torch.clamp(class_weights, max=cmax)
    
     
    alpha_weights = class_weights.to(device)
    
    print(f" -> Focal alpha count level: {alpha_count_level}")
    print(f" -> Focal alpha class counts: {class_counts_np.tolist()}")
    print(f" -> Focal raw alpha (inv-freq): {raw_alpha.numpy().round(3).tolist()}")
    print(f" -> FocalLoss Alpha (weights): {alpha_weights.cpu().numpy().round(2)}")

     
    loss_fn = FocalLoss(alpha=alpha_weights, gamma=FOCAL_GAMMA).to(device)

     
    # dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)
    
    # =====================================================================
    if not DIAGNOSE: 
        del train_df, val_df_aligned, test_df_aligned
        gc.collect()
    
    expert_graph_info = train_dataset.expert_graphs

     
    # node_fields_for_model = train_dataset.node_fields
    # print(f" - Model will be built for {len(node_fields_for_model)} nodes.")

    g = torch.Generator()
    g.manual_seed(SEED)

     
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, generator=g, 
                              drop_last=True, 
                              persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
                              prefetch_factor=DATALOADER_PREFETCH, 
                              collate_fn=train_dataset.collate_from_index,
                              )
                            #   collate_fn=custom_collate_flat_dict)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, 
                            persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
                            prefetch_factor=DATALOADER_PREFETCH, 
                            collate_fn=val_dataset.collate_from_index,
                            )
                            # collate_fn=custom_collate_flat_dict)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, 
                            persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
                            prefetch_factor=DATALOADER_PREFETCH, 
                            collate_fn=test_dataset.collate_from_index,
                            )
                            #  collate_fn=custom_collate_flat_dict)
    
     
    
    
    field_embedder = FieldEmbedding(config_path, vocab_path)
    field_embedder.to(device)

    pta_model = HierarchicalMoE(
        config_path=config_path,
        vocab_path=vocab_path,
        num_classes=num_classes,
        expert_graph_info=expert_graph_info,   
        use_flow_features=USE_FLOW_FEATURES_THIS_RUN,
        num_flow_features=len(train_dataset.flow_feature_names) if USE_FLOW_FEATURES_THIS_RUN else 0,
        hidden_dim=GNN_HIDDEN_DIM, 
        dropout_rate=DROPOUT_RATE,
        expert_gate_noise_std=EXPERT_GATE_NOISE_STD,
    )#.to(device)

     
     
    if torch.cuda.is_available():
        print("Setting torch.set_float32_matmul_precision('high')")
        torch.set_float32_matmul_precision('high')  

    pta_model.to(device)
    flow_packet_weighter = None
    if TRAIN_TARGET == 'flow' and FLOW_AGG_METHOD == 'learned_attn_logits':
        flow_packet_weighter = FlowLogitAttentionPool(
            num_classes=num_classes,
            hidden_dim=FLOW_ATTN_HIDDEN_DIM,
            dropout=FLOW_ATTN_DROPOUT,
        ).to(device)
        print("Flow learnable aggregator enabled (learned_attn_logits).")
    elif TRAIN_TARGET == 'flow' and FLOW_AGG_METHOD == 'repr_logits_attn':
        flow_packet_weighter = FlowReprLogitAggregator(
            repr_dim=GNN_HIDDEN_DIM,
            num_classes=num_classes,
            hidden_dim=FLOW_ATTN_HIDDEN_DIM,
            dropout=FLOW_ATTN_DROPOUT,
            use_prob_input=FLOW_REPR_LOGIT_USE_PROBS,
            use_logit_branch=FLOW_REPR_USE_LOGIT_BRANCH,
            attn_temperature=FLOW_REPR_ATTN_TAU,
            use_residual_mean=FLOW_REPR_USE_RESIDUAL_MEAN,
            use_logit_residual_baseline=FLOW_REPR_USE_LOGIT_RESIDUAL_BASELINE,
            logit_residual_init=FLOW_REPR_LOGIT_RESIDUAL_INIT,
            first_n_packets=FLOW_REPR_FIRST_N_PACKETS,
            pool_mode=FLOW_REPR_POOL_MODE,
        ).to(device)
        print("Flow learnable aggregator enabled (repr_logits_attn).")

    optim_params = list(pta_model.parameters())
    if flow_packet_weighter is not None:
        optim_params += list(flow_packet_weighter.parameters())
    optimizer = optim.AdamW(optim_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add weight_decay
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_this_run)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',   
        factor=0.2,   
        patience=4,   
        verbose=True,   
        min_lr=1e-6   
    )

     
    # dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)

     
    if not DIAGNOSE: 
        training_results = []
        best_f1 = 0.0
        best_val_f1_macro = 0.0 
        epochs_since_best = 0 
        epochs_needs_early_stop = 0
        best_epoch = -1
        best_model_state = None
        best_flow_weighter_state = None
        for epoch in range(NUM_EPOCHS): 
            if stop_training:
                print("Learning rate too low. Stopping training early.")
                break

            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            agg_only_mode = (
                TRAIN_TARGET == 'flow'
                and FLOW_AGG_METHOD == 'repr_logits_attn'
                and epoch < max(0, int(FLOW_REPR_TRAIN_AGG_ONLY_EPOCHS))
            )
            if agg_only_mode:
                # Freeze backbone and only train flow aggregator for a few warmup epochs.
                set_module_requires_grad(pta_model, False)
                if flow_packet_weighter is not None:
                    set_module_requires_grad(flow_packet_weighter, True)
                print("   -> Aggregator-only warmup active: backbone frozen.")
            else:
                set_module_requires_grad(pta_model, True)
                if flow_packet_weighter is not None:
                    set_module_requires_grad(flow_packet_weighter, True)

            train_metrics, _ = train_one_epoch(pta_model, train_loader, # loss_fn, 
                                               optimizer, # scheduler, 
                                               device, num_classes, loss_fn,
                                               alpha=MASK_ENTROPY_REG_ALPHA,
                                               train_target=TRAIN_TARGET,
                                               flow_agg_method=FLOW_AGG_METHOD,
                                               flow_topk=FLOW_TOPK,
                                               flow_soft_temp=FLOW_SOFT_TEMP,
                                               flow_packet_weighter=flow_packet_weighter,
                                               flow_loss_use_packet_aux=FLOW_LOSS_USE_PACKET_AUX,
                                               flow_packet_aux_weight=FLOW_PACKET_AUX_WEIGHT,
                                               current_epoch=epoch,
                                               repr_detach_warmup_epochs=FLOW_REPR_DETACH_WARMUP_EPOCHS,
                                               repr_permanent_detach=FLOW_REPR_PERMANENT_DETACH,
                                               flow_repr_gamma_clamp_max=FLOW_REPR_GAMMA_CLAMP_MAX,
                                               collect_dual_level_metrics=ENABLE_DUAL_LEVEL_METRICS,
                                               use_amp=use_amp_this_run,
                                               amp_dtype=amp_dtype,
                                               scaler=scaler)
                                            #    dynamic_weights=dynamic_weights)
            # val_metrics, _, val_per_class_f1 = evaluate(pta_model, val_loader, # loss_fn, 
            #                           device, num_classes)
            val_metrics, _ = evaluate(pta_model, val_loader, # loss_fn, 
                                      device, num_classes, loss_fn,
                                      eval_target=TRAIN_TARGET,
                                      flow_agg_method=FLOW_AGG_METHOD,
                                      flow_topk=FLOW_TOPK,
                                      flow_soft_temp=FLOW_SOFT_TEMP,
                                      flow_packet_weighter=flow_packet_weighter,
                                      collect_dual_level_metrics=ENABLE_DUAL_LEVEL_METRICS,
                                      use_amp=use_amp_this_run,
                                      amp_dtype=amp_dtype)
            
            # beta = 2.0 
            # new_weights = (1.0 - val_per_class_f1)**beta
             
            # new_weights = new_weights / new_weights.mean() 
            # dynamic_weights = new_weights.to(device)

             
             
             

             
             
             
            # momentum = 0.9 
             

            # updated_weights = (momentum * dynamic_weights_cpu) + ((1 - momentum) * new_weights)

             
            # dynamic_weights = updated_weights.to(device)
             

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1 (macro): {train_metrics['f1_macro']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            if ENABLE_DUAL_LEVEL_METRICS:
                print(
                    f"  Train Packet F1: {train_metrics['packet_f1_macro']:.4f} | "
                    f"Train Flow F1: {train_metrics['flow_f1_macro']:.4f} | "
                    f"Target: {TRAIN_TARGET}"
                )
                print(
                    f"  Val   Packet F1: {val_metrics['packet_f1_macro']:.4f} | "
                    f"Val   Flow F1: {val_metrics['flow_f1_macro']:.4f}"
                )

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Summary (Current LR: {current_lr:.1e}):")
            # print(f"Epoch {epoch+1} Summary (LR: {scheduler.get_last_lr()[0]:.1e}):")

            training_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'], 
                'train_recall_macro': train_metrics['recall_macro'], 
                'train_precision_macro': train_metrics['precision_macro'], 
                'train_f1_macro': train_metrics['f1_macro'], 
                'train_recall_weighted': train_metrics['recall_weighted'], 
                'train_precision_weighted': train_metrics['precision_weighted'], 
                'train_f1_weighted': train_metrics['f1_weighted'], 
                'train_packet_acc': train_metrics['packet_accuracy'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_packet_f1_macro': train_metrics['packet_f1_macro'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_packet_f1_weighted': train_metrics['packet_f1_weighted'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_packet_samples': train_metrics['packet_samples'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_flow_acc': train_metrics['flow_accuracy'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_flow_f1_macro': train_metrics['flow_f1_macro'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_flow_f1_weighted': train_metrics['flow_f1_weighted'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'train_flow_samples': train_metrics['flow_samples'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'], 
                'val_recall_macro': val_metrics['recall_macro'], 
                'val_precision_macro': val_metrics['precision_macro'], 
                'val_f1_macro': val_metrics['f1_macro'], 
                'val_recall_weighted': val_metrics['recall_weighted'], 
                'val_precision_weighted': val_metrics['precision_weighted'], 
                'val_f1_weighted': val_metrics['f1_weighted'], 
                'val_packet_acc': val_metrics['packet_accuracy'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_packet_f1_macro': val_metrics['packet_f1_macro'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_packet_f1_weighted': val_metrics['packet_f1_weighted'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_packet_samples': val_metrics['packet_samples'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_flow_acc': val_metrics['flow_accuracy'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_flow_f1_macro': val_metrics['flow_f1_macro'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_flow_f1_weighted': val_metrics['flow_f1_weighted'] if ENABLE_DUAL_LEVEL_METRICS else None,
                'val_flow_samples': val_metrics['flow_samples'] if ENABLE_DUAL_LEVEL_METRICS else None,
            })
            if (epoch + 1) % LOG_FLUSH_EVERY_EPOCH == 0:
                try:
                    pd.DataFrame(training_results).to_csv(
                        os.path.join(res_path, dataset_name + '_' + train_set_name + '_training_log.csv'),
                        index=False
                    )
                except Exception as e:
                    print(f"[Warning] Failed to flush training log at epoch {epoch+1}: {e}")

            current_val_f1_macro = val_metrics['f1_macro']
            scheduler.step(current_val_f1_macro)

            if current_val_f1_macro > best_val_f1_macro:
                 
                print(f" -> Validation Macro F1 improved from {best_val_f1_macro:.4f} to {current_val_f1_macro:.4f}. Saving state...")
                best_val_f1_macro = current_val_f1_macro
                best_epoch = epoch + 1
                 
                best_model_state = copy.deepcopy(pta_model.state_dict())
                if flow_packet_weighter is not None:
                    best_flow_weighter_state = copy.deepcopy(flow_packet_weighter.state_dict())
                torch.save(pta_model.state_dict(), os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_model.pth'))
                if flow_packet_weighter is not None:
                    torch.save(
                        flow_packet_weighter.state_dict(),
                        os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_flow_aggregator.pth')
                    )
                epochs_since_best = 0
                epochs_needs_early_stop = 0
            else:
                 
                epochs_since_best += 1
                epochs_needs_early_stop += 1
                print(f" -> Validation Macro F1 did not improve for {epochs_since_best} epoch(s). Best was {best_val_f1_macro:.4f} at epoch {best_epoch}.")

                if epochs_since_best >= ROLLBACK_PATIENCE:
                    print(f"\n!!! Performance has not improved for {ROLLBACK_PATIENCE} epochs. Rolling back to best model from epoch {best_epoch}. !!!")

                    if best_model_state:
                         
                        pta_model.load_state_dict(best_model_state)
                        if flow_packet_weighter is not None and best_flow_weighter_state is not None:
                            flow_packet_weighter.load_state_dict(best_flow_weighter_state)

                         
                        # print("   -> Aggressively reducing current learning rate by half...")
                        # new_lr = optimizer.param_groups[0]['lr'] * 0.5
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = new_lr

                         
                        epochs_since_best = 0

                         
                        # if new_lr < MIN_LR_FOR_TRAINING:
                        #     print(f"   -> Learning rate ({new_lr:.1e}) has fallen below minimum. Triggering final early stop.")
                         
                    else:
                        print("   -> Warning: No best model state found. Stopping training.")
                        break   
                 
                current_lr = optimizer.param_groups[0]['lr']
                if epochs_needs_early_stop >= EARLY_STOP_PATIENCE: 
                    print(f"Model's performance has not involved in {epochs_needs_early_stop} epoches. Triggering final early stop.")
                    stop_training = True   

        print("\nTraining complete!")

         
        print("\n" + "="*50)
        print("###   Learned Feature Importance Report   ###")

        importance_reports_dict = pta_model.get_feature_importance()

         
        # print(importance_report.to_string())

         
        # importance_report.to_csv(os.path.join(res_path,train_set_name + '_feature_importance_report.csv'), index=False)
        # print("\nFeature importance report saved to 'feature_importance_report.csv'")
        # print("="*50)

        all_reports_list = []
        for expert_name, expert_df in importance_reports_dict.items():
            print(f"\n--- Importance for Expert: '{expert_name}' ---")
             
            print(expert_df.to_string())
        
             
            expert_df_with_name = expert_df.copy()
            expert_df_with_name['expert_name'] = expert_name
            all_reports_list.append(expert_df_with_name)           
         
        combined_report_df = pd.concat(all_reports_list).reset_index(drop=True)

         
        report_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_feature_importance_report.csv')
        combined_report_df.to_csv(report_output_path, index=False)

        print(f"\nCombined feature importance report saved to: {report_output_path}")
        print("="*50)

         
        print("\n" + "="*50)
        print("###   Learned Expert Layer Importance Report (Macro)   ###")
        
         
        try:
            expert_layer_report = pta_model.get_expert_importance()
            
             
            print(expert_layer_report.to_string())
            
             
            expert_report_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_lastbatch_expert_layer_importance.csv')
            expert_layer_report.to_csv(expert_report_path, index=False)
            print(f"\nExpert layer importance saved to: {expert_report_path}")
        except Exception as e:
            print(f"Warning: Could not generate expert importance report. Error: {e}")
        
        print("="*50)
        # ============================================================================

         
        pta_model.load_state_dict(torch.load(os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_model.pth')))
        pta_model.to(device)
        if flow_packet_weighter is not None:
            agg_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_flow_aggregator.pth')
            if os.path.exists(agg_path):
                flow_packet_weighter.load_state_dict(torch.load(agg_path, map_location=device))
                flow_packet_weighter.to(device)
            else:
                print(f"Warning: flow aggregator checkpoint not found: {agg_path}")
        test_metrics, test_confusion_matrix = evaluate(
            pta_model, test_loader, device, num_classes, loss_fn,
            eval_target=TRAIN_TARGET,
            flow_agg_method=FLOW_AGG_METHOD,
            flow_topk=FLOW_TOPK,
            flow_soft_temp=FLOW_SOFT_TEMP,
            flow_packet_weighter=flow_packet_weighter,
            use_amp=use_amp_this_run,
            amp_dtype=amp_dtype
        )
        print(f"\nFinal Test Performance:")
        print(f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        flow_metrics = None
        flow_confusion_matrix = None
        if ENABLE_FLOW_AGG_EVAL and TRAIN_TARGET == 'packet':
            print("\n[Flow Aggregation Eval] Building flow-level predictions from packet logits...")
            packet_logits, packet_labels = collect_packet_logits_labels(pta_model, test_loader, device)
            flow_metrics, flow_confusion_matrix = evaluate_flow_aggregation_from_packet_logits(
                packet_logits=packet_logits,
                packet_labels=packet_labels,
                flow_ids=test_stream_ids_for_eval,
                num_classes=num_classes,
                use_prob_mean=FLOW_AGG_USE_PROB_MEAN,
            )
            print(
                f"  Flow Acc: {flow_metrics['accuracy']:.4f} | "
                f"Flow F1 (Macro): {flow_metrics['f1_macro']:.4f} | "
                f"Flows: {flow_metrics['num_flows']} | "
                f"Inconsistent flows: {flow_metrics['inconsistent_flows']}"
            )

         
        print("\nSaving confusion matrix...")

         
         
        int_to_label = {i: label for label, i in label_to_int.items()}
        class_names = [int_to_label[i] for i in range(num_classes)]

         
        confusion_matrix_df = pd.DataFrame(
            test_confusion_matrix.cpu().numpy(),   
            index=class_names,
            columns=class_names
        )

         
        cm_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_final_test_confusion_matrix.csv')
        confusion_matrix_df.to_csv(cm_output_path)

        print(f"Confusion matrix saved to: {cm_output_path}")
        if ENABLE_FLOW_AGG_EVAL and TRAIN_TARGET == 'packet' and flow_confusion_matrix is not None:
            flow_confusion_matrix_df = pd.DataFrame(
                flow_confusion_matrix.cpu().numpy(),
                index=class_names,
                columns=class_names
            )
            flow_cm_output_path = os.path.join(
                res_path, dataset_name + '_' + train_set_name + '_final_test_flow_confusion_matrix.csv'
            )
            flow_confusion_matrix_df.to_csv(flow_cm_output_path)
            print(f"Flow confusion matrix saved to: {flow_cm_output_path}")

        training_results.append({
            'epoch': 'final_test',
            'train_loss': None,
            'train_acc': None, 
            'train_recall_macro': None, 
            'train_precision_macro': None, 
            'train_f1_macro': None, 
            'train_recall_weighted': None, 
            'train_precision_weighted': None, 
            'train_f1_weighted': None, 
            'val_loss': test_metrics['loss'],
            'val_acc': test_metrics['accuracy'], 
            'val_recall_macro': test_metrics['recall_macro'], 
            'val_precision_macro': test_metrics['precision_macro'], 
            'val_f1_macro': test_metrics['f1_macro'], 
            'val_recall_weighted': test_metrics['recall_weighted'], 
            'val_precision_weighted': test_metrics['precision_weighted'], 
            'val_f1_weighted': test_metrics['f1_weighted'],
            'flow_acc': (
                flow_metrics['accuracy'] if flow_metrics is not None
                else (test_metrics['accuracy'] if TRAIN_TARGET == 'flow' else None)
            ),
            'flow_f1_macro': (
                flow_metrics['f1_macro'] if flow_metrics is not None
                else (test_metrics['f1_macro'] if TRAIN_TARGET == 'flow' else None)
            ),
            'flow_num_flows': (flow_metrics['num_flows'] if flow_metrics is not None else None),
            'flow_inconsistent_flows': (flow_metrics['inconsistent_flows'] if flow_metrics is not None else None),
        })

        results_df = pd.DataFrame(training_results)
        results_df.to_csv(os.path.join(res_path,dataset_name + '_' + train_set_name + '_training_log.csv'), index=False)
        print(f"\nTraining log saved to {train_set_name}_training_log.csv")

        print("\nComputing dataset-level expert importance...")

        expected_expert_weights = compute_dataset_expert_importance(
            pta_model,
            test_loader,   
            device
        )

        print("Expected expert weights:", expected_expert_weights.numpy())

        expert_names = pta_model.gnn_expert_names.copy()
        if pta_model.use_flow_features:
            expert_names.append("Flow_Features_Block")

        df_expert = pd.DataFrame({
            "expert_name": expert_names,
            "importance_score": expected_expert_weights.numpy()
        })

        expert_csv_path = os.path.join(
            res_path,
            dataset_name + '_' + train_set_name + '_expert_layer_importance.csv'
        )

        df_expert.to_csv(expert_csv_path, index=False)

        print(f"Expert importance saved to: {expert_csv_path}")

    elif DIAGNOSE:

        print("\n==============================")
        print("  Diagnose Eval-Only Mode ")
        print("==============================\n")

        best_model_path = os.path.join(
            res_path,
            dataset_name + '_' + train_set_name + '_best_model.pth'
        )

        if not os.path.exists(best_model_path):
            print(f"Error: saved model file not found: {best_model_path}")
            exit()

        print("Loading saved model...")
        pta_model.load_state_dict(torch.load(best_model_path, map_location=device))
        pta_model.to(device)
        if flow_packet_weighter is not None:
            agg_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_flow_aggregator.pth')
            if os.path.exists(agg_path):
                flow_packet_weighter.load_state_dict(torch.load(agg_path, map_location=device))
                flow_packet_weighter.to(device)
            else:
                print(f"Warning: flow aggregator checkpoint not found: {agg_path}")
        pta_model.eval()
        print("Model loaded successfully.\n")

        test_metrics, test_confusion_matrix = evaluate(
            pta_model,
            test_loader,
            device,
            num_classes,
            loss_fn,
            eval_target=TRAIN_TARGET,
            flow_agg_method=FLOW_AGG_METHOD,
            flow_topk=FLOW_TOPK,
            flow_soft_temp=FLOW_SOFT_TEMP,
            flow_packet_weighter=flow_packet_weighter,
            use_amp=use_amp_this_run,
            amp_dtype=amp_dtype
        )
        print("Final Test (Eval-Only):")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Acc: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

        int_to_label = {i: label for label, i in label_to_int.items()}
        class_names = [int_to_label[i] for i in range(num_classes)]
        confusion_matrix_df = pd.DataFrame(
            test_confusion_matrix.cpu().numpy(),
            index=class_names,
            columns=class_names
        )
        cm_output_path = os.path.join(
            res_path,
            dataset_name + '_' + train_set_name + '_diagnose_test_confusion_matrix.csv'
        )
        confusion_matrix_df.to_csv(cm_output_path)
        print(f"Confusion matrix saved to: {cm_output_path}")

        flow_metrics = None
        if ENABLE_FLOW_AGG_EVAL and TRAIN_TARGET == 'packet':
            packet_logits, packet_labels = collect_packet_logits_labels(pta_model, test_loader, device)
            flow_metrics, flow_confusion_matrix = evaluate_flow_aggregation_from_packet_logits(
                packet_logits=packet_logits,
                packet_labels=packet_labels,
                flow_ids=test_stream_ids_for_eval,
                num_classes=num_classes,
                use_prob_mean=FLOW_AGG_USE_PROB_MEAN,
            )
            print("Flow Aggregation (Eval-Only):")
            print(f"  Flow Acc: {flow_metrics['accuracy']:.4f}")
            print(f"  Flow F1 (Macro): {flow_metrics['f1_macro']:.4f}")
            print(f"  Num flows: {flow_metrics['num_flows']}")
            print(f"  Inconsistent flows: {flow_metrics['inconsistent_flows']}")

            flow_confusion_matrix_df = pd.DataFrame(
                flow_confusion_matrix.cpu().numpy(),
                index=class_names,
                columns=class_names
            )
            flow_cm_output_path = os.path.join(
                res_path,
                dataset_name + '_' + train_set_name + '_diagnose_test_flow_confusion_matrix.csv'
            )
            flow_confusion_matrix_df.to_csv(flow_cm_output_path)
            print(f"Flow confusion matrix saved to: {flow_cm_output_path}")

        diagnose_summary = {
            'test_loss': test_metrics['loss'],
            'test_acc': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_precision_macro': test_metrics['precision_macro'],
            'test_recall_macro': test_metrics['recall_macro'],
            'flow_acc': (flow_metrics['accuracy'] if flow_metrics is not None else None),
            'flow_f1_macro': (flow_metrics['f1_macro'] if flow_metrics is not None else None),
            'flow_num_flows': (flow_metrics['num_flows'] if flow_metrics is not None else None),
            'flow_inconsistent_flows': (flow_metrics['inconsistent_flows'] if flow_metrics is not None else None),
        }
        diagnose_output_path = os.path.join(
            res_path,
            dataset_name + '_' + train_set_name + '_diagnose_eval_results.csv'
        )
        pd.DataFrame([diagnose_summary]).to_csv(diagnose_output_path, index=False)
        print(f"Diagnose eval summary saved to: {diagnose_output_path}")
