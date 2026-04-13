from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import calculate_metrics
from utils.flow_batch_components import aggregate_logits_by_flow_tensor


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad




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
