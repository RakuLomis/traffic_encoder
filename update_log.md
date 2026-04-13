# Update Log: Active Runtime Map for `train_test_pipeline_flow_enhance.py`

## Scope
This document records the **actual runtime dependency chain** of the current active training pipeline (`train_test_pipeline_flow_enhance.py`) for project shrinking/archiving decisions.

---

## 1) Entry Script and Core Runtime Path

### Entry
- `train_test_pipeline_flow_enhance.py`

### High-level runtime flow
1. Load/split/sample CSV data in script.
2. Optional flow-feature engineering in script.
3. Build dataset via `GNNTrafficDataset`.
4. Build model via `HierarchicalMoE`.
5. Train/eval with flow-aware loss + aggregation.
6. Export metrics/log/confusion matrix/importance reports.

---

## 2) File-by-file Runtime Dependencies (Used by Current Version)

## 2.1 `train_test_pipeline_flow_enhance.py`
### Locally defined and actively used
- Utility:
  - `robust_hex_to_int`
  - `robust_timestamp_to_tsval`
  - `set_seed`
  - `seed_worker`
  - `set_module_requires_grad`
- Batching / aggregation:
  - `FlowCentricBatchSampler`
  - `FlowLogitAttentionPool`
  - `FlowReprLogitAggregator`
  - `aggregate_logits_by_flow_tensor`
- Train/eval:
  - `train_one_epoch`
  - `evaluate`
  - `collect_packet_logits_labels`
  - `evaluate_flow_aggregation_from_packet_logits`
  - `compute_dataset_expert_importance`
- Main path (`if __name__ == '__main__':`)
  - Data loading/sampling + optional flow features
  - Dataset/DataLoader creation
  - Model/optimizer/loss/scheduler
  - Epoch loop + checkpointing + final reports

### External modules called from this script
- `utils/data_loader_ptga_le.py`
  - `GNNTrafficDataset`
- `models/ProtocolTreeGAttention_le.py`
  - `HierarchicalMoE`
- `models/FieldEmbedding.py`
  - `FieldEmbedding` (instantiated for compatibility/check; model also uses its own)
- `utils/loss_functions.py`
  - `FocalLoss`
- `utils/metrics.py`
  - `calculate_metrics`
- `utils/dataframe_tools.py`
  - `stratified_flow_sample_from_csv_stream`
  - `stratified_hybrid_sample_from_csv_stream`

---

## 2.2 `utils/data_loader_ptga_le.py`
### Runtime class used
- `GNNTrafficDataset`

### Methods actively used by current pipeline
- `__init__`
  - Loads YAML config/vocab
  - Builds `flow_ids`
  - Builds `expert_definitions`
  - Applies `YAML ∩ CSV` physical-field whitelist
  - Supports virtual sink settings (`enable_virtual_sink`, `virtual_sink_name`)
  - Builds `expert_graphs`
  - Calls preprocessing/vectorization
- `_create_edge_index_from_tree`
- `_augment_edge_index_with_virtual_sink`
- `_preprocess_all`
- `_preload_to_tensors`
- `_get_batched_edge_index`
- `collate_from_index` (**critical in DataLoader**)
- `__len__`, `__getitem__`

### Upstream functions used inside this file
- From `utils/dataframe_tools.py`:
  - `protocol_tree`
  - `add_root_layer`
- From `utils/data_loader.py`:
  - `_preprocess_address`

---

## 2.3 `models/ProtocolTreeGAttention_le.py`
### Runtime classes used
- `PTGAMiniExpert`
- `HierarchicalMoE`

### `PTGAMiniExpert` used methods
- `__init__`
  - Builds per-node abstract token table (`abstract_node_embeddings`)
  - Builds per-dim aligners
  - Builds GAT layers
  - Initializes feature mask logits
- `_align_fused`
- `forward`
  - Field embedding -> alignment -> node features -> GAT -> graph embedding

### `HierarchicalMoE` used methods
- `__init__`
  - Creates shared `FieldEmbedding`
  - Instantiates per-expert `PTGAMiniExpert`
  - Builds gating network and classifier head
- `forward`
  - Collect expert embeddings
  - Sigmoid gating (`expert_weights = sigmoid(gating_logits)`)
  - Weighted sum fusion
  - Packet logits output
  - Optional return packet repr
- Reporting APIs used by entry script
  - `get_feature_importance`
  - `get_expert_importance`

---

## 2.4 `models/FieldEmbedding.py`
### Runtime class used
- `FieldEmbedding`

### Methods used
- `__init__`
  - Loads field embedding config and vocab map
  - Builds embedding layers for field types
- `forward`
  - Produces per-field embedded vectors used in PTGA experts

---

## 2.5 `utils/loss_functions.py`
### Runtime object used
- `FocalLoss`

### Called behavior in current pipeline
- Packet-level focal loss
- Flow-level focal loss
- Combined objective when `TRAIN_TARGET='flow'` and packet auxiliary loss enabled

---

## 2.6 `utils/metrics.py`
### Runtime function used
- `calculate_metrics`

### Called in
- `train_one_epoch`
- `evaluate`
- `evaluate_flow_aggregation_from_packet_logits`

---

## 2.7 `utils/dataframe_tools.py`
### Functions actively used by this version
- Imported/called directly in main:
  - `stratified_flow_sample_from_csv_stream`
  - `stratified_hybrid_sample_from_csv_stream`
- Called indirectly from dataset builder:
  - `protocol_tree`
  - `add_root_layer`

---

## 3) Current Strict/Flow-centric Batch Settings (from code)
- `TRAIN_TARGET = 'flow'`
- `ENABLE_FLOW_CENTRIC_BATCHING = True`
- `FLOW_CENTRIC_BATCH_FLOWS = 64`
- `FLOW_CENTRIC_PACKETS_PER_FLOW = FLOW_REPR_FIRST_N_PACKETS` (currently 64)
- `FLOW_CENTRIC_DROP_LAST = False`
- `FLOW_CENTRIC_LOW_RAM_SAFE_MODE = True`
  - On Windows: `num_workers=0`, `pin_memory=False`, `persistent_workers=False`, `prefetch=1`

---

## 4) Flow Aggregation Runtime Branch (current)
- `FLOW_AGG_METHOD = 'repr_logits_attn'`
- `FLOW_REPR_USE_LOGIT_BRANCH = False` (repr-only branch)
- `FLOW_REPR_POOL_MODE = 'max'`
- `FLOW_REPR_FIRST_N_PACKETS = 64`
- `FLOW_LOSS_USE_PACKET_AUX = True`
- `FLOW_PACKET_AUX_WEIGHT = 0.5`

---

## 5) Importance Outputs: Current Semantics
- Feature importance (`*_feature_importance_report.csv`):
  - From model parameter `sigmoid(feature_mask_logits)` per node/field.
  - Global learned gates, **not per-sample averaging**.
- Expert importance (`*_expert_layer_importance.csv`):
  - Dataset-level expectation over packet samples:
  - `E_x[omega_k(x)]` computed by `compute_dataset_expert_importance`.
- Last-batch expert importance (`*_lastbatch_expert_layer_importance.csv`):
  - Mean expert weights over the **latest forward batch only**.

---

## 6) Archive Candidates (Not Active in Current Runtime Path)
The following imports are present in `train_test_pipeline_flow_enhance.py` but are not part of the current active execution path and can be considered for archival/refactor:

- `utils.data_loader.TrafficDataset`
- `models.ProtocolTreeAttention.ProtocolTreeAttention`
- `models.MoEPTA.MoEPTA`
- `models.ProtocolTreeGAttention.ProtocolTreeGAttention`
- `utils.model_utils.diagnose_gate_weights_for_class`
- `utils.dataframe_tools.protocol_tree` / `get_file_path` / `output_csv_in_fold` / `padding_or_truncating`
- `sklearn.model_selection.train_test_split`
- `torch.profiler` imports (`profile`, `record_function`, `ProfilerActivity`)
- `torch.optim.RAdam`
- `utils.dataframe_tools.stratified_sample_dataframe`
- `utils.dataframe_tools.stratified_hybrid_sample_dataframe_optimized`
- `utils.dataframe_tools.stratified_aggressive_balancing`
- `sys`

> Note: Keep them until you finish cleanup PR; then remove unused imports and dead code in one controlled commit.

---

## 7) Suggested Next Cleanup Steps (Safe Order)
1. Remove unused imports in `train_test_pipeline_flow_enhance.py`.
2. Isolate active config block into a small config module.
3. Move deprecated/experimental branches to `archive/`.
4. Keep this log synchronized whenever runtime dependencies change.
