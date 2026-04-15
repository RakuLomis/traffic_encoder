# SSL Checklist (PTGAMoE)

This checklist is for building and validating the self-supervised PTGAMoE pipeline without violating no-leakage constraints.

## 1. Data Isolation (Must Pass)
- Persist split flow IDs:
  - `train_flow_ids.txt`
  - `val_flow_ids.txt`
  - `test_flow_ids.txt`
- Assert disjointness before any SSL run:
  - `train ∩ val = ∅`
  - `train ∩ test = ∅`
  - `val ∩ test = ∅`
- SSL pretraining reads only train-flow samples.

## 2. SSL Dataset Build
- Build `ssl_train.csv` from train flows only.
- Optional: build `ssl_val.csv` from a subset of train flows.
- Save snapshot stats:
  - packet count
  - flow count
  - class distribution (if labels exist)

## 3. Stage-A Augmentations
- Node feature masking:
  - mask physical fields only
  - ratio default `0.15`
  - do not mask `root`, abstract nodes, or sink node
- Sink-edge dropout:
  - apply only on sink-related edges
  - default `p_drop=0.10` (range `0.05~0.15`)

## 4. Model Outputs Needed
- PTGAMoE forward must support:
  - fused packet representation (`packet_repr`)
  - per-expert packet embeddings (`expert_embedding_dict`)

## 5. SSL Losses
- Instance contrastive loss:
  - `L_inst = NT-Xent(v1_repr, v2_repr)`
- Optional layer consistency loss:
  - `L_layer = NT-Xent(z_tcp, z_tls)` (or similar pair)
- Stage-A total:
  - `L_ssl = λ1 * L_inst + λ3 * L_layer`
  - start with `λ1=1.0`, `λ3=0.2`

## 6. Monitoring (Every Epoch)
- `L_total, L_inst, L_layer`
- gradient norms:
  - backbone
  - projection/loss heads
- throughput:
  - epoch time
  - GPU memory/utilization

## 7. Leakage Guard Artifacts
- Save `leakage_check.json` with:
  - split sizes
  - intersection counts
  - run timestamp/config

## 8. Outputs to Save
- `ssl_pretrain_best.pth`
- `ssl_training_log.csv`
- `ssl_config_snapshot.yaml`

## 9. Transfer to Supervised Fine-tuning
- Initialize encoder from SSL checkpoint.
- Keep no-leakage split unchanged.
- Compare against:
  - supervised from scratch
  - SSL Stage-A init

## 10. Exit Criteria
- Leakage checks all zero.
- SSL loss decreases stably.
- Fine-tuned flow-level F1 is >= baseline.
