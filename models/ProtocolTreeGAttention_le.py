import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd


# ============================================================
# 1️⃣  PTGAMiniExpert  （未改动）
# ============================================================

# class PTGAMiniExpert(nn.Module):

#     def __init__(self,
#                  field_embedder: FieldEmbedding,
#                  node_fields_list: List[str],
#                  edge_index: torch.Tensor,
#                  hidden_dim: int = 128,
#                  num_heads: int = 4,
#                  dropout_rate: float = 0.3):

#         super().__init__()

#         self.field_embedder = field_embedder
#         self.hidden_dim = hidden_dim
#         self.dropout_rate = dropout_rate
#         self.node_fields = node_fields_list

#         self.register_buffer('edge_index', edge_index)

#         self.aligners = nn.ModuleDict()
#         unique_dims = set()
#         for field_name in self.node_fields:
#             if field_name in self.field_embedder.embedding_slices:
#                 dim = self.field_embedder.embedding_slices[field_name][1] - \
#                       self.field_embedder.embedding_slices[field_name][0]
#                 if dim > 0:
#                     unique_dims.add(dim)

#         for dim in unique_dims:
#             self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

#         self.conv1 = GATConv(hidden_dim, hidden_dim,
#                              heads=num_heads, dropout=dropout_rate)
#         self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim,
#                              heads=1, concat=False, dropout=dropout_rate)

#         num_nodes = len(self.node_fields)
#         initial_logits = torch.full((num_nodes,), 2.2)
#         self.feature_mask_logits = nn.Parameter(initial_logits)
#         self.align_norm = nn.LayerNorm(hidden_dim)
#         self.dummy_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
#         # self.abstract_node_embeddings = nn.Embedding(num_nodes, hidden_dim)

#     def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor]):
#         aligned_vectors = {}
#         vectors_by_dim = defaultdict(list)
#         names_by_dim = defaultdict(list)

#         for name, vec in embedded_vectors.items():
#             dim = vec.shape[-1]
#             if f'aligner_from_{dim}' in self.aligners:
#                 vectors_by_dim[dim].append(vec)
#                 names_by_dim[dim].append(name)

#         for dim, vecs in vectors_by_dim.items():
#             stacked_vecs = torch.stack(vecs, dim=1)
#             aligner = self.aligners[f'aligner_from_{dim}']
#             aligned_stacked = aligner(stacked_vecs)
#             for i, name in enumerate(names_by_dim[dim]):
#                 aligned_vectors[name] = aligned_stacked[:, i, :]

#         return aligned_vectors

#     def forward(self, data):

#         batch_dict = {key: val for key, val in data
#                       if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}

#         embedded_vectors = self.field_embedder(batch_dict)
#         aligned_vectors = self._align_fused(embedded_vectors)

#         node_feature_list = []
#         for field_name in self.node_fields:
#             vec = aligned_vectors.get(field_name)
#             if vec is not None:
#                 node_feature_list.append(vec)
#             else:
#                 dummy_vec = self.dummy_token.expand(data.num_graphs, -1)
#                 node_feature_list.append(dummy_vec)

#         stacked_x = torch.stack(node_feature_list, dim=1)
#         stacked_x_normed = self.align_norm(stacked_x)

#         feature_gate = torch.sigmoid(self.feature_mask_logits)
#         mask_for_broadcast = feature_gate.view(1, -1, 1)
#         gated_stacked_x = stacked_x_normed * mask_for_broadcast

#         x = gated_stacked_x.view(-1, self.hidden_dim)

#         edge_index = data.edge_index.to(x.device)
#         batch_idx = data.batch.to(x.device)

#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = self.conv2(x, edge_index)

#         graph_embedding = global_mean_pool(x, batch_idx)

#         return graph_embedding, feature_gate

class PTGAMiniExpert(nn.Module):
    """
    Per-node abstract token version.
    Each node (including abstract nodes) owns an independent learnable token.
    """

    def __init__(self, 
                 field_embedder: FieldEmbedding, 
                 node_fields_list: List[str],
                 edge_index: torch.Tensor,
                 hidden_dim: int = 128, 
                 num_heads: int = 4, 
                 dropout_rate: float = 0.3):
        
        super().__init__()
        
        self.field_embedder = field_embedder
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.node_fields = node_fields_list
        self.num_nodes = len(self.node_fields)

        self.register_buffer('edge_index', edge_index)

        # ============================================================
        # 1️⃣  为所有节点创建独立 token（核心修改）
        # ============================================================
        self.abstract_node_embeddings = nn.Embedding(
            num_embeddings=self.num_nodes,
            embedding_dim=hidden_dim
        )

        nn.init.normal_(self.abstract_node_embeddings.weight, mean=0.0, std=0.01)

        # ============================================================
        # 2️⃣  对齐层（保持不变）
        # ============================================================
        self.aligners = nn.ModuleDict()
        unique_dims = set()

        for field_name in self.node_fields:
            if field_name in self.field_embedder.embedding_slices:
                dim = (
                    self.field_embedder.embedding_slices[field_name][1]
                    - self.field_embedder.embedding_slices[field_name][0]
                )
                if dim > 0:
                    unique_dims.add(dim)

        for dim in unique_dims:
            self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

        # ============================================================
        # 3️⃣  GAT layers
        # ============================================================
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)

        # ============================================================
        # 4️⃣  Feature mask
        # ============================================================
        initial_logits = torch.full((self.num_nodes,), 2.2)
        self.feature_mask_logits = nn.Parameter(initial_logits)

        self.align_norm = nn.LayerNorm(hidden_dim)

    # ================================================================
    # 对齐函数（保持不变）
    # ================================================================
    def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor]):
        aligned_vectors = {}
        vectors_by_dim = defaultdict(list)
        names_by_dim = defaultdict(list)
        
        for name, vec in embedded_vectors.items():
            dim = vec.shape[-1]
            if f'aligner_from_{dim}' in self.aligners:
                vectors_by_dim[dim].append(vec)
                names_by_dim[dim].append(name)
        
        for dim, vecs in vectors_by_dim.items():
            stacked_vecs = torch.stack(vecs, dim=1)
            aligner = self.aligners[f'aligner_from_{dim}']
            aligned_stacked = aligner(stacked_vecs)
            for i, name in enumerate(names_by_dim[dim]):
                aligned_vectors[name] = aligned_stacked[:, i, :]
        
        return aligned_vectors

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, data):

        batch_dict = {
            key: val
            for key, val in data
            if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']
        }

        embedded_vectors = self.field_embedder(batch_dict)
        aligned_vectors = self._align_fused(embedded_vectors)

        # Compatible with both PyG `Batch` and custom batched `Data`.
        if hasattr(data, 'num_graphs'):
            B = data.num_graphs
        elif hasattr(data, 'y') and data.y is not None:
            B = int(data.y.size(0))
        elif hasattr(data, 'batch') and data.batch is not None and data.batch.numel() > 0:
            B = int(data.batch.max().item()) + 1
        else:
            raise ValueError("Cannot infer batch size from input data.")
        device = next(self.parameters()).device

        node_feature_list = []

        # ============================================================
        # 核心逻辑：per-node token
        # ============================================================
        for node_idx, field_name in enumerate(self.node_fields):

            vec = aligned_vectors.get(field_name)

            if vec is not None:
                node_feature_list.append(vec)
            else:
                # 使用该节点专属 token
                token = self.abstract_node_embeddings(
                    torch.tensor(node_idx, device=device)
                )
                token = token.unsqueeze(0).expand(B, -1)
                node_feature_list.append(token)

        stacked_x = torch.stack(node_feature_list, dim=1)  # [B, N, D]
        stacked_x = self.align_norm(stacked_x)

        # ============================================================
        # Feature gating
        # ============================================================
        feature_gate = torch.sigmoid(self.feature_mask_logits)
        mask = feature_gate.view(1, -1, 1)
        stacked_x = stacked_x * mask

        x = stacked_x.view(-1, self.hidden_dim)

        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        graph_embedding = global_mean_pool(x, batch_idx)

        return graph_embedding, feature_gate


# ============================================================
# 2️⃣  HierarchicalMoE  （核心修改）
# ============================================================

class HierarchicalMoE(nn.Module):

    def __init__(self,
                 config_path: str,
                 vocab_path: str,
                 num_classes: int,
                 expert_graph_info: Dict[str, Dict],
                 use_flow_features: bool = False,
                 num_flow_features: int = 0,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout_rate: float = 0.3,
                 expert_gate_noise_std: float = 0.0):

        super().__init__()

        self.use_flow_features = use_flow_features
        self.hidden_dim = hidden_dim
        self.expert_gate_noise_std = float(expert_gate_noise_std)

        self.shared_field_embedder = FieldEmbedding(config_path, vocab_path)

        # -----------------------------
        # 初始化 GNN 专家
        # -----------------------------
        self.experts = nn.ModuleDict()
        for name, graph_info in expert_graph_info.items():
            self.experts[name] = PTGAMiniExpert(
                field_embedder=self.shared_field_embedder,
                node_fields_list=graph_info['all_nodes'],
                edge_index=graph_info['edge_index'],
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )

        self.gnn_expert_names = sorted(list(self.experts.keys()))
        self.num_gnn_experts = len(self.gnn_expert_names)
        self.num_total_experts = self.num_gnn_experts + \
                                 (1 if self.use_flow_features else 0)

        # -----------------------------
        # Flow expert
        # -----------------------------
        if self.use_flow_features:
            self.flow_stats_embedder = nn.Sequential(
                nn.LayerNorm(num_flow_features),
                nn.Linear(num_flow_features, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.flow_output_norm = nn.LayerNorm(hidden_dim)

        # -----------------------------
        # ===== MODIFIED: 动态 gating =====
        # -----------------------------
        total_concat_dim = hidden_dim * self.num_total_experts

        self.gating_network = nn.Sequential(
            nn.Linear(total_concat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_total_experts)
        )

        # 缓存最近一次 expert weights（用于 get_expert_importance）
        self._latest_expert_weights = None

        # -----------------------------
        # 分类头
        # -----------------------------
        self.aggregator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        batch_dict: Dict[str, Any],
        return_packet_repr: bool = False,
        return_expert_embeddings: bool = False,
    ):

        expert_embeddings = []
        expert_embedding_dict = {}
        all_gates = {}

        # 1️⃣ GNN experts
        for name in self.gnn_expert_names:
            if name not in batch_dict:
                continue

            embedding, gate = self.experts[name](batch_dict[name])
            expert_embeddings.append(embedding)
            expert_embedding_dict[name] = embedding
            all_gates[name] = gate

        # 2️⃣ Flow expert
        if self.use_flow_features:
            flow_input = batch_dict['flow_stats']
            if flow_input.dim() == 3 and flow_input.shape[1] == 1:
                flow_input = flow_input.squeeze(1)

            flow_embedding = self.flow_stats_embedder(flow_input)
            flow_embedding = self.flow_output_norm(flow_embedding)
            expert_embeddings.append(flow_embedding)
            expert_embedding_dict["Flow_Features_Block"] = flow_embedding

        if not expert_embeddings:
            raise ValueError("No expert produced embeddings.")

        # 3️⃣ 动态 gating
        z_concat = torch.cat(expert_embeddings, dim=1)
        gating_logits = self.gating_network(z_concat)
        expert_weights = torch.sigmoid(gating_logits)
        if self.training and self.expert_gate_noise_std > 0.0:
            noise = torch.randn_like(expert_weights) * self.expert_gate_noise_std
            expert_weights = torch.clamp(expert_weights + noise, 0.0, 1.0)

        # 缓存当前 batch 的 expert weights
        self._latest_expert_weights = expert_weights.detach()

        # 4️⃣ 加权融合
        stacked = torch.stack(expert_embeddings, dim=1)
        expert_weights = expert_weights.unsqueeze(-1)
        weighted_sum = torch.sum(stacked * expert_weights, dim=1)

        logits = self.aggregator(weighted_sum)

        if return_packet_repr and return_expert_embeddings:
            return logits, all_gates, weighted_sum, expert_embedding_dict
        if return_packet_repr:
            return logits, all_gates, weighted_sum
        if return_expert_embeddings:
            return logits, all_gates, expert_embedding_dict
        return logits, all_gates

    # =========================================================
    # Feature Importance（保持原接口）
    # =========================================================

    def get_feature_importance(self) -> Dict[str, 'pd.DataFrame']:

        import pandas as pd

        reports = {}

        for name, expert_model in self.experts.items():
            with torch.no_grad():
                weights = torch.sigmoid(
                    expert_model.feature_mask_logits
                ).cpu().numpy()

            df = pd.DataFrame({
                'feature_name': expert_model.node_fields,
                'importance_score': weights
            })

            reports[name] = df.sort_values(
                by='importance_score',
                ascending=False
            ).reset_index(drop=True)

        return reports

    # =========================================================
    # Expert Importance（保持原接口）
    # =========================================================

    def get_expert_importance(self):

        import pandas as pd

        if self._latest_expert_weights is None:
            raise RuntimeError(
                "No forward pass has been executed yet. "
                "Run at least one batch before calling get_expert_importance()."
            )

        # 计算 batch 平均权重
        avg_weights = self._latest_expert_weights.mean(dim=0).cpu().numpy()

        expert_names = list(self.gnn_expert_names)

        if self.use_flow_features:
            expert_names.append("Flow_Features_Block")

        if len(avg_weights) != len(expert_names):
            raise RuntimeError("Mismatch between expert weights and expert names.")

        df = pd.DataFrame({
            'expert_name': expert_names,
            'importance_score': avg_weights
        }).sort_values(
            by='importance_score',
            ascending=False
        ).reset_index(drop=True)

        return df

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_mean_pool
# from models.FieldEmbedding import FieldEmbedding
# from collections import defaultdict
# from typing import Dict, List, Tuple, Any
# import pandas as pd
# from torch_geometric.data import Data, Batch

# # ==============================================================================
# # 1. “微型专家” (PTGAMiniExpert)
# # ==============================================================================

# class PTGAMiniExpert(nn.Module):
#     """
#     【新】“微型”PTGA专家，专门用于处理一个协议层（如'ip', 'tcp'）的图。
#     它不包含最终的分类器，只负责输出一个固定维度的嵌入向量。
#     """
#     def __init__(self, 
#                  field_embedder: FieldEmbedding, 
#                  node_fields_list: List[str],
#                  edge_index: torch.Tensor,
#                  hidden_dim: int = 128, 
#                  num_heads: int = 4, 
#                  dropout_rate: float = 0.3):
        
#         super().__init__()
        
#         self.field_embedder = field_embedder # 接收【共享的】嵌入器
#         self.hidden_dim = hidden_dim
#         self.dropout_rate = dropout_rate
#         self.node_fields = node_fields_list # 这个专家自己的节点列表
        
#         # 将这个专家的图结构 (edge_index) 注册为模型的缓冲区
#         # 这样它就可以被 .to(device) 自动移动
#         self.register_buffer('edge_index', edge_index)

#         # --- 1. 创建对齐层 (Aligners) ---
#         # (与您之前的逻辑相同，但只为这个专家需要的字段创建)
#         self.aligners = nn.ModuleDict()
#         unique_dims = set()
#         for field_name in self.node_fields:
#             if field_name in self.field_embedder.embedding_slices:
#                 dim = self.field_embedder.embedding_slices[field_name][1] - self.field_embedder.embedding_slices[field_name][0]
#                 if dim > 0:
#                     unique_dims.add(dim)
        
#         for dim in unique_dims:
#             self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

#         # --- 2. 创建GNN层 ---
#         self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
#         self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
#         # --- 3. 创建“急救”组件 (特征掩码、LayerNorm、哑元) ---
#         num_nodes = len(self.node_fields)
#         initial_logits = torch.full((num_nodes,), 2.2) # 门控默认“打开”
#         # initial_logits = torch.zeros(num_nodes)   # sigmoid = 0.5
#         self.feature_mask_logits = nn.Parameter(initial_logits)
#         self.align_norm = nn.LayerNorm(hidden_dim)
#         self.dummy_token = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.01)

#         self.gating_layer = nn.ModuleDict({
#             field: nn.Linear(self.hidden_dim, 1) for field in self.node_fields
#         })

#     def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor]):
#         """辅助函数：执行高效的“融合对齐”操作。"""
#         # (这个函数与您之前的版本完全相同)
#         aligned_vectors = {}
#         vectors_by_dim = defaultdict(list)
#         names_by_dim = defaultdict(list)
        
#         for name, vec in embedded_vectors.items():
#             dim = vec.shape[-1]
#             if f'aligner_from_{dim}' in self.aligners:
#                 vectors_by_dim[dim].append(vec)
#                 names_by_dim[dim].append(name)
        
#         for dim, vecs in vectors_by_dim.items():
#             stacked_vecs = torch.stack(vecs, dim=1)
#             aligner = self.aligners[f'aligner_from_{dim}']
#             aligned_stacked = aligner(stacked_vecs)
#             for i, name in enumerate(names_by_dim[dim]):
#                 aligned_vectors[name] = aligned_stacked[:, i, :]
#         return aligned_vectors

#     def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Definition of the forward pass for `PTGAMiniExpert`.

#         Args:
#             data (Data): The input data object containing graph information.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#                 - The graph embedding tensor.
#                 - The feature gate tensor.
#         """
        
#         # --- a) 初始嵌入 ---
#         # 【关键】我们只嵌入这个专家关心的、且存在于当前批次中的字段
#         batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
#         embedded_vectors = self.field_embedder(batch_dict)
        
#         # --- b) 对齐 ---
#         aligned_vectors = self._align_fused(embedded_vectors)
        
#         # --- c) 组装节点特征矩阵 x ---
#         node_feature_list = []
#         for field_name in self.node_fields:
#             vec = aligned_vectors.get(field_name)
#             if vec is not None:
#                 node_feature_list.append(vec)
#             else:
#                 # 【优雅地处理缺失】
#                 # 如果这个包没有某个字段(例如抽象节点, 或真的缺失)，
#                 # 我们使用可学习的“哑元”来填充
#                 dummy_vec = self.dummy_token.expand(data.num_graphs, -1)
#                 node_feature_list.append(dummy_vec)

#         stacked_x = torch.stack(node_feature_list, dim=1) # [B, N, D]
#         stacked_x_normed = self.align_norm(stacked_x)
        
#         # --- d) 应用特征掩码 ---
#         feature_gate = torch.sigmoid(self.feature_mask_logits)

#         # # ===== [NEW] Light Drop-Feature Regularization =====
#         # if self.training:
#         #     drop_prob = 0.05  # 非常保守，推荐从 0.05 开始
#         #     drop_mask = torch.bernoulli(
#         #         torch.full_like(feature_gate, 1.0 - drop_prob)
#         #     )
#         #     feature_gate = feature_gate * drop_mask
#         # # ===================================================

#         mask_for_broadcast = feature_gate.view(1, -1, 1)
#         gated_stacked_x = stacked_x_normed * mask_for_broadcast
        
#         x = gated_stacked_x.view(-1, self.hidden_dim) # [B*N, D]
        
#         # --- e) GNN计算 ---
#         # 【关键】使用这个专家【自己】的图结构
#         # edge_index = self.edge_index.to(x.device) 
#         edge_index = data.edge_index.to(x.device)

#         batch_idx = data.batch.to(x.device)
        
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = self.conv2(x, edge_index)
        
#         # --- f) 全局池化 ---
#         # graph_embedding 代表了这个协议层（例如'ip'）的最终语义向量
#         graph_embedding = global_mean_pool(x, batch_idx) # [B, D]
        
#         return graph_embedding, feature_gate

# # ==============================================================================
# # 2. “顶层聚合器” (HierarchicalMoE)
# # ==============================================================================

# class HierarchicalMoE(nn.Module):
#     """
#     【终极架构】
#     一个“分层语义”的专家混合模型。
#     它“拥有”所有微型专家，并负责“聚合”它们的意见，做出最终分类。
#     """
#     def __init__(self, 
#                  config_path: str, 
#                  vocab_path: str, 
#                  num_classes: int, 
#                  expert_graph_info: Dict[str, Dict], # <-- 从Dataset中获取的专家定义
#                  use_flow_features: bool = False,
#                  num_flow_features: int = 0,
#                  hidden_dim: int = 128, 
#                  num_heads: int = 4, 
#                  dropout_rate: float = 0.3):
#         super().__init__()
        
#         self.use_flow_features = use_flow_features
        
#         # --- 1. 创建【唯一】的、将被所有专家【共享】的FieldEmbedding ---
#         self.shared_field_embedder = FieldEmbedding(config_path, vocab_path)
        
#         # --- 2. 创建【所有】的“微型”专家 ---
#         self.experts = nn.ModuleDict()
#         total_embedding_dim = 0
        
#         for name, graph_info in expert_graph_info.items():
#             print(f" -> Initializing expert: '{name}'")
#             self.experts[name] = PTGAMiniExpert(
#                 field_embedder=self.shared_field_embedder,
#                 node_fields_list=graph_info['all_nodes'],
#                 edge_index=graph_info['edge_index'],
#                 hidden_dim=hidden_dim,
#                 num_heads=num_heads,
#                 dropout_rate=dropout_rate
#             )
#             total_embedding_dim += hidden_dim # 每个专家贡献 hidden_dim
#         total_gnn_dim = total_embedding_dim
#         # self.num_gnn_experts = len(self.experts) # 记录 GNN 专家数量

#         # --- 3. 【可选】创建“流特征”嵌入器 ---
#         # self.flow_stats_embedder = None
#         # self.num_flow_experts = 0
#         if self.use_flow_features:
#             if num_flow_features <= 0:
#                 raise ValueError("num_flow_features 必须大于0")
            
#             print(" -> Initializing flow stats embedder...")
#             flow_embed_dim = hidden_dim // 2 # 64
#             # flow_embed_dim = total_embedding_dim // 2
#             # self.flow_stats_embedder = nn.Sequential(
#             #     nn.LayerNorm(num_flow_features), 
#             #     nn.Linear(num_flow_features, 64),
#             #     nn.LeakyReLU(),
#             #     nn.Linear(64, flow_embed_dim)
#             # )

#             self.flow_stats_embedder = nn.Sequential(
#                 nn.LayerNorm(num_flow_features),
#                 nn.Linear(num_flow_features, hidden_dim), # 64 -> 128
#                 nn.LeakyReLU(),
#                 nn.Dropout(p=dropout_rate),
#                 nn.Linear(hidden_dim, hidden_dim * 2), # 新增一层 (128 -> 256)
#                 nn.LeakyReLU(),
#                 nn.Dropout(p=dropout_rate),
#                 nn.Linear(hidden_dim * 2, flow_embed_dim) # 输出 (256 -> 64)
#             )
#             # self.num_flow_experts = 1
#             self.flow_output_norm = nn.LayerNorm(flow_embed_dim)
#             total_embedding_dim += flow_embed_dim # 流特征贡献 flow_embed_dim

#         # --- 2. 【!! 核心修改：独立的归一化层 !!】 ---
        
#         # # a. 为 *所有* GNN 专家的拼接输出创建一个 LayerNorm
#         # self.gnn_output_norm = nn.LayerNorm(total_gnn_dim)
        
#         # # --- 【!! 实施建议 5：专家重要性门控 !!】 ---
#         # num_expert_blocks = 1 + (1 if self.use_flow_features else 0)
#         # self.expert_gate = nn.Parameter(torch.zeros(num_expert_blocks))
#         # print(f" -> Initializing Expert Gate with {num_expert_blocks} weights.")

#         # # --- 4. 创建最终的“聚合器” (一个简单的MLP) ---
#         # print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
#         # self.aggregator = nn.Sequential(
#         #     # nn.LayerNorm(total_embedding_dim), 
#         #     nn.Linear(total_embedding_dim, hidden_dim * 2), # 放大
#         #     nn.LeakyReLU(),
#         #     nn.Dropout(p=dropout_rate),
#         #     nn.Linear(hidden_dim * 2, num_classes)
#         # )

#         # --- 【!! 核心修改 1：移除 GNN 块归一化 !!】 ---
#         # (我们不再有“融合”的 GNN 块了，所以这个norm必须被删除)
#         # self.gnn_output_norm = nn.LayerNorm(total_gnn_dim)

#         # --- 【!! 核心修改 2：创建“每层”的专家门控 !!】 ---

#         # a. 获取所有 GNN 专家的名字 (并排序，以保证顺序)
#         self.gnn_expert_names = sorted(list(self.experts.keys()))
#         num_gnn_experts = len(self.gnn_expert_names)
#         print(f" -> Found {num_gnn_experts} GNN experts: {self.gnn_expert_names}")

#         # b. 确定总门控数量
#         num_flow_experts = 1 if self.use_flow_features else 0
#         num_total_experts = num_gnn_experts + num_flow_experts

#         # c. 创建门控参数 (例如: 4个GNN + 1个Flow = 5个参数)
#         self.expert_gate = nn.Parameter(torch.zeros(num_total_experts))

#         # d. 存储名称列表，供“get_expert_importance”函数使用
#         self.all_expert_names = self.gnn_expert_names + (["Flow_Features_Block"] if self.use_flow_features else [])
#         print(f" -> Initializing Expert Gate with {num_total_experts} weights for: {self.all_expert_names}")

#         # --- 【!! 核心修改 3：激活 Aggregator 的归一化 !!】 ---
#         print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
#         self.aggregator = nn.Sequential(
#             # (现在所有专家都已加权，我们在“拼接后”进行一次总归一化)
#             nn.LayerNorm(total_embedding_dim), # <-- 【!! 激活此行 !!】
#             nn.Linear(total_embedding_dim, hidden_dim * 2), # 放大
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(hidden_dim * 2, num_classes)
#         )



#     def forward(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
#         all_gates = {} # 用于特征重要性
#         gated_embeddings_list = [] # 存储“加权后”的专家输出

#         # --- a) 获取所有专家门控权重 ---
#         # 形状: [num_total_experts] (例如: 5个)
#         gate_weights = torch.sigmoid(self.expert_gate)

#         # # ===== Expert-level stochastic gating =====
#         # if self.training:
#         #     drop_prob = 0.1
#         #     expert_drop_mask = torch.bernoulli(
#         #         torch.full_like(gate_weights, 1.0 - drop_prob)
#         #     )
#         #     gate_weights = gate_weights * expert_drop_mask
#         # # =========================================

#         # --- b) GNN 专家处理 (逐个加权) ---

#         # (我们必须按 __init__ 中定义的 gnn_expert_names 顺序迭代)
#         for i, expert_name in enumerate(self.gnn_expert_names):
        
#             # 检查批次中是否有此专家的数据 (例如，非TLS包没有'tls'专家)
#             if expert_name in batch_dict:
#                 expert_model = self.experts[expert_name]
#                 data_for_this_expert = batch_dict[expert_name]
                
#                 # 1. 运行 GNN 专家
#                 embedding, gate = expert_model(data_for_this_expert) # [B, D_hidden]
#                 all_gates[expert_name] = gate # 存储特征门控
                
#                 # 2. 【!! 核心 !!】应用“顶层”专家门控
#                 gated_embedding = embedding * gate_weights[i] # [B, D] * [1]
#                 gated_embeddings_list.append(gated_embedding)

#             else:
#                 # 【!! 健壮性 !!】如果这个专家不存在于批次中
#                 # 我们必须添加一个“零向量占位符”来保持拼接维度
                
#                 # a. 获取输出维度
#                 expert_hidden_dim = self.experts[expert_name].hidden_dim
#                 # b. 获取批次大小 (这有点tricky，我们从'all_gates'借一个)
#                 if not all_gates:
#                     # 如果这是第一个专家，我们必须从 batch_dict 找一个
#                     any_expert_name = next(iter(batch_dict))
#                     batch_size = batch_dict[any_expert_name].num_graphs
#                 else:
#                     # 我们可以从已处理的专家那里安全地获取 B

#                     any_processed_gate = next(iter(all_gates.values()))
#                     # (这不可靠，gate是[N]，不是B)
#                     # 让我们用一个更可靠的方法
#                     any_expert_name = next(iter(batch_dict))
#                     batch_size = batch_dict[any_expert_name].num_graphs

#                 # c. 创建零张量
#                 zero_embedding = torch.zeros(
#                     (batch_size, expert_hidden_dim), 
#                     device=gate_weights.device, 
#                     type=gate_weights.dtype
#                 )
#                 gated_embeddings_list.append(zero_embedding)

#         if not gated_embeddings_list:
#             raise ValueError("没有任何 GNN 专家输出了嵌入向量！")

#         # --- c) 【条件】处理流特征 (逐个加权) ---
#         if self.use_flow_features:
#             if 'flow_stats' not in batch_dict: raise ValueError("...")
            
#             flow_stats_input = batch_dict['flow_stats']
#             if flow_stats_input.dim() == 3 and flow_stats_input.shape[1] == 1:
#                 flow_stats_input = flow_stats_input.squeeze(1)
                
#             # 1. 运行 Flow 专家
#             flow_embedding = self.flow_stats_embedder(flow_stats_input)
#             flow_normed = self.flow_output_norm(flow_embedding) 
            
#             # 2. 【!! 核心 !!】应用“顶层”专家门控 (最后一个门控)
#             flow_gate_weight = gate_weights[-1]
#             gated_flow_embedding = flow_normed * flow_gate_weight
#             gated_embeddings_list.append(gated_flow_embedding)
            
#         # --- d) 最终拼接 & 分类 ---
#         # (现在拼接的是 *所有* 专家的 *加权* 输出)
#         combined_embedding = torch.cat(gated_embeddings_list, dim=1) 
#         # (Aggregator 现在会先执行 LayerNorm)
#         logits = self.aggregator(combined_embedding) 
#         return logits, all_gates
    
#     def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
#         """
#         【新】一个辅助函数，用于分析【所有】专家的特征重要性。
#         """
#         reports = {}
#         for name, expert_model in self.experts.items():
#             with torch.no_grad():
#                 final_mask_weights = torch.sigmoid(expert_model.feature_mask_logits).cpu().numpy()
            
#             feature_importance_df = pd.DataFrame({
#                 'feature_name': expert_model.node_fields,
#                 'importance_score': final_mask_weights
#             })
#             reports[name] = feature_importance_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
#         return reports

#     def get_expert_importance(self) -> pd.DataFrame:
#         """
#         【新 - V2版】分析“每层”GNN专家和Flow专家的重要性。
#         """
#         # 1. 获取在 __init__ 中定义的名称列表
#         # (例如: ['eth', 'ip', 'tcp', 'tls', 'Flow_Features_Block'])
#         expert_names = self.all_expert_names

#         # 2. 获取权重
#         with torch.no_grad():
#             final_gate_weights = torch.sigmoid(self.expert_gate).cpu().numpy()

#         # 3. 创建 DataFrame
#         num_weights = len(final_gate_weights)
#         num_names = len(expert_names)

#         if num_weights != num_names:
        
#             raise RuntimeError(f"专家重要性权重({num_weights})与名称({num_names})数量不匹配！")

#         importance_df = pd.DataFrame({
#             'expert_name': expert_names,
#             'importance_score': final_gate_weights
#         })
#         return importance_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
