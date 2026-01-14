import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
from torch_geometric.data import Data, Batch

# ==============================================================================
# 1. “微型专家” (PTGAMiniExpert)
# ==============================================================================

class PTGAMiniExpert(nn.Module):
    """
    【新】“微型”PTGA专家，专门用于处理一个协议层（如'ip', 'tcp'）的图。
    它不包含最终的分类器，只负责输出一个固定维度的嵌入向量。
    """
    def __init__(self, 
                 field_embedder: FieldEmbedding, 
                 node_fields_list: List[str],
                 edge_index: torch.Tensor,
                 hidden_dim: int = 128, 
                 num_heads: int = 4, 
                 dropout_rate: float = 0.3):
        
        super().__init__()
        
        self.field_embedder = field_embedder # 接收【共享的】嵌入器
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.node_fields = node_fields_list # 这个专家自己的节点列表
        
        # 将这个专家的图结构 (edge_index) 注册为模型的缓冲区
        # 这样它就可以被 .to(device) 自动移动
        self.register_buffer('edge_index', edge_index)

        # --- 1. 创建对齐层 (Aligners) ---
        # (与您之前的逻辑相同，但只为这个专家需要的字段创建)
        self.aligners = nn.ModuleDict()
        unique_dims = set()
        for field_name in self.node_fields:
            if field_name in self.field_embedder.embedding_slices:
                dim = self.field_embedder.embedding_slices[field_name][1] - self.field_embedder.embedding_slices[field_name][0]
                if dim > 0:
                    unique_dims.add(dim)
        
        for dim in unique_dims:
            self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

        # --- 2. 创建GNN层 ---
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
        # --- 3. 创建“急救”组件 (特征掩码、LayerNorm、哑元) ---
        num_nodes = len(self.node_fields)
        initial_logits = torch.full((num_nodes,), 2.2) # 门控默认“打开”
        # initial_logits = torch.zeros(num_nodes)   # sigmoid = 0.5
        self.feature_mask_logits = nn.Parameter(initial_logits)
        self.align_norm = nn.LayerNorm(hidden_dim)
        self.dummy_token = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.01)

    def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor]):
        """辅助函数：执行高效的“融合对齐”操作。"""
        # (这个函数与您之前的版本完全相同)
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

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Definition of the forward pass for `PTGAMiniExpert`.

        Args:
            data (Data): The input data object containing graph information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The graph embedding tensor.
                - The feature gate tensor.
        """
        
        # --- a) 初始嵌入 ---
        # 【关键】我们只嵌入这个专家关心的、且存在于当前批次中的字段
        batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
        embedded_vectors = self.field_embedder(batch_dict)
        
        # --- b) 对齐 ---
        aligned_vectors = self._align_fused(embedded_vectors)
        
        # --- c) 组装节点特征矩阵 x ---
        node_feature_list = []
        for field_name in self.node_fields:
            vec = aligned_vectors.get(field_name)
            if vec is not None:
                node_feature_list.append(vec)
            else:
                # 【优雅地处理缺失】
                # 如果这个包没有某个字段(例如抽象节点, 或真的缺失)，
                # 我们使用可学习的“哑元”来填充
                dummy_vec = self.dummy_token.expand(data.num_graphs, -1)
                node_feature_list.append(dummy_vec)

        stacked_x = torch.stack(node_feature_list, dim=1) # [B, N, D]
        stacked_x_normed = self.align_norm(stacked_x)
        
        # --- d) 应用特征掩码 ---
        feature_gate = torch.sigmoid(self.feature_mask_logits)

        # ===== [NEW] Light Drop-Feature Regularization =====
        if self.training:
            drop_prob = 0.05  # 非常保守，推荐从 0.05 开始
            drop_mask = torch.bernoulli(
                torch.full_like(feature_gate, 1.0 - drop_prob)
            )
            feature_gate = feature_gate * drop_mask
        # ===================================================

        mask_for_broadcast = feature_gate.view(1, -1, 1)
        gated_stacked_x = stacked_x_normed * mask_for_broadcast
        
        x = gated_stacked_x.view(-1, self.hidden_dim) # [B*N, D]
        
        # --- e) GNN计算 ---
        # 【关键】使用这个专家【自己】的图结构
        # edge_index = self.edge_index.to(x.device) 
        edge_index = data.edge_index.to(x.device)

        batch_idx = data.batch.to(x.device)
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # --- f) 全局池化 ---
        # graph_embedding 代表了这个协议层（例如'ip'）的最终语义向量
        graph_embedding = global_mean_pool(x, batch_idx) # [B, D]
        
        return graph_embedding, feature_gate

# ==============================================================================
# 2. “顶层聚合器” (HierarchicalMoE)
# ==============================================================================

class HierarchicalMoE(nn.Module):
    """
    【终极架构】
    一个“分层语义”的专家混合模型。
    它“拥有”所有微型专家，并负责“聚合”它们的意见，做出最终分类。
    """
    def __init__(self, 
                 config_path: str, 
                 vocab_path: str, 
                 num_classes: int, 
                 expert_graph_info: Dict[str, Dict], # <-- 从Dataset中获取的专家定义
                 use_flow_features: bool = False,
                 num_flow_features: int = 0,
                 hidden_dim: int = 128, 
                 num_heads: int = 4, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.use_flow_features = use_flow_features
        
        # --- 1. 创建【唯一】的、将被所有专家【共享】的FieldEmbedding ---
        self.shared_field_embedder = FieldEmbedding(config_path, vocab_path)
        
        # --- 2. 创建【所有】的“微型”专家 ---
        self.experts = nn.ModuleDict()
        total_embedding_dim = 0
        
        for name, graph_info in expert_graph_info.items():
            print(f" -> Initializing expert: '{name}'")
            self.experts[name] = PTGAMiniExpert(
                field_embedder=self.shared_field_embedder,
                node_fields_list=graph_info['all_nodes'],
                edge_index=graph_info['edge_index'],
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            total_embedding_dim += hidden_dim # 每个专家贡献 hidden_dim
        total_gnn_dim = total_embedding_dim
        # self.num_gnn_experts = len(self.experts) # 记录 GNN 专家数量

        # --- 3. 【可选】创建“流特征”嵌入器 ---
        # self.flow_stats_embedder = None
        # self.num_flow_experts = 0
        if self.use_flow_features:
            if num_flow_features <= 0:
                raise ValueError("num_flow_features 必须大于0")
            
            print(" -> Initializing flow stats embedder...")
            flow_embed_dim = hidden_dim // 2 # 64
            # flow_embed_dim = total_embedding_dim // 2
            # self.flow_stats_embedder = nn.Sequential(
            #     nn.LayerNorm(num_flow_features), 
            #     nn.Linear(num_flow_features, 64),
            #     nn.LeakyReLU(),
            #     nn.Linear(64, flow_embed_dim)
            # )

            self.flow_stats_embedder = nn.Sequential(
                nn.LayerNorm(num_flow_features),
                nn.Linear(num_flow_features, hidden_dim), # 64 -> 128
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim * 2), # 新增一层 (128 -> 256)
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim * 2, flow_embed_dim) # 输出 (256 -> 64)
            )
            # self.num_flow_experts = 1
            self.flow_output_norm = nn.LayerNorm(flow_embed_dim)
            total_embedding_dim += flow_embed_dim # 流特征贡献 flow_embed_dim

        # --- 2. 【!! 核心修改：独立的归一化层 !!】 ---
        
        # # a. 为 *所有* GNN 专家的拼接输出创建一个 LayerNorm
        # self.gnn_output_norm = nn.LayerNorm(total_gnn_dim)
        
        # # --- 【!! 实施建议 5：专家重要性门控 !!】 ---
        # num_expert_blocks = 1 + (1 if self.use_flow_features else 0)
        # self.expert_gate = nn.Parameter(torch.zeros(num_expert_blocks))
        # print(f" -> Initializing Expert Gate with {num_expert_blocks} weights.")

        # # --- 4. 创建最终的“聚合器” (一个简单的MLP) ---
        # print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
        # self.aggregator = nn.Sequential(
        #     # nn.LayerNorm(total_embedding_dim), 
        #     nn.Linear(total_embedding_dim, hidden_dim * 2), # 放大
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim * 2, num_classes)
        # )

        # --- 【!! 核心修改 1：移除 GNN 块归一化 !!】 ---
        # (我们不再有“融合”的 GNN 块了，所以这个norm必须被删除)
        # self.gnn_output_norm = nn.LayerNorm(total_gnn_dim)

        # --- 【!! 核心修改 2：创建“每层”的专家门控 !!】 ---

        # a. 获取所有 GNN 专家的名字 (并排序，以保证顺序)
        self.gnn_expert_names = sorted(list(self.experts.keys()))
        num_gnn_experts = len(self.gnn_expert_names)
        print(f" -> Found {num_gnn_experts} GNN experts: {self.gnn_expert_names}")

        # b. 确定总门控数量
        num_flow_experts = 1 if self.use_flow_features else 0
        num_total_experts = num_gnn_experts + num_flow_experts

        # c. 创建门控参数 (例如: 4个GNN + 1个Flow = 5个参数)
        self.expert_gate = nn.Parameter(torch.zeros(num_total_experts))

        # d. 存储名称列表，供“get_expert_importance”函数使用
        self.all_expert_names = self.gnn_expert_names + (["Flow_Features_Block"] if self.use_flow_features else [])
        print(f" -> Initializing Expert Gate with {num_total_experts} weights for: {self.all_expert_names}")

        # --- 【!! 核心修改 3：激活 Aggregator 的归一化 !!】 ---
        print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
        self.aggregator = nn.Sequential(
            # (现在所有专家都已加权，我们在“拼接后”进行一次总归一化)
            nn.LayerNorm(total_embedding_dim), # <-- 【!! 激活此行 !!】
            nn.Linear(total_embedding_dim, hidden_dim * 2), # 放大
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )


    # def forward(self, batch_dict: Dict[str, Data]) -> torch.Tensor: 
    #     """
    #     前向传播：分解 -> 专家处理 -> 融合 -> 分类
    #     """
    #     expert_embeddings = []
    #     all_gates = {} # 用于收集所有门控，以便计算正则化损失

    #     # --- a) “语义分解”与专家处理 ---
    #     for expert_name, expert_model in self.experts.items():
    #         # 从批处理字典中，获取这位专家的数据
    #         # data_for_this_expert = batch_dict[expert_name] # 这在DataLoader v1.x中有效
            
    #         # 【健壮性】PyG DataLoader会把字典“压平”，我们需要在设备上重新组装
    #         data_for_this_expert = batch_dict[expert_name].to(next(self.parameters()).device)
            
    #         # 【关键】调用专家模型，得到嵌入和门控
    #         embedding, gate = expert_model(data_for_this_expert)
    #         expert_embeddings.append(embedding)
    #         all_gates[expert_name] = gate

            
    #     # --- b) 【可选】提取流特征 ---
    #     flow_embedding = None # 初始化
    #     if self.use_flow_features:
    #         if 'flow_stats' not in batch_dict:
    #             raise ValueError("模型处于 use_flow_features=True 模式, 但GNNTrafficDataset未提供 'data.flow_stats'。")
            
    #         flow_stats_input = batch_dict['flow_stats'].to(next(self.parameters()).device)

    #         flow_stats_input = torch.nan_to_num(flow_stats_input, nan=0.0, posinf=0.0, neginf=0.0)
    #         # --- 【!! 核心修复 !!】 ---
    #         # DataLoader collate added an extra dimension (dim 1).
    #         # We need to remove it before passing to the embedder.
    #         if flow_stats_input.dim() == 3 and flow_stats_input.shape[1] == 1:
    #             flow_stats_input = flow_stats_input.squeeze(1) # [B, 1, num_features] -> [B, num_features]
    #         # --- [!! 修复结束 !!] ---

    #         flow_embedding = self.flow_stats_embedder(flow_stats_input)
    #         expert_embeddings.append(flow_embedding)
        
    #     # --- c) “最终决策” (融合) ---
    #     combined_embedding = torch.cat(expert_embeddings, dim=1)
        
    #     # --- d) 分类 ---
    #     logits = self.aggregator(combined_embedding)
        
    #     # 【重要】返回logits，以及【所有】专家的门控，以便计算总正则化损失
    #     return logits, all_gates

    # def forward(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
    #     gnn_embeddings = [] # <-- 【修改】只收集 GNN
    #     all_gates = {} 
        
    #     # --- a) GNN 专家处理 ---
    #     for expert_name, expert_model in self.experts.items():
    #         if expert_name not in batch_dict: continue
    #         data_for_this_expert = batch_dict[expert_name]
    #         embedding, gate = expert_model(data_for_this_expert)
    #         gnn_embeddings.append(embedding) # <-- 只添加 GNN 嵌入
    #         all_gates[expert_name] = gate
            
    #     if not gnn_embeddings:
    #          raise ValueError("没有任何 GNN 专家输出了嵌入向量！")

    #     # --- b) 独立归一化 (不变) ---
    #     gnn_combined = torch.cat(gnn_embeddings, dim=1)
    #     gnn_normed = self.gnn_output_norm(gnn_combined)
    #     final_embeddings_list = [gnn_normed]
            
    #     # --- c) 【条件】处理流特征 ---
    #     if self.use_flow_features:
    #         if 'flow_stats' not in batch_dict: raise ValueError("...")
            
    #         flow_stats_input = batch_dict['flow_stats']
    #         if flow_stats_input.dim() == 3 and flow_stats_input.shape[1] == 1:
    #             flow_stats_input = flow_stats_input.squeeze(1)
            
    #         flow_embedding = self.flow_stats_embedder(flow_stats_input)
    #         flow_normed = self.flow_output_norm(flow_embedding) 
    #         final_embeddings_list.append(flow_normed)
        
    #     # --- d) 最终拼接 & 分类 (不变) ---
    #     combined_embedding = torch.cat(final_embeddings_list, dim=1) 
    #     logits = self.aggregator(combined_embedding) 
        
    #     return logits, all_gates

    def forward(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        all_gates = {} # 用于特征重要性
        gated_embeddings_list = [] # 存储“加权后”的专家输出

        # --- a) 获取所有专家门控权重 ---
        # 形状: [num_total_experts] (例如: 5个)
        gate_weights = torch.sigmoid(self.expert_gate)

        # # ===== Expert-level stochastic gating =====
        # if self.training:
        #     drop_prob = 0.1
        #     expert_drop_mask = torch.bernoulli(
        #         torch.full_like(gate_weights, 1.0 - drop_prob)
        #     )
        #     gate_weights = gate_weights * expert_drop_mask
        # # =========================================

        # --- b) GNN 专家处理 (逐个加权) ---

        # (我们必须按 __init__ 中定义的 gnn_expert_names 顺序迭代)
        for i, expert_name in enumerate(self.gnn_expert_names):
        
            # 检查批次中是否有此专家的数据 (例如，非TLS包没有'tls'专家)
            if expert_name in batch_dict:
                expert_model = self.experts[expert_name]
                data_for_this_expert = batch_dict[expert_name]
                
                # 1. 运行 GNN 专家
                embedding, gate = expert_model(data_for_this_expert) # [B, D_hidden]
                all_gates[expert_name] = gate # 存储特征门控
                
                # 2. 【!! 核心 !!】应用“顶层”专家门控
                gated_embedding = embedding * gate_weights[i] # [B, D] * [1]
                gated_embeddings_list.append(gated_embedding)

            else:
                # 【!! 健壮性 !!】如果这个专家不存在于批次中
                # 我们必须添加一个“零向量占位符”来保持拼接维度
                
                # a. 获取输出维度
                expert_hidden_dim = self.experts[expert_name].hidden_dim
                # b. 获取批次大小 (这有点tricky，我们从'all_gates'借一个)
                if not all_gates:
                    # 如果这是第一个专家，我们必须从 batch_dict 找一个
                    any_expert_name = next(iter(batch_dict))
                    batch_size = batch_dict[any_expert_name].num_graphs
                else:
                    # 我们可以从已处理的专家那里安全地获取 B

                    any_processed_gate = next(iter(all_gates.values()))
                    # (这不可靠，gate是[N]，不是B)
                    # 让我们用一个更可靠的方法
                    any_expert_name = next(iter(batch_dict))
                    batch_size = batch_dict[any_expert_name].num_graphs

                # c. 创建零张量
                zero_embedding = torch.zeros(
                    (batch_size, expert_hidden_dim), 
                    device=gate_weights.device, 
                    type=gate_weights.dtype
                )
                gated_embeddings_list.append(zero_embedding)

        if not gated_embeddings_list:
            raise ValueError("没有任何 GNN 专家输出了嵌入向量！")

        # --- c) 【条件】处理流特征 (逐个加权) ---
        if self.use_flow_features:
            if 'flow_stats' not in batch_dict: raise ValueError("...")
            
            flow_stats_input = batch_dict['flow_stats']
            if flow_stats_input.dim() == 3 and flow_stats_input.shape[1] == 1:
                flow_stats_input = flow_stats_input.squeeze(1)
                
            # 1. 运行 Flow 专家
            flow_embedding = self.flow_stats_embedder(flow_stats_input)
            flow_normed = self.flow_output_norm(flow_embedding) 
            
            # 2. 【!! 核心 !!】应用“顶层”专家门控 (最后一个门控)
            flow_gate_weight = gate_weights[-1]
            gated_flow_embedding = flow_normed * flow_gate_weight
            gated_embeddings_list.append(gated_flow_embedding)
            
        # --- d) 最终拼接 & 分类 ---
        # (现在拼接的是 *所有* 专家的 *加权* 输出)
        combined_embedding = torch.cat(gated_embeddings_list, dim=1) 
        # (Aggregator 现在会先执行 LayerNorm)
        logits = self.aggregator(combined_embedding) 
        return logits, all_gates
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        【新】一个辅助函数，用于分析【所有】专家的特征重要性。
        """
        reports = {}
        for name, expert_model in self.experts.items():
            with torch.no_grad():
                final_mask_weights = torch.sigmoid(expert_model.feature_mask_logits).cpu().numpy()
            
            feature_importance_df = pd.DataFrame({
                'feature_name': expert_model.node_fields,
                'importance_score': final_mask_weights
            })
            reports[name] = feature_importance_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
        return reports

    def get_expert_importance(self) -> pd.DataFrame:
        """
        【新 - V2版】分析“每层”GNN专家和Flow专家的重要性。
        """
        # 1. 获取在 __init__ 中定义的名称列表
        # (例如: ['eth', 'ip', 'tcp', 'tls', 'Flow_Features_Block'])
        expert_names = self.all_expert_names

        # 2. 获取权重
        with torch.no_grad():
            final_gate_weights = torch.sigmoid(self.expert_gate).cpu().numpy()

        # 3. 创建 DataFrame
        num_weights = len(final_gate_weights)
        num_names = len(expert_names)

        if num_weights != num_names:
        
            raise RuntimeError(f"专家重要性权重({num_weights})与名称({num_names})数量不匹配！")

        importance_df = pd.DataFrame({
            'expert_name': expert_names,
            'importance_score': final_gate_weights
        })
        return importance_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
