import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
from torch_geometric.data import Data

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
        self.flow_stats_embedder = None
        self.flow_output_norm = None
        flow_embed_dim = 0
        # self.num_flow_experts = 0
        if self.use_flow_features:
            if num_flow_features <= 0:
                raise ValueError("num_flow_features 必须大于0")
            
            print(" -> Initializing flow stats embedder...")
            flow_embed_dim = hidden_dim // 2 # 64
            # flow_embed_dim = total_embedding_dim // 2
            self.flow_stats_embedder = nn.Sequential(
                nn.LayerNorm(num_flow_features), 
                nn.Linear(num_flow_features, 64),
                nn.LeakyReLU(),
                nn.Linear(64, flow_embed_dim)
            )
            # self.num_flow_experts = 1
            self.flow_output_norm = nn.LayerNorm(flow_embed_dim)
            # 【!! 新增 1：流特征重要性门控 !!】
            # 为 *输入* 的流特征创建一个可学习的门控
            # 形状: [num_flow_features] (e.g., 8)
            self.flow_feature_gate = nn.Parameter(torch.full((num_flow_features,), 2.2)) 

        total_embedding_dim += flow_embed_dim # 流特征贡献 flow_embed_dim

        # --- 2. 【!! 核心修改：独立的归一化层 !!】 ---
        
        # a. 为 *所有* GNN 专家的拼接输出创建一个 LayerNorm
        self.gnn_output_norm = nn.LayerNorm(total_gnn_dim)
        
        # 【!! 新增 2：专家重要性门控 !!】
        # 为 *输出* 的专家嵌入创建一个可学习的门控
        # 1个GNN块 + 1个Flow块
        num_expert_blocks = 1 + (1 if self.use_flow_features else 0)
        # 形状: [num_expert_blocks] (e.g., 2)
        self.expert_gate = nn.Parameter(torch.full((num_expert_blocks,), 2.2))

        # --- 4. 创建最终的“聚合器” (一个简单的MLP) ---
        print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
        self.aggregator = nn.Sequential(
            # nn.LayerNorm(total_embedding_dim), 
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

    def forward(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        gnn_embeddings = [] # <-- 【修改】只收集 GNN
        all_gates = {} 
        
        # --- a) GNN 专家处理 ---
        for expert_name, expert_model in self.experts.items():
            if expert_name not in batch_dict: continue
            data_for_this_expert = batch_dict[expert_name]
            embedding, gate = expert_model(data_for_this_expert)
            gnn_embeddings.append(embedding) # <-- 只添加 GNN 嵌入
            all_gates[expert_name] = gate
            
        if not gnn_embeddings:
             raise ValueError("没有任何 GNN 专家输出了嵌入向量！")

        # --- 【!! 核心修改：独立归一化 !!】 ---
        
        # 1. 拼接 *所有* GNN 专家，并归一化
        gnn_combined = torch.cat(gnn_embeddings, dim=1) # [B, total_gnn_dim]
        gnn_normed = self.gnn_output_norm(gnn_combined) # [B, total_gnn_dim]
        
        # 2. 准备最终要拼接的列表
        final_embeddings_list = [gnn_normed]
            
        # 3. 【条件】处理流特征
        if self.use_flow_features:
            if 'flow_stats' not in batch_dict: raise ValueError("...")
            
            flow_stats_input = batch_dict['flow_stats']
            if flow_stats_input.dim() == 3 and flow_stats_input.shape[1] == 1:
                flow_stats_input = flow_stats_input.squeeze(1)

            # 【!! 新增 1：应用流特征门控 !!】
            # gate: [num_features] -> [1, num_features]
            flow_gate_weights = torch.sigmoid(self.flow_feature_gate).unsqueeze(0)
            # [B, num_features] * [1, num_features]
            gated_flow_input = flow_stats_input * flow_gate_weights
            
            # flow_embedding = self.flow_stats_embedder(flow_stats_input) # [B, flow_embed_dim]
            flow_embedding = self.flow_stats_embedder(gated_flow_input) # <-- 使用门控后的输入
            # 独立归一化流特征
            flow_normed = self.flow_output_norm(flow_embedding) # [B, flow_embed_dim]
            
            final_embeddings_list.append(flow_normed) # 添加到列表
        
        # --- c) 最终拼接 ---
        # (现在拼接的是两个“音量相同”的向量)
        combined_embedding = torch.cat(final_embeddings_list, dim=1) 

        # 【!! 新增 2：应用专家门控 !!】
        # gate: [num_blocks] -> [1, num_blocks]
        expert_gate_weights = torch.sigmoid(self.expert_gate).unsqueeze(0)
        
        # --- d) 分类 ---
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

