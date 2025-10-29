import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List, Tuple
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
        B = data.num_graphs # Get batch size from the Batch object
        # --- c) 组装节点特征矩阵 x ---
        node_feature_list = []

        for field_name in self.node_fields:
            vec = aligned_vectors.get(field_name)

            processed_vec = None
            if vec is not None:
                # node_feature_list.append(vec)
                # --- 【!! 核心修复 !!】 ---
                # Check if the vector is 3D with a middle dimension of 1
                if vec.dim() == 3 and vec.shape[1] == 1:
                    # Squeeze it to 2D: [B, 1, D] -> [B, D]
                    processed_vec = vec.squeeze(1)
                elif vec.dim() == 2:
                    # If it's already 2D, use it directly
                    processed_vec = vec
                else:
                    # Handle unexpected shapes if necessary
                    raise ValueError(f"Unexpected shape for aligned vector '{field_name}': {vec.shape}")
                # --- [!! 修复结束 !!] ---
            else:
                # 【优雅地处理缺失】
                # 如果这个包没有某个字段(例如抽象节点, 或真的缺失)，
                # 我们使用可学习的“哑元”来填充
                # dummy_vec = self.dummy_token.expand(data.num_graphs, -1)
                # node_feature_list.append(dummy_vec)
                processed_vec = self.dummy_token.expand(B, -1)
            node_feature_list.append(processed_vec)
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

        # --- 3. 【可选】创建“流特征”嵌入器 ---
        self.flow_stats_embedder = None
        if self.use_flow_features:
            if num_flow_features <= 0:
                raise ValueError("num_flow_features 必须大于0")
            
            print(" -> Initializing flow stats embedder...")
            # flow_embed_dim = hidden_dim // 2 # 64
            flow_embed_dim = hidden_dim 
            self.flow_stats_embedder = nn.Sequential(
                nn.Linear(num_flow_features, 64),
                nn.LeakyReLU(),
                nn.Linear(64, flow_embed_dim)
            )
            # total_embedding_dim += flow_embed_dim # 流特征贡献 flow_embed_dim

        # # --- 4. 创建最终的“聚合器” (一个简单的MLP) ---
        # print(f" -> Initializing aggregator (input dim: {total_embedding_dim})")
        # self.aggregator = nn.Sequential(
        #     nn.Linear(total_embedding_dim, hidden_dim * 2), # 放大
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim * 2, num_classes)
        # )

        # 【!! 核心修复 1：添加可学习的“专家类型”编码 !!】
        #
        # 计算我们总共有多少“输入 Token”
        # ( GNN 专家数 + 1个流专家 + 1个 [CLS] Token )
        self.num_experts = len(self.experts)
        num_tokens = self.num_experts
        if self.use_flow_features:
            num_tokens += 1
    
        # 1 (CLS) + N (GNN) + 1 (Flow)
        total_seq_len = 1 + num_tokens 
    
        # 可学习的“位置/类型”编码
        self.positional_embedding = nn.Parameter(torch.randn(1, total_seq_len, hidden_dim))
    
        # 【!! 核心修复 2：添加输入规范化层 !!】
        #
        # 这个 LayerNorm 将“驯服”所有专家，使它们处于同一尺度
        self.input_norm = nn.LayerNorm(hidden_dim)
        print(f" -> Initializing Transformer Aggregator (dim: {hidden_dim})")
        # 1. 定义一个 [CLS] (Class) Token，它将代表“最终意见”
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 2. 你的新 aggregator 是一个 Transformer 层
        self.agg_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, # 4 个注意力头 (可以调整)
            dim_feedforward=hidden_dim * 4, # 标准配置
            dropout=0.1, # (可以调整)
            batch_first=True # <-- 重要！确保输入是 [B, SeqLen, Dim]
        )

        # 3. 最终的分类器
        # self.agg_classifier = nn.Linear(hidden_dim, num_classes)
        self.agg_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # 放大
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, batch_dict: Dict[str, Data]) -> torch.Tensor: 
        """
        前向传播：分解 -> 专家处理 -> 融合 -> 分类
        """
        expert_embeddings = []
        all_gates = {} # 用于收集所有门控，以便计算正则化损失

        # --- a) “语义分解”与专家处理 ---
        for expert_name, expert_model in self.experts.items():
            # 从批处理字典中，获取这位专家的数据
            # data_for_this_expert = batch_dict[expert_name] # 这在DataLoader v1.x中有效
            
            # 【健壮性】PyG DataLoader会把字典“压平”，我们需要在设备上重新组装
            data_for_this_expert = batch_dict[expert_name].to(next(self.parameters()).device)
            
            # 【关键】调用专家模型，得到嵌入和门控
            embedding, gate = expert_model(data_for_this_expert)
            expert_embeddings.append(embedding)
            all_gates[expert_name] = gate
            
        # --- b) 【可选】提取流特征 ---
        if self.use_flow_features:
            if 'flow_stats' not in batch_dict:
                raise ValueError("模型处于 use_flow_features=True 模式, 但GNNTrafficDataset未提供 'data.flow_stats'。")
            
            flow_stats_input = batch_dict['flow_stats'].to(next(self.parameters()).device)
            flow_embedding = self.flow_stats_embedder(flow_stats_input)
            expert_embeddings.append(flow_embedding)
        
        # # --- c) “最终决策” (融合) ---
        # combined_embedding = torch.cat(expert_embeddings, dim=1)
        
        # # --- d) 分类 ---
        # logits = self.aggregator(combined_embedding)
        
        # # 【重要】返回logits，以及【所有】专家的门控，以便计算总正则化损失
        # return logits, all_gates

        # --- c) 【!! 核心修改：智能融合 (V2) !!】 ---
                # 1. 堆叠: [B, N_experts_total, D]
        expert_seq = torch.stack(expert_embeddings, dim=1)
                # 2. 准备 [CLS] Token: [B, 1, D]
        B = expert_seq.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
                # 3. 拼接: [B, 1 + N_experts_total, D]
        full_seq = torch.cat([cls_tokens, expert_seq], dim=1) 
                # 4. 【新】添加“专家类型/位置”编码
        #    我们广播 self.positional_embedding 到整个批次
        full_seq = full_seq + self.positional_embedding
                # 5. 【新】在送入 Transformer 之前，对整个序列进行规范化
        #    这将解决“耳语 vs 大喊”的数值尺度问题
        full_seq = self.input_norm(full_seq)
                # 6. 通过 Transformer 运行 (不变)
        attn_output = self.agg_attention(full_seq)
                # 7. 只取出 [CLS] Token (不变)
        final_embedding = attn_output[:, 0, :]
                # --- d) 分类 (不变) ---
        logits = self.agg_classifier(final_embedding)
        
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

