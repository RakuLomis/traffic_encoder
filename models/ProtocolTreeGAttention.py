import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List
import pandas as pd

class ProtocolTreeGAttention(nn.Module):
    def __init__(self, #config_path: str, vocab_path: str, 
                num_classes: int, node_fields_list: List[str], field_embedder: FieldEmbedding, 
                hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        # self.field_embedder = FieldEmbedding(config_path, vocab_path)
        self.field_embedder = field_embedder
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.node_fields = node_fields_list # 接收从Dataset传来的、当前Block的节点列表
        
        # --- 1. 创建“融合”的对齐层 (Fused Aligners) ---
        self.aligners = nn.ModuleDict()
        # 扫描所有可能的字段，按其嵌入维度创建唯一的对齐层
        unique_dims = set(s[1] - s[0] for s in self.field_embedder.embedding_slices.values())
        for dim in unique_dims:
            if dim > 0:
                self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

        # --- 2. 创建GNN层和分类器 ---
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        # self.classifier = nn.Linear(hidden_dim, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(), # 可能造成了梯度消失
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim // 2, num_classes)
        # )
        self.classifier = nn.Sequential(
            # 第一个线性层
            nn.Linear(hidden_dim, hidden_dim // 2),
            # “稳定器”：批量归一化
            # nn.BatchNorm1d(hidden_dim),
            # “防死亡”激活函数
            nn.LeakyReLU(),
            # Dropout层
            nn.Dropout(p=dropout_rate),
            # 第二个线性层，输出最终的logits
            nn.Linear(hidden_dim // 2, num_classes)
        )
        # ==================== “急救”方案核心修改点 ====================
        
        # a) 【急救方案 1】修正掩码初始化，让门控默认“打开”
        num_nodes = len(self.node_fields)
        initial_logits = torch.full((num_nodes,), 2.2) # sigmoid(2.2) ≈ 0.9
        self.feature_mask_logits = nn.Parameter(initial_logits)

        # b) 【急救方案 2】增加LayerNorm，将对齐后的向量“救活”
        self.align_norm = nn.LayerNorm(hidden_dim)
        
        # c) 【急救方案 3】用可学习的“哑元”替换零向量，保证梯度流
        self.dummy_token = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.01)

        # =====================================================================


    def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor], device: torch.device):
        """
        一个辅助函数，用于执行高效的“融合对齐”操作。
        """
        aligned_vectors = {}
        
        # a) 按维度对输入向量进行分组
        vectors_by_dim = defaultdict(list)
        names_by_dim = defaultdict(list)
        for name, vec in embedded_vectors.items():
            dim = vec.shape[-1]
            # 只有当存在对应维度的对齐层时，才处理
            if f'aligner_from_{dim}' in self.aligners:
                vectors_by_dim[dim].append(vec)
                names_by_dim[dim].append(name)
        
        # b) 对每个维度组执行一次批处理对齐
        for dim, vecs in vectors_by_dim.items():
            # (num_vecs_in_group, batch_size, dim) -> (batch_size, num_vecs_in_group, dim)
            stacked_vecs = torch.stack(vecs, dim=1)
            
            aligner = self.aligners[f'aligner_from_{dim}']
            # 一次性完成对齐
            aligned_stacked = aligner(stacked_vecs)
            
            # c) 将结果“散射”回字典
            for i, name in enumerate(names_by_dim[dim]):
                aligned_vectors[name] = aligned_stacked[:, i, :]
        
        return aligned_vectors

    # def forward(self, data) -> torch.Tensor:
    # def forward_features(self, data) -> torch.Tensor:
    #     # data 是一个由PyG DataLoader准备好的批处理图对象
        
    #     # --- 步骤一：初始嵌入 ---
    #     batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
    #     embedded_vectors = self.field_embedder(batch_dict)
        
    #     # ==================== 核心修改点：向量化的对齐与组装 ====================
        
    #     # --- 步骤二：融合对齐 ---
    #     aligned_vectors = self._align_fused(embedded_vectors, data.edge_index.device)
        
    #     # --- 步骤三：并行组装节点特征矩阵 x ---
    #     node_feature_list = []
    #     # 按照Dataset中定义的节点顺序进行组装
    #     for field_name in self.node_fields:
    #         # 使用 .get() 方法，如果字段不存在于对齐后的字典中（例如抽象节点），则返回None
    #         vec = aligned_vectors.get(field_name)
    #         if vec is not None:
    #             node_feature_list.append(vec)
    #         else:
    #             # 为抽象节点或缺失的真实节点创建零向量占位符
    #             zero_vec = torch.zeros(data.num_graphs, self.hidden_dim, device=data.edge_index.device)
    #             node_feature_list.append(zero_vec)

    #     # (batch_size, num_nodes, hidden_dim)
    #     stacked_x = torch.stack(node_feature_list, dim=1)
    #     # (batch_size * num_nodes, hidden_dim)
    #     x = stacked_x.view(-1, self.hidden_dim)
        
    #     # ========================================================================

    #     # --- 步骤四：GNN计算 ---
    #     edge_index, batch_idx = data.edge_index, data.batch
    #     x = F.dropout(x, p=self.dropout_rate, training=self.training)
    #     x = self.conv1(x, edge_index)
    #     x = F.elu(x)
    #     x = self.conv2(x, edge_index)
        
    #     # --- 步骤五：全局池化和分类 ---
    #     graph_embedding = global_mean_pool(x, batch_idx)
    #     return graph_embedding
    #     # logits = self.classifier(graph_embedding)
        
    #     # return logits
    # def forward(self, data) -> torch.Tensor: 
    #     graph_embedding = self.forward_features(data) 
    #     logits = self.classifier(graph_embedding)
    #     return logits

    def forward(self, data) -> torch.Tensor: 
        # data 是一个由PyG DataLoader准备好的批处理图对象
        
        # --- a) 步骤一：初始嵌入 (无变化) ---
        batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
        embedded_vectors = self.field_embedder(batch_dict)
        
        # --- b) 步骤二：对齐 & 构建节点特征矩阵 (无变化) ---
        # aligned_vectors_list = []
        # for field_name in self.node_fields:
        #     if field_name in embedded_vectors:
        #         vec = embedded_vectors[field_name]
        #         # 检查是否存在对应的aligner
        #         aligner_key = field_name.replace('.', '__')
        #         if aligner_key in self.aligners:
        #             aligner = self.aligners[aligner_key]
        #             aligned_vectors_list.append(aligner(vec))
        #         else: # 如果没有对齐层（例如，嵌入维度为0），则添加零向量
        #             aligned_vectors_list.append(torch.zeros(data.num_graphs, self.hidden_dim, device=data.edge_index.device))
        #     else:
        #         # 为抽象节点或缺失的真实节点创建零向量占位符
        #         zero_vec = torch.zeros(data.num_graphs, self.hidden_dim, device=data.edge_index.device)
        #         aligned_vectors_list.append(zero_vec)
        aligned_vectors = self._align_fused(embedded_vectors, data.edge_index.device)
        
        node_feature_list = []
        for field_name in self.node_fields:
            vec = aligned_vectors.get(field_name)
            if vec is not None:
                node_feature_list.append(vec)
            else:
                # 【急救方案 3 应用】使用可学习的dummy_token替换零向量
                dummy_vec = self.dummy_token.expand(data.num_graphs, -1)
                node_feature_list.append(dummy_vec)

        # (batch_size, num_nodes, hidden_dim)
        stacked_x = torch.stack(node_feature_list, dim=1)
        # stacked_x = torch.stack(aligned_vectors_list, dim=1)
        
        # ==================== 核心修改点 2：应用特征掩码 ====================
        #
        # 在将节点特征送入GNN层之前，我们先用掩码对其进行加权
        #
        # a) 将我们学习的logits，通过sigmoid函数转换成0到1之间的“门控”权重
        #    形状: [num_nodes]
        feature_gate = torch.sigmoid(self.feature_mask_logits)
        
        # b) 为了将掩码应用到批处理数据上，我们需要调整其形状以进行广播
        #    [num_nodes] -> [1, num_nodes, 1]
        mask_for_broadcast = feature_gate.view(1, -1, 1)
        
        # c) 执行加权操作 (Gating)。每个节点的特征向量，都会乘以它对应的学习到的权重
        gated_stacked_x = stacked_x * mask_for_broadcast
        
        # d) 将加权后的张量“压平”，送入GNN
        x = gated_stacked_x.view(-1, self.hidden_dim)
        #
        # =======================================================================

        # ==================== “急救”方案诊断点 ====================
        # 在将x送入GNN前，检查其方差
        if self.training: # 只在训练时打印，避免干扰评估
            x_var = x.var().item()
            if x_var < 1e-4:
                print(f"\n[DIAGNOSTIC WARNING] Variance of GNN input 'x' is critically low: {x_var:.2e}. Gradients may vanish.")
        # ========================================================

        # --- c) 步骤三：GNN计算 (现在使用加权后的 x) ---
        edge_index, batch_idx = data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # --- d) 步骤四：全局池化和分类 (无变化) ---
        graph_embedding = global_mean_pool(x, batch_idx)
        logits = self.classifier(graph_embedding)
        
        return logits, feature_gate

    def get_feature_importance(self) -> pd.DataFrame:
        """
        一个辅助函数，用于在训练后分析学到的特征重要性。
        """
        with torch.no_grad():
            # 获取最终学到的特征权重 (0到1之间)
            final_mask_weights = torch.sigmoid(self.feature_mask_logits).cpu().numpy()
        
        # 创建一个清晰的报告
        feature_importance_df = pd.DataFrame({
            'feature_name': self.node_fields,
            'importance_score': final_mask_weights
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
        
        return feature_importance_df

