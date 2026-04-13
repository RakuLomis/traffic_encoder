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
                num_flow_features: int = 0, use_flow_features: bool = False, 
                hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        # self.field_embedder = FieldEmbedding(config_path, vocab_path)
        self.field_embedder = field_embedder
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.node_fields = node_fields_list # 接收从Dataset传来的、当前Block的节点列表
        self.use_flow_features = use_flow_features
        
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

        if self.use_flow_features:
            # --- 如果“开关”打开 ---
            if num_flow_features <= 0:
                raise ValueError("num_flow_features 必须大于0，当 use_flow_features=True 时。")
            
            # a) 创建“流特征”嵌入器
            flow_embed_dim = hidden_dim // 2 # 64
            self.flow_stats_embedder = nn.Sequential(
                nn.Linear(num_flow_features, 64),
                nn.LeakyReLU(),
                nn.Linear(64, flow_embed_dim)
            )
            # b) 计算融合后的维度
            combined_dim = hidden_dim + flow_embed_dim
            
        else:
            # --- 如果“开关”关闭 ---
            self.flow_stats_embedder = None
            # b) 维度保持不变
            combined_dim = hidden_dim

        # # ==================== 核心修改点 2：新增“流特征”嵌入器 ====================
        # # 这个MLP负责将流统计特征 (例如 [avg_len, std_len, pkt_count]) 
        # # 编码到一个与GNN输出兼容的维度 (例如 64)
        # flow_embed_dim = hidden_dim // 2 # 这是一个可调的超参数，64维是一个好起点
        # self.flow_stats_embedder = nn.Sequential(
        #     nn.Linear(num_flow_features, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, flow_embed_dim)
        # )
        # # ========================================================================

        # # ==================== 核心修改点 3：修改“分类器”以接收融合后的特征 ====================
        # # 
        # # 我们的最终特征向量将是 GNN输出 和 流特征嵌入 拼接而成的
        # # GNN输出维度: hidden_dim
        # # 流特征输出维度: flow_embed_dim
        # # 总输入维度: hidden_dim + flow_embed_dim
        # #
        # combined_dim = hidden_dim + flow_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), # 第一个线性层接收“加宽”的特征
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes) # 第二个线性层输出最终logits
        )

        # self.classifier = nn.Sequential(
        #     # 第一个线性层
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     # “稳定器”：批量归一化
        #     # nn.BatchNorm1d(hidden_dim),
        #     # “防死亡”激活函数
        #     nn.LeakyReLU(),
        #     # Dropout层
        #     nn.Dropout(p=dropout_rate),
        #     # 第二个线性层，输出最终的logits
        #     nn.Linear(hidden_dim // 2, num_classes)
        # )
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


    def forward(self, data) -> torch.Tensor: 
        # data 是一个由PyG DataLoader准备好的批处理图对象
        
        # --- a) 步骤一：初始嵌入 (无变化) ---
        batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
        embedded_vectors = self.field_embedder(batch_dict)
        
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

        if self.use_flow_features:
            # --- 如果“开关”打开，则执行融合 ---
            if not hasattr(data, 'flow_stats'):
                raise ValueError("模型处于 use_flow_features=True 模式, 但GNNTrafficDataset未提供 'data.flow_stats'。")
                
            flow_stats_input = data.flow_stats
            flow_stats_embedding = self.flow_stats_embedder(flow_stats_input)
            
            final_features = torch.cat([graph_embedding, flow_stats_embedding], dim=1)
        
        else:
            # --- 如果“开关”关闭，则直接使用GNN的输出 ---
            final_features = graph_embedding

        # # ==================== 核心修改点 4：嵌入流特征并融合 ====================
        
        # # --- b) 步骤二：流级别“上下文”特征提取 ---
        # #    data.flow_stats 是由GNNTrafficDataset.__getitem__提供的张量
        # #    形状: [batch_size, num_flow_features]
        # flow_stats_input = data.flow_stats
        
        # # flow_stats_embedding 是从【流上下文】中学到的“文章风格”
        # # 形状: [batch_size, flow_embed_dim]
        # flow_stats_embedding = self.flow_stats_embedder(flow_stats_input)
        
        # # --- c) 步骤三：特征融合 ---
        # # 将“笔迹”和“文章风格”两种信息，在特征维度上拼接在一起
        # # 形状: [batch_size, hidden_dim + flow_embed_dim]
        # combined_features = torch.cat([graph_embedding, flow_stats_embedding], dim=1)
        
        # =======================================================================

        # logits = self.classifier(graph_embedding)
        logits = self.classifier(final_features)
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

