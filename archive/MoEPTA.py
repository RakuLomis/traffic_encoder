import torch 
import torch.nn as nn
from typing import Dict, List, Tuple 
from models.FieldEmbedding import FieldEmbedding 
from models.ProtocolTreeAttention import ProtocolTreeAttention, AttentionAggregator 
from utils.dataframe_tools import protocol_tree
import pandas as pd
import os 
from collections import Counter
from tqdm import tqdm 

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

class RoutingEmbedder(nn.Module):
    """
    一个专门的模块，用于将通用的路由特征嵌入成一个固定维度的路由向量。
    (已修正版本)
    """
    def __init__(self, field_embedder, routing_fields: List[str], routing_dim: int):
        super().__init__()
        self.routing_fields = routing_fields
        self.embedders = nn.ModuleDict()
        total_input_dim = 0
        
        for field in self.routing_fields:
            layer_key = field.replace('.', '__')
            if layer_key in field_embedder.embedding_layers:
                embedding_layer = field_embedder.embedding_layers[layer_key]
                self.embedders[layer_key] = embedding_layer
                
                # ==================== 核心修改点 开始 ====================
                # 智能地判断层的类型，并从正确的属性获取输出维度
                
                output_dim = 0
                if isinstance(embedding_layer, nn.Embedding):
                    output_dim = embedding_layer.embedding_dim
                elif isinstance(embedding_layer, nn.Linear):
                    output_dim = embedding_layer.out_features
                # 假设您的_AddressEmbedding类有一个output_dim属性
                # elif isinstance(embedding_layer, _AddressEmbedding):
                #     output_dim = embedding_layer.output_dim
                
                total_input_dim += output_dim
                # ==================== 核心修改点 结束 ====================

        if total_input_dim == 0:
            raise ValueError("RoutingEmbedder did not find any valid routing fields to embed.")
            
        self.projector = nn.Linear(total_input_dim, routing_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # ... (forward方法与之前版本完全相同，无需修改) ...
        routing_embs = []
        for field in self.routing_fields:
            layer_key = field.replace('.', '__')
            if field in features and layer_key in self.embedders:
                inp = features[field]
                layer = self.embedders[layer_key]
                if isinstance(layer, nn.Linear):
                    inp = inp.view(-1, 1).float()
                routing_embs.append(layer(inp))
        
        if not routing_embs:
            # 如果批次中不包含任何路由特征，返回一个零向量
            batch_size = next(iter(features.values())).shape[0]
            return torch.zeros(batch_size, self.projector.out_features, device=self.projector.weight.device)

        concatenated_embs = torch.cat(routing_embs, dim=1)
        return self.projector(concatenated_embs)
    
def find_common_routing_fields(block_directory: str, eligible_blocks: list, top_k: int = 10):
    """
    扫描所有合格的Block，找出最常出现的、可作为路由依据的通用字段。

    :param block_directory: 包含所有Block CSV文件的目录。
    :param eligible_blocks: 合格的Block文件名列表。
    :param top_k: 选择出现频率最高的k个字段。
    :return: 一个包含top_k个最通用字段的列表。
    """
    print("正在自动寻找最佳的通用路由特征...")
    field_counter = Counter()
    
    for block_filename in eligible_blocks:
        block_path = os.path.join(block_directory, block_filename)
        # 只读表头，速度很快
        df_columns = pd.read_csv(block_path, dtype=str, nrows=0).columns.tolist()
        
        # 排除元数据
        feature_columns = [f for f in df_columns if f not in ['index', 'label', 'label_id']]
        field_counter.update(feature_columns)
        
    # most_common(k)会返回一个 [(字段, 出现次数), ...] 的列表
    # 我们只取字段名
    common_fields = [field for field, count in field_counter.most_common(top_k)]
    
    print(f"找到的最通用的 {top_k} 个路由特征是: {common_fields}")
    return common_fields

# class MoEPTA(nn.Module):
#     def __init__(self, block_directory: str, config_path: str, vocab_path: str, eligible_blocks: list, 
#                  num_classes: int, routing_input_dim: int = 256): # routing_input_dim需要根据路由特征确定
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device} in MoE")
        
#         self.field_embedder = FieldEmbedding(config_path, vocab_path)
        
#         # 专家网络
#         self.experts = nn.ModuleDict()
#         self.expert_keys = []
#         print("Building expert for each block.")
#         for block_filename in tqdm(eligible_blocks, desc="Initializing Experts"):
#             block_name = os.path.splitext(block_filename)[0]
#             block_path = os.path.join(block_directory, block_filename)
#             block_df = pd.read_csv(block_path, dtype=str, nrows=0)
#             block_ptree = protocol_tree(block_df.columns.tolist())
            
#             # 对齐
#             expert_model = ProtocolTreeAttention(
#                 field_embedder=self.field_embedder,
#                 protocol_tree=block_ptree,
#                 num_classes=num_classes 
#             )
#             expert_key = f"expert_{block_name}"
#             self.experts[expert_key] = expert_model
#             self.expert_keys.append(expert_key)
            
#         self.gating_network = GatingNetwork(input_dim=routing_input_dim, num_experts=len(self.experts))

#     def forward(self, batch_of_blocks: Dict[str, Dict]) -> torch.Tensor:
#         block_name = next(iter(batch_of_blocks.keys()))
#         features = batch_of_blocks[block_name]

#         routing_input = torch.cat([
#             features.get('ip.len', torch.zeros_like(features[next(iter(features))])).float().view(-1, 1),
#             features.get('tcp.len', torch.zeros_like(features[next(iter(features))])).float().view(-1, 1)
#         ], dim=1)

        
#         batch_size = next(iter(features.values())).shape[0]
#         routing_vector = torch.randn(batch_size, 256).to(self.device)
        

#         gating_weights = self.gating_network(routing_vector)
#         # -> (batch_size, num_experts)


#         expert_key = f"expert_{block_name}"
#         expert_model = self.experts[expert_key]
        
#         expert_logits = expert_model(features)
        
#         expert_idx = self.expert_keys.index(expert_key)
        

#         weight_for_this_expert = gating_weights[:, expert_idx].unsqueeze(1)

#         final_output = expert_logits * weight_for_this_expert
        
#         return final_output

class MoEPTA(nn.Module):
    def __init__(self, block_directory: str, config_path: str, vocab_path: str, eligible_blocks: list, 
                 num_classes: int, routing_fields: List[str], routing_input_dim: int = 256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 共享的FieldEmbedding模块，现在在内部创建以确保设备正确
        self.field_embedder = FieldEmbedding(config_path, vocab_path)
        
        # --- 1. 创建专家网络 ---
        self.experts = nn.ModuleDict()
        self.expert_keys = [] # 保持专家顺序

        for block_filename in tqdm(eligible_blocks, desc="Initializing Experts"):
            block_name = os.path.splitext(block_filename)[0]
            block_path = os.path.join(block_directory, block_filename)
            block_df = pd.read_csv(block_path, dtype=str, nrows=0)
            block_ptree = protocol_tree(block_df.columns.tolist())
            
            # 对齐
            expert_model = ProtocolTreeAttention(
                field_embedder=self.field_embedder,
                protocol_tree=block_ptree,
                num_classes=num_classes 
            )
            expert_key = f"expert_{block_name}"
            self.experts[expert_key] = expert_model
            self.expert_keys.append(expert_key)
            
        # --- 2. 创建路由和门控网络 ---
        self.routing_embedder = RoutingEmbedder(self.field_embedder, routing_fields, routing_input_dim)
        self.gating_network = GatingNetwork(input_dim=routing_input_dim, num_experts=len(self.experts))

        # --- 3. 最终的分类器 ---
        # 假设所有专家的输出都是logits，维度为num_classes
        self.num_classes = num_classes

    def forward(self, batch_of_blocks: Dict[str, Dict]) -> torch.Tensor:
        # --- a) 准备门控网络的输入 ---
        # 假设我们总是能从输入中找到路由特征
        # 注意: 这里的'batch_of_blocks'现在包含了多个Block的数据
        
        # 从所有可用的Block数据中提取路由特征
        # 我们需要一种方法来整合来自不同Block的路由特征，这里简化处理
        # 假设我们使用第一个可用Block的特征作为路由依据
        first_block_name = next(iter(batch_of_blocks.keys()))
        routing_features = batch_of_blocks[first_block_name]
        
        routing_vector = self.routing_embedder(routing_features)
        gating_weights = self.gating_network(routing_vector) # (batch_size, num_experts)

        # --- b) 并行执行所有专家 ---
        expert_outputs = torch.zeros(routing_vector.shape[0], len(self.experts), self.num_classes).to(self.device)

        # 遍历批次中【当前存在】的专家
        for block_name, features in batch_of_blocks.items():
            expert_key = f"expert_{block_name}"
            if expert_key in self.experts:
                expert_model = self.experts[expert_key]
                expert_idx = self.expert_keys.index(expert_key)
                
                # 专家输出logits
                expert_logits = expert_model(features)
                expert_outputs[:, expert_idx, :] = expert_logits

        # --- c) 加权求和所有专家的意见 ---
        # gating_weights: (B, E) -> (B, E, 1)
        # expert_outputs: (B, E, C)
        # 结果: (B, C)
        final_logits = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        # 还可以返回门控权重，以供分析或计算辅助损失
        return final_logits #, gating_weights