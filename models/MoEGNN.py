import torch
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict
from models.FieldEmbedding import FieldEmbedding
from models.ProtocolTreeGAttention import ProtocolTreeGAttention 
import os
from tqdm import tqdm

class RoutingEmbedder(nn.Module):
    """
    一个专门的模块，用于将通用的路由特征嵌入成一个固定维度的路由向量。
    """
    def __init__(self, field_embedder: FieldEmbedding, routing_fields: List[str], routing_dim: int):
        super().__init__()
        self.routing_fields = routing_fields
        self.embedders = nn.ModuleDict()
        total_input_dim = 0
        
        # 从主嵌入器中“借用”路由特征对应的嵌入层
        for field in self.routing_fields:
            if field in field_embedder.embedding_slices:
                layer_key = field.replace('.', '__')
                self.embedders[layer_key] = field_embedder.embedding_layers[layer_key]
                start, end = field_embedder.embedding_slices[field]
                total_input_dim += (end - start)
        
        self.projector = nn.Linear(total_input_dim, routing_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        routing_embs = []
        for field in self.routing_fields:
            if field in features:
                layer_key = field.replace('.', '__')
                inp = features[field]
                layer = self.embedders[layer_key]
                # 对数值型和地址型特征进行必要的形状调整
                if isinstance(layer, nn.Linear):
                    inp = inp.view(-1, 1).float()
                routing_embs.append(layer(inp))
        
        concatenated_embs = torch.cat(routing_embs, dim=1)
        return self.projector(concatenated_embs)

class GatingNetwork(nn.Module):
    """
    一个简单的门控网络。
    它接收一个通用的输入向量，并为所有专家输出一个权重分布。
    """
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
    

class MoEPTGA(nn.Module):
    def __init__(self, block_directory: str, config_path: str, vocab_path: str, eligible_blocks: list, 
                 num_classes: int, routing_fields: List[str], routing_dim: int = 256, 
                 hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        # 共享的FieldEmbedding模块，现在在内部创建以确保设备正确
        # 注意：PTGA专家自己也会创建，这里是为了RoutingEmbedder
        shared_field_embedder = FieldEmbedding(config_path, vocab_path)
        
        # --- 1. 创建路由和门控网络 ---
        self.routing_embedder = RoutingEmbedder(shared_field_embedder, routing_fields, routing_dim)
        self.num_experts = len(eligible_blocks)
        self.gating_network = GatingNetwork(input_dim=routing_dim, num_experts=self.num_experts)
        
        # --- 2. 创建专家网络 ---
        self.experts = nn.ModuleDict()
        self.expert_keys = [] # 保持专家顺序
        print("Building expert for each block...")
        for block_filename in tqdm(eligible_blocks, desc="Initializing Experts"):
            block_name = os.path.splitext(block_filename)[0]
            
            # PTA模型现在作为专家
            expert_model = ProtocolTreeGAttention(
                config_path=config_path,
                vocab_path=vocab_path,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
            expert_key = f"expert_{block_name}"
            self.experts[expert_key] = expert_model
            self.expert_keys.append(expert_key)
            
    def forward(self, batch_of_blocks: Dict[str, Dict]) -> torch.Tensor:
        # --- a) 准备门控网络的输入 ---
        # 从所有可用的Block数据中提取路由特征
        # 我们需要一个方法来整合来自不同Block的路由特征
        # 一个健壮的方法是要求 collate_fn 准备一个专门的路由特征批次
        routing_features = batch_of_blocks.get("routing_features")
        if routing_features is None:
            # 简化的备用方案：使用第一个可用Block的特征
            first_block_data = next(iter(batch_of_blocks.values()))
            routing_features = {k: v for k, v in first_block_data if k in self.routing_embedder.routing_fields}

        routing_vector = self.routing_embedder(routing_features)
        gating_weights = self.gating_network(routing_vector) # (total_batch_size, num_experts)

        # --- b) 并行执行所有专家，并进行加权求和 ---
        # 我们需要根据路由向量的来源，将权重和专家输出对齐
        # 这是一个非常复杂的“专家输出与权重对齐”问题
        
        # 为了给您一个能运行且体现核心思想的版本，我们在此简化
        # 假设批次中只有一个Block，这与您当前的串行训练循环匹配
        block_name = next(iter(batch_of_blocks.keys()))
        expert_key = f"expert_{block_name}"
        expert_model = self.experts[expert_key]
        expert_idx = self.expert_keys.index(expert_key)
        
        expert_logits = expert_model(batch_of_blocks[block_name])
        weight_for_this_expert = gating_weights[:, expert_idx].unsqueeze(1)
        
        final_logits = expert_logits * weight_for_this_expert
        
        # 同时返回门控权重，以供后续计算“负载均衡损失”
        return final_logits, gating_weights