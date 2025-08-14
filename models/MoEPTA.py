import torch 
import torch.nn as nn
from typing import Dict, List, Tuple 
from models.FieldEmbedding import FieldEmbedding 
from models.ProtocolTreeAttention import ProtocolTreeAttention, AttentionAggregator 
from utils.dataframe_tools import protocol_tree
import pandas as pd
import os 
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

class MoEPTA(nn.Module):
    def __init__(self, block_directory: str, config_path: str, vocab_path: str, eligible_blocks: list, 
                 num_classes: int, routing_input_dim: int = 256): # routing_input_dim需要根据路由特征确定
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} in MoE")
        
        self.field_embedder = FieldEmbedding(config_path, vocab_path)
        
        # 专家网络
        self.experts = nn.ModuleDict()
        self.expert_keys = []
        print("Building expert for each block.")
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
            
        self.gating_network = GatingNetwork(input_dim=routing_input_dim, num_experts=len(self.experts))

    def forward(self, batch_of_blocks: Dict[str, Dict]) -> torch.Tensor:
        block_name = next(iter(batch_of_blocks.keys()))
        features = batch_of_blocks[block_name]

        routing_input = torch.cat([
            features.get('ip.len', torch.zeros_like(features[next(iter(features))])).float().view(-1, 1),
            features.get('tcp.len', torch.zeros_like(features[next(iter(features))])).float().view(-1, 1)
        ], dim=1)

        
        batch_size = next(iter(features.values())).shape[0]
        routing_vector = torch.randn(batch_size, 256).to(self.device)
        

        gating_weights = self.gating_network(routing_vector)
        # -> (batch_size, num_experts)


        expert_key = f"expert_{block_name}"
        expert_model = self.experts[expert_key]
        
        expert_logits = expert_model(features)
        
        expert_idx = self.expert_keys.index(expert_key)
        

        weight_for_this_expert = gating_weights[:, expert_idx].unsqueeze(1)

        final_output = expert_logits * weight_for_this_expert
        
        return final_output
