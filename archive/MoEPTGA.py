import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os
from typing import Dict, List

# 导入您已经定义好的模块
from models.FieldEmbedding import FieldEmbedding
from models.ProtocolTreeGAttention import ProtocolTreeGAttention
# from utils.dataframe_tools import generate_protocol_tree_and_nodes
from utils.dataframe_tools import protocol_tree, add_root_layer

class MoE_PTGA(nn.Module):
    def __init__(self, block_directory: str, config_path: str, vocab_path: str, 
                 eligible_blocks: List[str], num_classes: int, 
                 hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        # --- 1. 创建【唯一】的、将被所有专家【共享】的FieldEmbedding模块 ---
        print("Initializing Shared FieldEmbedding Module...")
        self.shared_field_embedder = FieldEmbedding(config_path, vocab_path)
        
        # --- 2. 创建专家“工具箱” ---
        self.experts = nn.ModuleDict()
        print(f"Building {len(eligible_blocks)} expert models...")
        for block_filename in tqdm(eligible_blocks, desc="Initializing Experts"):
            block_name = os.path.splitext(block_filename)[0]
            
            # a) 为每个专家动态确定其节点列表
            block_df = pd.read_csv(os.path.join(block_directory, block_filename), dtype=str, nrows=0)
            
            available_fields = set(block_df.columns)
            configured_fields = set(self.shared_field_embedder.config.keys())
            real_nodes = sorted(list(available_fields.intersection(configured_fields)))
            # _, all_nodes_in_tree = generate_protocol_tree_and_nodes(real_nodes)
            ptree = protocol_tree(real_nodes)
            add_root_layer(ptree) # Now ptree contains root and different layers
            all_nodes_in_tree = set(ptree.keys()) 
            for children in self.ptree.values(): 
                all_nodes_in_tree.update(children)
            node_fields_for_expert = sorted(list(all_nodes_in_tree))
            
            # b) 创建专家实例，并将【共享的】field_embedder“注入”进去
            expert_model = ProtocolTreeGAttention(
                field_embedder=self.shared_field_embedder, # <-- 注入共享实例
                num_classes=num_classes,
                node_field_list=node_fields_for_expert,
                hidden_dim=hidden_dim, 
                num_heads=num_heads, 
                dropout_rate=dropout_rate
            )
            self.experts[f"expert_{block_name}"] = expert_model
            
    def forward(self, batched_graph: Dict, block_name: str) -> torch.Tensor:
        """
        一个确定性路由的前向传播。
        它只执行与传入的block_name对应的那个专家。

        :param batched_graph: 由对应Block的DataLoader准备好的批处理图对象。
        :param block_name: 明确指定要使用哪个专家的名称 (不带'.csv'后缀)。
        :return: 该专家的输出logits。
        """
        expert_key = f"expert_{block_name}"
        if expert_key not in self.experts:
            # 这是一个安全检查，理论上不应该发生
            raise ValueError(f"Unknown expert requested: {expert_key}")
            
        expert_to_run = self.experts[expert_key]
        
        return expert_to_run(batched_graph)