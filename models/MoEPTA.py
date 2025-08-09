import torch 
import torch.nn as nn
from typing import Dict, List, Tuple 
from models.FieldEmbedding import FieldEmbedding 
from models.ProtocolTreeAttention import ProtocolTreeAttention, AttentionAggregator 
from utils.dataframe_tools import protocol_tree
import pandas as pd
import os 
from tqdm import tqdm 

class MoEPTA(nn.Module): 
    def __init__(self, block_directory: str, config_path: str, vocab_path: str, eligible_blocks: list, 
                 block_num_classes: dict, num_classes: int, final_aggregator_dim: int=64, num_heads: int=4):
        super().__init__() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} in MoE")
        self.field_embedder = FieldEmbedding(config_path, vocab_path).to(self.device)
        self.experts = nn.ModuleDict() 

        print("Building expert for each block. ") 
        # block_csv_files = sorted([f for f in os.listdir(block_directory) if f.endswith('.csv')]) 

        self.expert_output_dims = {}

        for block_filename in tqdm(eligible_blocks, desc="Initializing Experts"):
            block_name = os.path.splitext(block_filename)[0]
            block_path = os.path.join(block_directory, block_filename)
            
            # 读取该Block的CSV以获取其独特的字段结构
            block_df = pd.read_csv(block_path, dtype=str, nrows=0) # 只读表头
            block_ptree = protocol_tree(block_df.columns.tolist())
            local_num_classes = block_num_classes[block_name]


            # 为这个Block创建一个专属的PTA模型实例
            # 注意：这个PTA模型现在不包含最后的分类器
            expert_model = ProtocolTreeAttention(
                field_embedder=self.field_embedder,
                # config_path=config_path, 
                # vocab_path=vocab_path, 
                protocol_tree=block_ptree,
                num_classes=local_num_classes # 暂时保留，但我们只用它倒数第二层的输出
            )
            expert_model.to(self.device)
            
            expert_key = f"expert_{block_name}"
            self.experts[expert_key] = expert_model
            # 记录这个专家输出向量的维度（即PTA模型中aligned_dim）
            self.expert_output_dims[expert_key] = expert_model.aligned_dim

        # --- 3. 创建对齐层和最终的决策者 ---
        
        # a) 创建线性层，将所有专家（可能维度不同）的输出对齐到统一维度
        self.aligners = nn.ModuleDict()
        for expert_key, output_dim in self.expert_output_dims.items():
            self.aligners[expert_key] = nn.Linear(output_dim, final_aggregator_dim)
            
        # b) 创建最终的注意力聚合器，作为“最终决策者”
        self.final_attention_combiner = AttentionAggregator(final_aggregator_dim, num_heads)
        
        # c) 创建最终的分类器
        self.classifier = nn.Linear(final_aggregator_dim, num_classes)

    def forward(self, batch_of_blocks: Dict[str, Dict]) -> torch.Tensor:
        """
        MoE_PTA的前向传播。

        :param batch_of_blocks: 一个字典，键是专家名称(e.g., 'expert_0')，
                                值是对应专家需要处理的批处理数据字典。
        :return: 最终的分类logits。
        """
        device = self.device
        expert_outputs = []
        
        # 1. 让每个专家处理其对应的数据块
        for expert_key, expert_model in self.experts.items():
            block_name = expert_key.replace('expert_', '')
            
            # 检查当前批次中是否有这个Block的数据
            if block_name in batch_of_blocks:
                block_data = batch_of_blocks[block_name]
                
                # 获取PTA模型在分类器之前的输出
                # 我们需要修改PTA模型，让它能返回中间向量
                packet_vector = expert_model.forward_features(block_data, device=device) # 假设有这个方法
                
                # 2. 将每个专家的输出对齐到统一维度
                aligner = self.aligners[expert_key]
                aligned_vector = aligner(packet_vector)
                
                expert_outputs.append(aligned_vector)
        
        if not expert_outputs:
            # 如果这个批次不包含任何此模型能处理的Block，则返回零
            # 需要从输入中获取batch_size
            any_batch = next(iter(batch_of_blocks.values()))
            any_tensor = next(iter(any_batch.values()))
            batch_size = any_tensor.shape[0]
            return torch.zeros(batch_size, self.classifier.out_features, device=self.classifier.weight.device)
            
        # 3. 将所有对齐后的专家意见堆叠起来
        # (num_experts, batch_size, final_dim) -> (batch_size, num_experts, final_dim)
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        
        # 4. 让“最终决策者”对所有意见进行注意力聚合
        final_packet_representation = self.final_attention_combiner(stacked_outputs)
        
        # 5. 得出最终分类结果
        logits = self.classifier(final_packet_representation)
        
        return logits