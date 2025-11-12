import yaml
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from typing import Dict

class _AddressEmbedding(nn.Module):
    """
    一个内部辅助模块，专门用于处理地址类型（IPv4, MAC）的嵌入。
    它将地址拆分为字节，对每个字节进行嵌入，然后通过聚合层（如CNN）学习地址的整体表示。
    """
    def __init__(self, num_octets, embedding_dim_per_octet, aggregation='cnn'):
        super().__init__()
        self.num_octets = num_octets
        # 为每个字节（octet）位置创建一个独立的嵌入层
        # 使用 ModuleList 来正确注册这些层
        self.octet_embedders = nn.ModuleList([
            nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim_per_octet)
            for _ in range(num_octets)
        ])
        
        self.aggregation = aggregation
        if self.aggregation == 'cnn':
            # 使用一维卷积在字节嵌入序列上学习局部模式
            self.agg_layer = nn.Conv1d(
                in_channels=embedding_dim_per_octet,
                out_channels=embedding_dim_per_octet, # 可以调整输出通道数
                kernel_size=3,
                padding=1
            )
        # 可以根据需要添加 'rnn' 等其他聚合方式
        # elif self.aggregation == 'rnn':
        #     ...

    def forward(self, x):
        # 输入 x 的形状应为 (batch_size, num_octets)
        # 例如对于IPv4，是 (batch_size, 4)
        
        embedded_octets = []
        for i in range(self.num_octets):
            # 对每个字节位置应用对应的嵌入层
            octet_tensor = x[:, i]
            embedded_octet = self.octet_embedders[i](octet_tensor)
            embedded_octets.append(embedded_octet)

        # 将嵌入后的字节向量堆叠起来
        # 形状变为 (batch_size, num_octets, embedding_dim_per_octet)
        x = torch.stack(embedded_octets, dim=1)
        
        if self.aggregation == 'cnn':
            # CNN希望输入是 (N, C_in, L_in)
            # 所以需要交换维度 (batch_size, embedding_dim_per_octet, num_octets)
            x = x.permute(0, 2, 1)
            x = self.agg_layer(x)
            # 将卷积后的结果进行全局平均池化，得到一个固定大小的向量
            # 输出形状 (batch_size, embedding_dim_per_octet)
            x = x.mean(dim=2)
        elif self.aggregation == 'concat':
            # 直接拼接，输出形状 (batch_size, num_octets * embedding_dim_per_octet)
            x = x.view(x.size(0), -1)
        # 默认使用求和
        else: # 'sum' or other fallbacks
            # 输出形状 (batch_size, embedding_dim_per_octet)
            x = x.sum(dim=1)
            
        return x


class FieldEmbedding(nn.Module):
    """
    主嵌入模块。
    根据YAML配置文件，为数据帧中的所有字段创建合适的嵌入层。
    """
    def __init__(self, config_path, vocab_path):
        super().__init__()
        
        # 1. 加载YAML配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config'] 

        with open(vocab_path, 'r') as f: 
            self.vocab_maps = yaml.safe_load(f) 
            
        self.embedding_layers = nn.ModuleDict() # 使用ModuleDict来存储层，它能正确注册并允许使用字符串键
        self.total_embedding_dim = 0 

        # slices is used to record the duration reflection of fields before and after embedding
        self.embedding_slices = {}
        sorted_field_names = sorted(self.config.keys()) 
        # # Replace '.' by '__'
        self.field_to_key_map = {name: name.replace('.', '__') for name in sorted_field_names}

        for field_name in tqdm(self.field_to_key_map, desc="Initing embedding layer and constructing tensor maps... "):
            field_config = self.config[field_name]
            
            # 将'.'替换为'__'以作为合法的模块键名
            layer_key = field_name.replace('.', '__')
            # layer_key = field_name
            field_type = field_config['type']
            
            layer = None
            output_dim = 0

            if field_type == 'categorical':
                embedding_dim = field_config['embedding_dim']
                # 从加载的词典中动态计算 vocab_size
                if field_name in self.vocab_maps:
                    # # vocab_size 就是该字段对应字典的长度
                    # vocab_size = len(self.vocab_maps[field_name])
                    # layer = nn.Embedding(vocab_size, embedding_dim)
                    # output_dim = embedding_dim
                    vocab_map = self.vocab_maps[field_name]
                    # --- [!! 关键修复 !!] ---

                    # [错误逻辑] vocab_size = len(vocab_map) 

                    # [正确逻辑]
                    if not vocab_map: # 安全检查，防止词汇表为空
                        print(f"\n警告：分类字段 '{field_name}' 的词汇表为空，将跳过。")
                        continue

                    # 1. 找到词汇表中所有索引（值）中的最大值
                    max_index = max(vocab_map.values())

                    # 2. 嵌入层的大小必须是 (最大索引 + 1)
                    vocab_size = max_index + 1
                    # --- [!! 修复结束 !!] --- 

                    # 假设你的 __OOV__ 索引是 0，你可以将 0 设为 padding_idx
                    # 这可以让模型学会“忽略”OOV值，并可能提高性能
                    padding_idx = 0 if 0 in vocab_map.values() else None                    
                    layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
                    output_dim = embedding_dim
                else:
                    # 如果一个分类字段在词典文件中找不到，打印警告并跳过
                    print(f"\n警告：分类字段 '{field_name}' 在词典文件 '{vocab_path}' 中未找到，将跳过此字段。")
                    continue # 跳过当前循环的剩余部分
            # 【!! 核心修改：添加新分支 !!】
            elif field_type == 'port_binned':
                # 这是我们的新类型
                embedding_dim = field_config['embedding_dim']
                # 我们的分箱逻辑产生 4 个 bin (0, 1, 2, 3)
                vocab_size = 4
                # Bin 0 (未知) 可以作为 padding_idx
                layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                output_dim = embedding_dim
            # --- [!! 修复结束 !!] ---
                
            elif field_type == 'numerical':
                embedding_dim = field_config['embedding_dim']
                layer = nn.Linear(1, embedding_dim)
                output_dim = embedding_dim

            elif field_type == 'address_ipv4':
                embedding_dim = field_config['embedding_dim_per_octet']
                layer = _AddressEmbedding(4, embedding_dim, field_config['aggregation'])
                output_dim = embedding_dim # 假设聚合后输出维度与embedding_dim_per_octet相同

            elif field_type == 'address_mac':
                embedding_dim = field_config['embedding_dim_per_octet']
                layer = _AddressEmbedding(6, embedding_dim, field_config['aggregation'])
                output_dim = embedding_dim 

            # ==================== CORE FIX ====================
            # Fix the ghost features, whose dim is 0 and cannot be used to forward. 
            # Add a check to prevent zero-dimension features from being added.
            if output_dim == 0 and layer is not None:
                print(f"\nWarning: Field '{field_name}' has an embedding_dim of 0. "
                      f"This feature will be completely ignored by the model.")
                continue # Skip to the next field
            # ================================================
            
            if layer is not None:
                self.embedding_layers[layer_key] = layer
                
                # --- 核心修改点 2：记录每个特征的切片位置 ---
                start_index = self.total_embedding_dim
                end_index = start_index + output_dim
                self.embedding_slices[field_name] = (start_index, end_index)
                
                # 更新总维度
                self.total_embedding_dim += output_dim
                # --------------------------------------------        

    def forward(self, batch_data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        此版本不再返回一个拼接的“大向量”，而是返回一个包含各个字段嵌入向量的字典。

        :param batch_data_dict: 输入的原始数据字典。
        :return: 一个字典，键是字段名，值是对应的嵌入向量张量。
        """
        embedded_vectors_dict = {}
        # 遍历输入批次中的字段
        for field_name, input_tensor in batch_data_dict.items():
            layer_key = self.field_to_key_map.get(field_name)

            # 只有当该字段在我们的配置中有对应的嵌入层时，才处理它
            if layer_key and layer_key in self.embedding_layers:
                layer = self.embedding_layers[layer_key]
                
                # --- （这部分逻辑与之前完全相同）---
                if isinstance(layer, _AddressEmbedding) and isinstance(input_tensor, list):
                    # input_tensor = torch.stack(input_tensor, dim=1)
                    target_device = next(layer.parameters()).device
                    input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=target_device)
                
                if isinstance(layer, nn.Linear):
                    input_tensor = input_tensor.view(-1, 1).float()
                # --- END FIX ---
                
                # --- 核心逻辑 ---
                # 计算嵌入向量，并将其存入输出字典
                embedded_vectors_dict[field_name] = layer(input_tensor)

        return embedded_vectors_dict