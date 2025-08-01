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
                # layer = nn.Embedding(field_config['vocab_size'], embedding_dim)
                # output_dim = embedding_dim
                # ==================== 核心修改点 开始 ====================
                # 从加载的词典中动态计算 vocab_size
                if field_name in self.vocab_maps:
                    # vocab_size 就是该字段对应字典的长度
                    vocab_size = len(self.vocab_maps[field_name])
                    layer = nn.Embedding(vocab_size, embedding_dim)
                    output_dim = embedding_dim
                else:
                    # 如果一个分类字段在词典文件中找不到，打印警告并跳过
                    print(f"\n警告：分类字段 '{field_name}' 在词典文件 '{vocab_path}' 中未找到，将跳过此字段。")
                    continue # 跳过当前循环的剩余部分
                # ==================== 核心修改点 结束 ====================
                
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

        # # Replace '.' by '__'
        # self.field_to_key_map = {name: name.replace('.', '__') for name in self.config.keys()}

        # # 2. 遍历配置，动态创建嵌入层
        # for field_name, field_config in tqdm(self.config.items(), desc="Creating embedding layer. "):
        #     layer_key = self.field_to_key_map[field_name]
        #     field_type = field_config['type']
            
        #     layer = None
        #     output_dim = 0

        #     if field_type == 'categorical':
        #         vocab_size = field_config['vocab_size']
        #         embedding_dim = field_config['embedding_dim']
        #         layer = nn.Embedding(vocab_size, embedding_dim)
        #         output_dim = embedding_dim
                
        #     elif field_type == 'numerical':
        #         embedding_dim = field_config['embedding_dim']
        #         layer = nn.Linear(1, embedding_dim)
        #         output_dim = embedding_dim

        #     elif field_type == 'address_ipv4':
        #         embedding_dim = field_config['embedding_dim_per_octet']
        #         aggregation = field_config['aggregation']
        #         layer = _AddressEmbedding(4, embedding_dim, aggregation)
        #         output_dim = embedding_dim

        #     elif field_type == 'address_mac':
        #         embedding_dim = field_config['embedding_dim_per_octet']
        #         aggregation = field_config['aggregation']
        #         layer = _AddressEmbedding(6, embedding_dim, aggregation)
        #         output_dim = embedding_dim
            
        #     if layer is not None:
        #         self.embedding_layers[layer_key] = layer
        #         self.total_embedding_dim += output_dim
    
    # def forward(self, batch_data_dict):
    #     """
    #     前向传播。
    #     :param batch_data_dict: 一个字典，键是字段名，值是对应的批处理数据张量。
    #                             e.g., {'ip.src': tensor, 'tcp.port': tensor, ...}
    #     :return: 一个拼接了所有字段嵌入向量的大的特征张量。
    #     """
    #     embedded_outputs = []
        
    #     # 按照ModuleDict中层的顺序进行迭代，保证每次拼接的顺序一致
    #     for field_name, layer_key in tqdm(self.field_to_key_map.items(), desc="Forwarding. "):
    #         # 检查批处理数据中是否存在该字段
    #         if field_name in batch_data_dict and layer_key in self.embedding_layers:
    #             # 获取对应的数据张量
    #             input_tensor = batch_data_dict[field_name]
    #             layer = self.embedding_layers[layer_key] 

    #             # ==================== 调试代码块 开始 ====================
    #             # 在执行嵌入前，检查索引是否在有效范围内
    #             if isinstance(layer, nn.Embedding):
    #                 # 获取输入张量中的最大值
    #                 max_index_in_batch = input_tensor.max()
    #                 # 获取该层的词典大小
    #                 configured_vocab_size = layer.num_embeddings
                    
    #                 if max_index_in_batch >= configured_vocab_size:
    #                     print("\n" + "="*60)
    #                     print(f"!!! ERROR DETECTED: Index out of range for field: '{field_name}' !!!")
    #                     print(f"    Max index value found in your data: {max_index_in_batch}")
    #                     print(f"    Vocab size configured in your YAML: {configured_vocab_size}")
    #                     print(f"    (Remember: valid indices are from 0 to vocab_size - 1)")
    #                     print("    SOLUTION: Please increase the 'vocab_size' for this field in your f2v.yaml file.")
    #                     print("="*60 + "\n")
    #                     # 主动抛出错误，并附带清晰的说明
    #                     raise IndexError(f"For field '{field_name}', input index {max_index_in_batch} "
    #                                      f"is out of range for vocab_size {configured_vocab_size}.")
    #             # ==================== 调试代码块 结束 ====================

    #             # --- 新增的修正逻辑 ---
    #             # 检查输入是否为地址类型且是一个列表（default_collate的特殊情况）
    #             if isinstance(layer, _AddressEmbedding) and isinstance(input_tensor, list):
    #                 # 如果是，使用torch.stack进行重组
    #                 input_tensor = torch.stack(input_tensor, dim=1)
    #             # --- 修正逻辑结束 ---
                
    #             # 特别处理数值型输入，需要确保其形状为 (batch_size, 1)
    #             if isinstance(layer, nn.Linear):
    #                 input_tensor = input_tensor.view(-1, 1).float()

    #             embedded_vector = layer(input_tensor)
    #             embedded_outputs.append(embedded_vector)
    #         else:
    #             # 如果某个字段在批数据中缺失，可以考虑填充一个零向量或采取其他策略
    #             # 这里简单地打印一个警告
    #             print(f"Warning: Field '{field_name}' not found in the input batch data.")

    #     # 4. 沿最后一个维度拼接所有嵌入向量
    #     # 输出形状 (batch_size, total_embedding_dim)
    #     return torch.cat(embedded_outputs, dim=-1) 
    # ==================== 核心修改点：重构 forward 方法 ====================
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
                    input_tensor = torch.stack(input_tensor, dim=1)
                
                # if isinstance(layer, nn.Linear):
                #     input_tensor = input_tensor.view(-input_tensor.size(0), 1).float()
                # --- CORE FIX ---
                # For numerical features, reshape the tensor to (batch_size, 1)
                # The correct argument is -1, not -input_tensor.size(0)
                if isinstance(layer, nn.Linear):
                    input_tensor = input_tensor.view(-1, 1).float()
                # --- END FIX ---
                
                # --- 核心逻辑 ---
                # 计算嵌入向量，并将其存入输出字典
                embedded_vectors_dict[field_name] = layer(input_tensor)

        return embedded_vectors_dict

# # --- 使用示例 ---
# if __name__ == '__main__':
#     # 假设您的YAML文件与此脚本在同一目录下或在指定路径
#     # 为了演示，我们先创建一个临时的YAML文件
    
#     # 找到当前文件所在目录
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # 定位到项目根目录（假设models文件夹在根目录下）
#     project_root = os.path.dirname(current_dir)
#     # 构建到utils文件夹下f2v.yaml的路径
#     config_file_path = os.path.join(project_root, 'utils', 'f2v.yaml')

#     print(f"Loading config from: {config_file_path}")

#     # 1. 实例化嵌入模块
#     field_embedder = FieldEmbedding(config_path=config_file_path)
#     print("\nField Embedding Module instantiated:")
#     print(field_embedder)
#     print(f"\nTotal concatenated embedding dimension: {field_embedder.total_embedding_dim}")

#     # 2. 创建一批虚拟数据 (batch_size = 4)
#     # 注意：真实数据需要经过预处理，将类别转为整数索引，地址转为字节列表等
#     batch_size = 4
#     dummy_batch = {
#         'eth.dst': torch.randint(0, 256, (batch_size, 6)),  # MAC地址，6个字节
#         'eth.dst.lg': torch.randint(0, 3, (batch_size,)),
#         'eth.dst.ig': torch.randint(0, 3, (batch_size,)),
#         'eth.src': torch.randint(0, 256, (batch_size, 6)),
#         'eth.src.lg': torch.randint(0, 3, (batch_size,)),
#         'eth.src.ig': torch.randint(0, 2, (batch_size,)),
#         'eth.type': torch.randint(0, 3, (batch_size,)),
#         'ip.version': torch.randint(0, 3, (batch_size,)),
#         'ip.hdr_len': torch.rand(batch_size) * 20, # 数值型
#         'ip.dsfield': torch.randint(0, 6, (batch_size,)),
#         'ip.dsfield.dscp': torch.randint(0, 6, (batch_size,)),
#         'ip.dsfield.ecn': torch.randint(0, 3, (batch_size,)),
#         'ip.len': torch.rand(batch_size) * 1500,
#         'ip.id': torch.rand(batch_size),
#         'ip.flags': torch.randint(0, 8, (batch_size,)),
#         'ip.flags.rb': torch.randint(0, 2, (batch_size,)),
#         'ip.flags.df': torch.randint(0, 2, (batch_size,)),
#         'ip.flags.mf': torch.randint(0, 2, (batch_size,)),
#         'ip.frag_offset': torch.rand(batch_size),
#         'ip.ttl': torch.rand(batch_size) * 128,
#         'ip.proto': torch.randint(0, 3, (batch_size,)),
#         'ip.checksum': torch.rand(batch_size),
#         'ip.src': torch.randint(0, 256, (batch_size, 4)), # IPv4地址，4个字节
#         'ip.dst': torch.randint(0, 256, (batch_size, 4)),
#         'tcp.srcport': torch.randint(0, 65536, (batch_size,)),
#         'tcp.dstport': torch.randint(0, 65536, (batch_size,)),
#         'tcp.stream': torch.randint(0, 2048, (batch_size,)),
#         'tcp.len': torch.rand(batch_size) * 1460,
#         'tcp.seq': torch.rand(batch_size),
#         'tcp.seq_raw': torch.rand(batch_size),
#         'tcp.ack': torch.rand(batch_size),
#         'tcp.ack_raw': torch.rand(batch_size),
#         'tcp.hdr_len': torch.rand(batch_size) * 40,
#         'tcp.flags': torch.randint(0, 96, (batch_size,)),
#         # ... 这里省略了所有TCP flag子字段的虚拟数据创建
#         'reassembled_segments': torch.randint(0, 1536, (batch_size,)),
#     }

#     # 3. 执行前向传播
#     output_vector = field_embedder(dummy_batch)

#     print(f"\nShape of the final concatenated output vector: {output_vector.shape}")
#     assert output_vector.shape == (batch_size, field_embedder.total_embedding_dim)