import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os

def _preprocess_address(addr_str, addr_type):
    """Turn the hex address into a dec addresss list. """
    if pd.isna(addr_str):
        num_octets = 4 if addr_type == 'address_ipv4' else 6
        return [0] * num_octets # 用0填充缺失的地址
    
    # 确保输入是字符串
    addr_str = str(addr_str)
    # 对于pyshark可能输出的'x'格式，先移除
    if 'x' in addr_str:
        addr_str = addr_str.split('x')[-1]
    
    octets = []
    if addr_type == 'address_ipv4':
        # e.g., 'c0a80503' -> ['c0', 'a8', '05', '03']
        for i in range(0, 8, 2):
            octets.append(int(addr_str[i:i+2], 16))
        return octets
    elif addr_type == 'address_mac':
        # e.g., 'f42d06784ee9' -> ['f4', '2d', ...]
        for i in range(0, 12, 2):
            octets.append(int(addr_str[i:i+2], 16))
        return octets
    print("Something was wrong, so we return a empty list. ")
    return []


class TrafficDataset(Dataset):
    """
    Load and preprocess raw dataframe for traffic fields. 
    """
    def __init__(self, csv_path, config_path, vocab_path):
        super().__init__()
        
        # 1. 加载YAML配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']

        # --- 新增：加载生成的字典 ---
        # 理想情况下，这个词典应该从训练数据中生成并保存/加载
        print(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f) # 使用 yaml.safe_load
        
        # -----------------------------
        
        # 2. 读取CSV数据
        self.raw_df = pd.read_csv(csv_path, dtype=str)
        # self.raw_df = pd.read_csv(csv_path, index_col='index')
        if 'index' in self.raw_df.columns:
            self.raw_df.set_index('index', inplace=True) 
        # 定义不需要进行十六进制转换的字段
        self.decimal_fields = {'tcp.stream', 'tcp.reassembled_segments'}
        
        # 3. 对整个DataFrame进行预处理，将所有值转换为数值格式
        self.processed_data = self._preprocess_dataframe()
        self.fields = list(self.processed_data.keys())

    def _preprocess_dataframe(self):
        """
        遍历DataFrame的所有列，根据YAML配置将其转换为数值。
        (已修正逻辑覆盖问题)
        """
        processed_data_dict = {}
        for field_name in self.raw_df.columns:
            if field_name not in self.config:
                if field_name != 'index': 
                    print(f"Warning: Field '{field_name}' not in config, skipping.")
                    pass
                continue
            
            # 使用互斥的 if/elif/else 结构来确保每个字段只被处理一次

            # 处理地址数据
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                field_type = self.config[field_name]['type']
                processed_column = self.raw_df[field_name].apply(lambda x: _preprocess_address(x, field_type))
            
            # 使用生成的词典进行映射
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map['__OOV__']
                
                # 安全地处理各种输入，统一为小写字符串进行查找
                processed_column = self.raw_df[field_name].apply(
                    lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
                )
            # 其实这样转换之后
            # 处理那些本身就是十进制的特殊字段
            elif field_name in self.decimal_fields:
                processed_column = self.raw_df[field_name].fillna(0).astype(int)

            # 处理其他所有需要从十六进制转为整数的字段
            elif self.config[field_name]['type'] in ['categorical', 'numerical']:

                def robust_hex_to_int(x):
                    if not pd.notna(x):
                        return 0
                    # Convert to string and handle cases like '45.0' by splitting at the decimal
                    str_x = str(x).split('.')[0]
                    try:
                        return int(str_x, 16)
                    except ValueError:
                        return 0 # Return 0 if it's still not a valid hex

                # processed_column = self.raw_df[field_name].apply(lambda x: int(str(x), 16) if pd.notna(x) else 0)
                processed_column = self.raw_df[field_name].apply(robust_hex_to_int)
                
            else:
                # 如果有任何未覆盖的情况，跳过该字段
                continue
            
            processed_data_dict[field_name] = processed_column
            
        return processed_data_dict

    # def _preprocess_dataframe(self):
    #     """
    #     遍历DataFrame的所有列，根据YAML配置将其转换为数值。
    #     """
    #     processed_data_dict = {}
    #     for field_name in self.raw_df.columns:
    #         if field_name not in self.config: # use the config file as standard
    #             if field_name != 'frame_num':
    #                 print(f"Warning: Field '{field_name}' not in config, skipping.")
    #             continue
            
    #         if field_name in self.vocab_maps:
    #             # 如果字段有预定义的词典，使用它
    #             vocab_map = self.vocab_maps[field_name]
    #             oov_index = vocab_map['__OOV__']
    #             # .get(key, default_value) 是一个安全的查字典方法
    #             processed_column = self.raw_df[field_name].astype(str).str.lower().apply(
    #                 lambda x: vocab_map.get(x.replace('0x',''), oov_index)
    #             ) 

    #         field_type = self.config[field_name]['type']
    #         column_data = self.raw_df[field_name]

    #         if field_name in self.decimal_fields:
    #             # 对于本身就是十进制的字段，直接转换为整数，处理缺失值
    #             processed_column = column_data.fillna(0).astype(int)
    #         elif field_type in ['categorical', 'numerical']:
    #             # 对于十六进制的分类和数值字段，转换为整数
    #             processed_column = column_data.apply(lambda x: int(str(x), 16) if pd.notna(x) else 0)
    #         elif field_type in ['address_ipv4', 'address_mac']:
    #             # 对于地址字段，使用辅助函数处理
    #             processed_column = column_data.apply(lambda x: _preprocess_address(x, field_type))
    #         else:
    #             # 跳过未知类型
    #             continue
            
    #         processed_data_dict[field_name] = processed_column
            
    #     return processed_data_dict

    def __len__(self):
        # 数据集的长度就是DataFrame的行数
        # 我们以第一列的长度为准
        first_key = next(iter(self.fields))
        return len(self.processed_data[first_key])

    def __getitem__(self, idx):
        # 获取索引为idx的一条数据
        # 返回一个字典，键为字段名，值为对应的数值
        sample = {field: self.processed_data[field].iloc[idx] for field in self.fields}
        return sample

# --- 使用示例 ---
if __name__ == '__main__':
    # 假设这是您的项目结构和文件路径
    # from models.FieldEmbedding import FieldEmbedding # 引入我们之前创建的模块
    
    # 为了演示，我们先伪造一个FieldEmbedding类
    class FieldEmbedding(torch.nn.Module):
        def __init__(self, config_path):
            super().__init__()
            print("Dummy FieldEmbedding model initialized.")
        def forward(self, x):
            print("Dummy forward pass.")
            return torch.rand(list(x.values())[0].size(0), 256) # 返回一个假的输出向量
    
    # 1. 设置路径
    # 假设脚本在项目根目录运行
    config_file_path = './utils/f2v.yaml'
    # 假设我们要加载这个分块后的CSV文件
    data_csv_path = './Data/Test/merge_tls_test_01/discrete/0.csv'

    print(f"Loading data from: {data_csv_path}")
    print(f"Loading config from: {config_file_path}")

    # 2. 创建Dataset实例
    traffic_dataset = TrafficDataset(csv_path=data_csv_path, config_path=config_file_path)
    
    # 看一条预处理后的数据
    print("\n--- Sample of preprocessed data [index 0] ---")
    sample_item = traffic_dataset[0]
    # 为了简洁，只打印前5个字段
    for i, (k, v) in enumerate(sample_item.items()):
        if i >= 5: break
        print(f"'{k}': {v}")
    
    # 3. 创建DataLoader实例
    # DataLoader会自动将多条数据打包成一个batch，并将数值转换为PyTorch Tensor
    batch_size = 4
    traffic_dataloader = DataLoader(
        dataset=traffic_dataset,
        batch_size=batch_size,
        shuffle=True # 在训练时打乱数据顺序
    )

    # 4. 实例化我们的嵌入模型
    field_embedder = FieldEmbedding(config_path=config_file_path)
    # field_embedder.to(device) # 如果使用GPU

    # 5. 从DataLoader中取一个批次的数据进行测试
    print(f"\n--- Testing with a DataLoader (batch_size={batch_size}) ---")
    first_batch = next(iter(traffic_dataloader))
    
    print("\n--- Shape of tensors in the first batch ---")
    # 为了简洁，只打印前5个字段的张量形状
    for i, (k, v) in enumerate(first_batch.items()):
        if i >= 5: break
        print(f"'{k}': {v.shape}")

    # 6. 将整个批次送入模型
    print("\n--- Passing batch to the model ---")
    output_vector = field_embedder(first_batch)
    print(f"\nShape of the final model output vector: {output_vector.shape}")