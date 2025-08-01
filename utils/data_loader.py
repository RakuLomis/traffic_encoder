import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import pandas as pd
import yaml
import os 
from tqdm import tqdm
import numpy as np

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


def custom_collate_fn(batch):
    """
    A custom collate function to batch features and labels efficiently.
    """
    # batch is a list of tuples: [(features_dict_0, label_0), (features_dict_1, label_1), ...]
    
    # 1. Separate features and labels
    feature_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # 2. Batch the features
    # Get all unique field names from the first sample
    field_names = feature_list[0].keys()
    batched_features = {}
    
    for field in field_names:
        # Collect all values for this field from the batch
        values = [sample[field] for sample in feature_list]
        
        # Stack them into a single tensor
        # Check if the item is a list (like an address) or a single number
        # if isinstance(values[0], list):
        #     batched_features[field] = torch.tensor(values, dtype=torch.long)
        # else:
        #     batched_features[field] = torch.tensor(values)
        # 将Python/Numpy的数值或列表，安全地转换为PyTorch Tensor
        try:
            # 对于地址（list）和分类/数值（numpy.int64），这个通用转换都有效
            batched_features[field] = torch.tensor(values, dtype=torch.long)
        except TypeError:
            # 如果出现混合类型等问题，可以逐个转换作为备用方案
            batched_features[field] = torch.tensor([v for v in values], dtype=torch.long)
            
    return batched_features, labels

class TrafficDataset(Dataset):
    """
    一个经过最终性能优化的数据集类。
    在初始化时将Pandas数据转换为更快的Numpy/List格式，以实现高效的多进程加载。
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
            
        self.labels = dataframe['label_id'].values
        
        print("正在一次性预处理所有特征...")
        raw_features = dataframe.drop(columns=['label', 'label_id', 'index'], errors='ignore')
        self.decimal_fields = {'tcp.stream'}
        processed_pandas_dict = self._preprocess_dataframe(raw_features)

        print("正在将数据转换为可快速访问的Numpy/List格式...")
        self.processed_data = {}
        for name, series in tqdm(processed_pandas_dict.items(), desc="Converting to fast-access format"):
            if not series.empty and isinstance(series.iloc[0], list):
                self.processed_data[name] = series.to_list()
            else:
                self.processed_data[name] = series.to_numpy()
        
        self.fields = list(self.processed_data.keys())

    def _preprocess_dataframe(self, df: pd.DataFrame):
        processed_data_dict = {}
        for field_name in df.columns:
            if field_name not in self.config:
                continue
            
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                field_type = self.config[field_name]['type']
                processed_column = df[field_name].apply(lambda x: _preprocess_address(x, field_type))
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', len(vocab_map))
                processed_column = df[field_name].apply(
                    lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
                )
            elif field_name in self.decimal_fields:
                processed_column = df[field_name].fillna(0).astype(int)
            elif self.config[field_name]['type'] in ['categorical', 'numerical']:
                def robust_hex_to_int(x):
                    if not pd.notna(x): return 0
                    str_x = str(x).split('.')[0]
                    try: return int(str_x, 16)
                    except ValueError: return 0
                processed_column = df[field_name].apply(robust_hex_to_int)
            else:
                continue
            processed_data_dict[field_name] = processed_column
        return processed_data_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {field: self.processed_data[field][idx] for field in self.fields}
        label = self.labels[idx]
        return features, label
"""
Original TrafficDataset
"""
# class TrafficDataset(Dataset):
#     """
#     Load and preprocess raw dataframe for traffic fields. 
#     """
#     # def __init__(self, csv_path, config_path, vocab_path): 
#     def __init__(self, dataframe: pd.DataFrame, config_path, vocab_path):
#         super().__init__()
        
#         # 1. 加载YAML配置
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)['field_embedding_config']

#         # --- 新增：加载生成的字典 ---
#         # 理想情况下，这个词典应该从训练数据中生成并保存/加载
#         print(f"Loading vocabulary from: {vocab_path}")
#         with open(vocab_path, 'r') as f:
#             self.vocab_maps = yaml.safe_load(f) # 使用 yaml.safe_load
        
#         # -----------------------------
        
#         # 2. 读取CSV数据
#         # self.raw_df = pd.read_csv(csv_path, dtype=str)
#         # 分离特征和标签
#         self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
#         self.raw_df = dataframe.drop(columns=['label', 'label_id'])

#         if 'index' in self.raw_df.columns:
#             self.raw_df.set_index('index', inplace=True) 
#         # 定义不需要进行十六进制转换的字段
#         self.decimal_fields = {'tcp.stream', 'tcp.reassembled_segments'}
        
#         # 3. 对整个DataFrame进行预处理，将所有值转换为数值格式
#         self.processed_data = self._preprocess_dataframe()
#         self.fields = list(self.processed_data.keys()) 
            

#     def _preprocess_dataframe(self):
#         """
#         遍历DataFrame的所有列，根据YAML配置将其转换为数值。
#         此版本集成了调试钩子，并修正了逻辑处理的优先级。
#         """
#         processed_data_dict = {}
#         for field_name in self.raw_df.columns:
#             # 首先，检查该字段是否在我们的总配置中，如果不在则完全跳过
#             if field_name not in self.config:
#                 if field_name not in ['index', 'label']: # 抑制对元数据列的警告
#                     # print(f"Warning: Field '{field_name}' not in config, skipping.")
#                     pass
#                 continue
            
#             # 使用严格互斥的 if/elif/else 结构，并按正确优先级排序

#             # 规则1 (最高优先级): 处理地址类型
#             if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
#                 field_type = self.config[field_name]['type']
#                 processed_column = self.raw_df[field_name].apply(lambda x: _preprocess_address(x, field_type))

#             # 规则2: 使用生成的词典进行映射
#             elif field_name in self.vocab_maps:
#                 vocab_map = self.vocab_maps[field_name]
#                 oov_index = vocab_map.get('__OOV__')
#                 if oov_index is None: # 安全检查
#                     oov_index = len(vocab_map)

#                 # ==================== 调试钩子 开始 ====================
#                 # 我们定义一个更详细的函数来替代简单的lambda
#                 def find_mismatch_and_map(x):
#                     if not pd.notna(x):
#                         return oov_index
                    
#                     # 确保我们用来查找的键，与generate_vocab.py中创建键的方式完全一致
#                     key_to_lookup = str(x).lower().replace('0x', '')
                    
#                     if key_to_lookup not in vocab_map:
#                         # 如果在词典中找不到这个键，就打印详细的错误报告
#                         print("\n" + "="*60)
#                         print("!!! Potential mismatch with vocab yaml !!!")
#                         print(f"FIELD:           '{field_name}'")
#                         print(f"Value from data (VALUE):   '{x}'")
#                         print(f"Key used to search (KEY):     '{key_to_lookup}'")
#                         print("This KEY is not found in vocab.yaml. ")
#                         print("It may be the reason of IndexError. ")
#                         print("Please check the vocab.yaml. ")
#                         print("="*60 + "\n")
#                         return oov_index # 将未找到的值映射到OOV索引
                    
#                     # 如果找到了，返回正确的索引
#                     return vocab_map[key_to_lookup]
#                 # ==================== 调试钩子 结束 ====================
                
#                 # 应用我们刚刚定义的这个更强大的函数
#                 processed_column = self.raw_df[field_name].apply(find_mismatch_and_map)

#             # 规则3: 处理特殊的十进制字段
#             elif field_name in self.decimal_fields:
#                 processed_column = self.raw_df[field_name].fillna(0).astype(int)

#             # 规则4 (最低优先级): 处理其他所有需要从十六进制转整数的字段
#             elif self.config[field_name]['type'] in ['categorical', 'numerical']:
#                 def robust_hex_to_int(x):
#                     if not pd.notna(x):
#                         return 0
#                     str_x = str(x).split('.')[0]
#                     try:
#                         return int(str_x, 16)
#                     except ValueError:
#                         return 0
#                 processed_column = self.raw_df[field_name].apply(robust_hex_to_int)
                
#             else:
#                 continue
            
#             processed_data_dict[field_name] = processed_column
            
#         return processed_data_dict

#     def __len__(self):
#         # 数据集的长度就是DataFrame的行数
#         # 我们以第一列的长度为准
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # # 获取索引为idx的一条数据
#         # # 返回一个字典，键为字段名，值为对应的数值
#         features = {field: self.processed_data[field].iloc[idx] for field in self.fields}
#         label = self.labels[idx]
#         return features, label 
    