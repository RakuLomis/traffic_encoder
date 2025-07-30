import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os 
from tqdm import tqdm

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

# class TrafficDataset(Dataset):
#     """
#     一个经过最终性能优化的数据集类。
#     所有数据在初始化时被完全转换为原生Python类型，以实现最快的多进程加载。
#     """
#     def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str):
#         super().__init__()
        
#         # 1. 加载配置文件 (无变化)
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)['field_embedding_config']
#         with open(vocab_path, 'r') as f:
#             self.vocab_maps = yaml.safe_load(f)
            
#         # 2. 预处理标签 (无变化)
#         self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        
#         # 3. 预处理特征 (在Pandas层面完成列式转换)
#         print("正在一次性预处理所有特征...")
#         # index列只是为了分块时的联系，对于模型是无用的，可以安全删除
#         raw_features = dataframe.drop(columns=['label', 'label_id', 'index'], errors='ignore')
            
#         self.decimal_fields = {'tcp.stream'} 
        
#         processed_pandas_dict = self._preprocess_dataframe(raw_features)

#         # ==================== 核心优化点 开始 ====================
#         # 4. 将预处理过的Pandas数据，完全转换为一个【Tensor字典】的列表
#         #    这是最耗时的一步，但它只在程序启动时执行一次！
        
#         print("正在将数据转换为Tensor格式以实现极速加载...")
#         self.items = []
#         num_samples = len(self.labels)
        
#         # 将pandas字典转换为更快的“列式”访问
#         processed_columns = {k: v.to_list() for k, v in processed_pandas_dict.items()}
#         column_names = list(processed_columns.keys())
        
#         for i in tqdm(range(num_samples), desc="Converting to Tensors"):
#             # 直接从Python列表中按索引取值，这比.iloc快得多
#             item_py = {name: processed_columns[name][i] for name in column_names}
            
#             # 将每个值都转换为Tensor
#             item_tensor = {
#                 k: torch.tensor(v, dtype=torch.long) if not isinstance(v, list) else torch.tensor(v, dtype=torch.long)
#                 for k, v in item_py.items()
#             }
#             self.items.append(item_tensor)
#         # ==================== 核心优化点 结束 ====================

#     # _preprocess_dataframe 函数保持不变，它高效地处理列
#     def _preprocess_dataframe(self, df: pd.DataFrame):
#         # ... (您上一版中正确的 _preprocess_dataframe 逻辑可以原封不动地放在这里) ...
#         processed_data_dict = {}
#         for field_name in df.columns:
#             if field_name not in self.config:
#                 continue
            
#             if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
#                 field_type = self.config[field_name]['type']
#                 processed_column = df[field_name].apply(lambda x: _preprocess_address(x, field_type))
#             elif field_name in self.vocab_maps:
#                 vocab_map = self.vocab_maps[field_name]
#                 oov_index = vocab_map.get('__OOV__', len(vocab_map))
#                 processed_column = df[field_name].apply(
#                     lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
#                 )
#             elif field_name in self.decimal_fields:
#                 processed_column = df[field_name].fillna(0).astype(int)
#             elif self.config[field_name]['type'] in ['categorical', 'numerical']:
#                 def robust_hex_to_int(x):
#                     if not pd.notna(x): return 0
#                     str_x = str(x).split('.')[0]
#                     try: return int(str_x, 16)
#                     except ValueError: return 0
#                 processed_column = df[field_name].apply(robust_hex_to_int)
#             else:
#                 continue
#             processed_data_dict[field_name] = processed_column
#         return processed_data_dict

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # __getitem__ 现在快如闪电：它只做一次列表索引，返回一个纯Python字典
#         return self.items[idx], self.labels[idx]

### Not fast enough
class TrafficDataset(Dataset):
    """
    An optimized Dataset for multi-process data loading.
    __init__ is lightweight, and the conversion work is done in __getitem__.
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str):
        super().__init__()
        # 1. Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)

        # 2. Store labels and the raw feature DataFrame
        #    This is very lightweight.
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        self.raw_df = dataframe.drop(columns=['label', 'label_id'])
        
        # Pre-process the entire DataFrame column-wise ONCE. This is efficient.
        self.processed_df = self._preprocess_dataframe(self.raw_df)
        
        # Store the field names for quick access in __getitem__
        self.fields = list(self.processed_df.columns)

    def _preprocess_dataframe(self, df: pd.DataFrame):
        # This function takes the raw feature DataFrame and returns a
        # fully processed DataFrame where all values are numerical indices or lists.
        # This is an efficient, column-wise operation.
        processed_data_dict = {}
        for field_name in df.columns:
            if field_name not in self.config:
                continue

            # ... (your correct if/elif/else block for processing each column)
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                field_type = self.config[field_name]['type']
                processed_column = df[field_name].apply(lambda x: _preprocess_address(x, field_type))
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', len(vocab_map))
                processed_column = df[field_name].apply(
                    lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
                )
            # ... (rest of your robust processing logic for other types)
            else:
                 continue # Skip columns that don't have a processing rule
            
            processed_data_dict[field_name] = processed_column
        
        return pd.DataFrame(processed_data_dict)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        # This is now called in parallel by multiple workers.
        # .iloc[idx] is a very fast way to get a single row from a processed DataFrame.
        feature_row = self.processed_df.iloc[idx]
        features = {field: feature_row[field] for field in self.fields}
        label = self.labels[idx]
        return features, label

#### Single process
# class TrafficDataset(Dataset):
#     """
#     一个经过性能优化的数据集类。
#     所有耗时的预处理和格式转换都在初始化时一次性完成。
#     """
#     def __init__(self, dataframe: pd.DataFrame, config_path, vocab_path):
#         super().__init__()
        
#         # 1. 加载配置文件 (无变化)
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)['field_embedding_config']
#         with open(vocab_path, 'r') as f:
#             self.vocab_maps = yaml.safe_load(f)
            
#         # 2. 预处理标签 (无变化)
#         self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        
#         # 3. 预处理特征 (这是之前__getitem__的慢速部分)
#         print("正在一次性预处理所有特征...")
#         raw_features = dataframe.drop(columns=['label', 'label_id'])
#         if 'index' in raw_features.columns:
#             raw_features.set_index('index', inplace=True)
            
#         self.decimal_fields = {'tcp.stream', 'tcp.reassembled_segments'}
        
#         # 这一步仍然是在Pandas层面操作，相对较快
#         processed_pandas_dict = self._preprocess_dataframe(raw_features)

#         # ==================== 核心优化点 开始 ====================
#         # 4. 将预处理过的Pandas Series一次性转换为一个 item 列表
#         #    列表中的每个 item 都是一个可以直接被模型使用的字典
        
#         self.items = []
#         num_samples = len(self.labels)
        
#         # 将pandas字典转换为更快的“列式”访问
#         processed_columns = list(processed_pandas_dict.values())
#         column_names = list(processed_pandas_dict.keys())
        
#         for i in tqdm(range(num_samples), desc="正在将数据转换为可快速访问的格式"):
#             # 从已经处理好的列中按行索引构建字典，这比原始的iloc快
#             item = {name: col.iloc[i] for name, col in zip(column_names, processed_columns)}
#             self.items.append(item)
#         # ==================== 核心优化点 结束 ====================

#     # _preprocess_dataframe 函数现在接收一个df作为参数，逻辑不变
#     def _preprocess_dataframe(self, df: pd.DataFrame):
#         # ... (您上一版中正确的 _preprocess_dataframe 逻辑可以原封不动地放在这里) ...
#         processed_data_dict = {}
#         for field_name in df.columns:
#             if field_name not in self.config:
#                 continue
            
#             if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
#                 field_type = self.config[field_name]['type']
#                 processed_column = df[field_name].apply(lambda x: _preprocess_address(x, field_type))
#             elif field_name in self.vocab_maps:
#                 vocab_map = self.vocab_maps[field_name]
#                 oov_index = vocab_map.get('__OOV__', len(vocab_map))
#                 processed_column = df[field_name].apply(
#                     lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
#                 )
#             elif field_name in self.decimal_fields:
#                 processed_column = df[field_name].fillna(0).astype(int)
#             elif self.config[field_name]['type'] in ['categorical', 'numerical']:
#                 def robust_hex_to_int(x):
#                     if not pd.notna(x): return 0
#                     str_x = str(x).split('.')[0]
#                     try: return int(str_x, 16)
#                     except ValueError: return 0
#                 processed_column = df[field_name].apply(robust_hex_to_int)
#             else:
#                 continue
#             processed_data_dict[field_name] = processed_column
#         return processed_data_dict

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # __getitem__ 现在变得极其快速：它只做一次列表索引
#         return self.items[idx], self.labels[idx]


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
            
#         # ------------- 加速处理
#         # 标签
#         # self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)

#         # # 一次性处理特征
#         # raw_features = dataframe.drop(columns=['label', 'label_id'])
#         # processed_dict = self._preprocess_dataframe(raw_features)

#         # 转换为 tensor 列表
#         # self.items = []
#         # n = len(self.labels)
#         # for i in tqdm(range(n), desc="Converting to tensors"):
#         #     item = {
#         #         k: torch.tensor(v.iloc[i]) if not isinstance(v.iloc[i], list)
#         #            else torch.tensor(v.iloc[i], dtype=torch.long)
#         #         for k, v in processed_dict.items()
#         #     }
#         #     self.items.append(item)

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
#         # first_key = next(iter(self.fields))
#         # return len(self.processed_data[first_key]) 
#         # return len(self.raw_df)
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # # 获取索引为idx的一条数据
#         # # 返回一个字典，键为字段名，值为对应的数值
#         # sample = {field: self.processed_data[field].iloc[idx] for field in self.fields}
#         # return sample
#         features = {field: self.processed_data[field].iloc[idx] for field in self.fields}
#         label = self.labels[idx]
#         return features, label 
#         # return self.items[idx], self.labels[idx]