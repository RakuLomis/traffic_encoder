import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data 
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree, add_root_layer
from utils.data_loader import _preprocess_address
from collections import defaultdict
import torch.nn as nn
from typing import Dict, List, Any
from torch_geometric.data import Batch
import gc

class GNNTrafficDataset(Dataset):
    """
    【!! 最终版：包含 Tensor Cache + 端口分箱 !!】
    
    一个为GNN模型准备数据的Dataset。
    它的__init__方法会构建“统一图”骨架，并创建一个多进程安全的“Tensor Cache”。
    它的__getitem__方法从Cache中高速切片，返回一个*单一*的全局图(Data)对象。
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, 
                 use_flow_features: bool = False):
        super().__init__()
        print(f"\nInitializing GNNTrafficDataset (Tensor Cache, Port Binning, Flow: {use_flow_features})...")
        self.use_flow_features = use_flow_features

        # 1. 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
        self.flow_feature_names = [
            'flow_avg_len', 'flow_std_len', 'flow_pkt_count',
            'flow_avg_iat', 'flow_std_iat', 'flow_max_iat',
            'flow_duration_per_pkt',
            'flow_large_pkt_ratio'
        ]
        
        # --- 【!! 核心修改：注入端口分箱逻辑 !!】 ---
        
        # 1. 定义哪些字段是端口 (YAML 中 type=categorical)
        self.port_fields = {'tcp.srcport', 'tcp.dstport'}
        
        # 2. 定义“分箱”词汇表 (逻辑Bin -> 嵌入Index)
        port_bin_vocab = {
            0: 0, # Bin 0: Unknown/NaN/Port 0
            1: 1, # Bin 1: Well-Known (1-1023)
            2: 2, # Bin 2: Registered (1024-49151)
            3: 3  # Bin 3: Ephemeral (49152+)
        }
        
        # 3. 强行注入 self.vocab_maps
        print(" -> Injecting semantic port binning vocabulary...")
        for field in self.port_fields:
            self.vocab_maps[field] = port_bin_vocab
            
        # 4. 【重要】定义哪些字段是 *十进制* 数值 (不是Hex)
        #    (这必须在 YAML 中也定义为 type: numerical)
        self.decimal_fields = {'tcp.stream'} 
        
        # --- [!! 注入结束 !!] ---

        # 2. 【修改】定义 *所有* 真实节点
        all_available_fields = set(dataframe.columns)
        self.all_feature_cols_to_process = set()
        
        expert_definitions = {
            'eth': {f for f in all_available_fields if f.startswith('eth.')},
            'ip': {f for f in all_available_fields if f.startswith('ip.')},
            'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f},
            'tcp_options': {f for f in all_available_fields if f.startswith('tcp.options.')},
            'tls_record': {f for f in all_available_fields if f.startswith('tls.record.')},
            'tls_handshake': {f for f in all_available_fields if f.startswith('tls.handshake.')},
            'tls_x509': {f for f in all_available_fields if f.startswith('tls.x509')}
        }
        
        for name, expert_fields in expert_definitions.items():
            real_nodes_for_expert = expert_fields.intersection(all_available_fields)
            self.all_feature_cols_to_process.update(real_nodes_for_expert)
        
        real_nodes_in_df = sorted(list(self.all_feature_cols_to_process))

        # 3. 【修改】预先生成 *一个* 全局图结构
        print("Pre-calculating the single Global Graph structure...")
        # (这假设 protocol_tree 和 add_root_layer 是你的 "统一图" F1=0.89 版本)
        ptree = protocol_tree(real_nodes_in_df)
        add_root_layer(ptree) # <-- 【重要】使用你的 "统一图" 版 add_root_layer
        
        ptree_nodes = set(ptree.keys())
        for children in ptree.values(): ptree_nodes.update(children)
        
        all_nodes_global = sorted(list(ptree_nodes))
        field_to_node_idx_global = {n: i for i, n in enumerate(all_nodes_global)}
        
        self.global_graph = {
            'real_nodes': set(real_nodes_in_df), 
            'all_nodes': all_nodes_global, 
            'field_to_node_idx': field_to_node_idx_global,
            'edge_index': self._create_edge_index_from_tree(ptree, field_to_node_idx_global)
        }
        print(f" -> Global graph created with {len(all_nodes_global)} total nodes.")

        if self.use_flow_features:
            print("Flow features enabled. Adding to processing list.")
            self.all_feature_cols_to_process.update(self.flow_feature_names)

        # 4. 【修改】预处理 (使用 Tensor Cache 方案)
        print(f"Pre-processing all {len(self.all_feature_cols_to_process)} columns...")
        processed_df = self._preprocess_all(dataframe)
        
        print("Converting processed DataFrame to tensor cache...")
        self.tensor_cache = {}
        cols_in_df = set(processed_df.columns)
        
        for col_name in tqdm(self.all_feature_cols_to_process, desc="Creating tensor cache"):
            if col_name not in cols_in_df: continue
            col_data = processed_df[col_name]
            
            # (这部分 Tensor Cache 逻辑保持不变)
            first_val = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
            if isinstance(first_val, (list, tuple)):
                length = 4 if 'ip' in col_name else (6 if 'mac' in col_name else 4)
                def fill_empty_addr(x, length=length):
                    if not isinstance(x, (list, tuple)) or len(x) == 0: return [0] * length
                    return x
                data_list = col_data.apply(fill_empty_addr).tolist()
                self.tensor_cache[col_name] = torch.tensor(data_list, dtype=torch.long)
            elif col_name in self.flow_feature_names:
                self.tensor_cache[col_name] = torch.tensor(col_data.values, dtype=torch.float)
            else:
                self.tensor_cache[col_name] = torch.tensor(col_data.values, dtype=torch.long)

        del processed_df
        del dataframe
        gc.collect()
        print("Tensor cache created. Processed DataFrame released.")

    
    def _create_edge_index_from_tree(self, ptree: Dict[str, List[str]], field_to_node_idx: Dict[str, int]) -> torch.Tensor:
        # ... (此函数保持不变) ...
        edge_list = []
        for parent, children in ptree.items(): 
            if parent in field_to_node_idx:
                parent_idx = field_to_node_idx[parent]
                for child in children:
                    if child in field_to_node_idx:
                        child_idx = field_to_node_idx[child]
                        edge_list.append([child_idx, parent_idx])
                        edge_list.append([parent_idx, child_idx])
        if not edge_list: 
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    
    def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【!! 最终修复版 (V2) !!】
        对整个DataFrame进行一次性的、类型安全的预处理。
        实现了端口分箱、时间戳解析和正确的类型分派。
        """
        processed_data_dict = {}
        
        # --- 1. 在函数内部定义辅助转换器 ---
        
        def robust_hex_to_int(x):
            """(用于 ip.len, tcp.len 等)"""
            if not pd.notna(x): return 0
            try:
                return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
            except ValueError:
                return 0

        def robust_timestamp_to_tsval(x):
            """(只用于 tcp.options.timestamp)"""
            if not pd.notna(x): return 0
            try:
                s = str(x).lower().replace('0x', '')
                if len(s) != 20 or not s.startswith('080a'): return 0
                tsval_hex = s[4:12]
                return int(tsval_hex, 16)
            except (ValueError, TypeError): return 0
            
        def bin_port_from_hex(x):
            """(只用于 port_fields)"""
            if not pd.notna(x): return 0 # Bin 0
            try:
                port_int = int(str(x).split('.')[0], 16)
                if 1 <= port_int <= 1023:
                    return 1 # Bin 1: Well-Known
                elif 1024 <= port_int <= 49151:
                    return 2 # Bin 2: Registered
                elif port_int >= 49152:
                    return 3 # Bin 3: Ephemeral
                else:
                    return 0 # Bin 0 (port 0)
            except ValueError:
                return 0
        
        # --- 辅助函数定义结束 ---
        
        cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
        for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
            
            # --- 分支 1: 流统计特征 (Flow Stats) ---
            if field_name in self.flow_feature_names:
                col_data_numeric = pd.to_numeric(df[field_name], errors='coerce')
                # 【!! 修复 !!】 确保清理 inf, -inf 和 nan
                col_data_np = col_data_numeric.values
                col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0) 
                processed_data_dict[field_name] = col_data_cleaned.astype(np.float32)
                continue
                
            # --- 分支 2: GNN 包头特征 ---
            if field_name not in self.config:
                continue 
            
            config = self.config[field_name]
            col_data = df[field_name]
            field_type = config['type']

            # --- 按类型分派 ---
            
            if field_type in ['address_ipv4', 'address_mac']:
                # 分支 2a: 地址
                processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
            elif field_name in self.port_fields:
                # 分支 2b (新): 端口分箱
                # (YAML type 必须是 'categorical' 才能到这里)
                
                # a. Hex Port -> Bin ID (0, 1, 2, 3)
                binned_data = col_data.apply(bin_port_from_hex)
                
                # b. Map Bin ID -> Embedding Index (使用注入的 {0:0, 1:1, ...})
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', 0)
                processed_data_dict[field_name] = binned_data.map(vocab_map).fillna(oov_index).astype(np.int32)
            
            elif field_type == 'categorical':
                # 分支 2c: *其他* 分类字段
                if field_name in self.vocab_maps:
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = vocab_map.get('__OOV__', 0)
                    processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
                else:
                    # (例如 tls.x509af.serialNumber, 保持跳过)
                    pass 

            elif field_type == 'numerical':
                # 分支 2d: 数值字段
                
                if field_name in self.decimal_fields:
                    # (例如 tcp.stream)
                    processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
                
                elif field_name == 'tcp.options.timestamp':
                    # (完整的 080a... 字符串)
                    processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).astype(np.int64)
                
                else:
                    # (默认: ip.len, tcp.len, tsval, tsecr ...)
                    processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
            else:
                 print(f"警告: 字段 '{field_name}' 的类型 '{field_type}' 无法识别。将跳过。")

        return pd.DataFrame(processed_data_dict)

    
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, idx) -> Data:
        """
        【全局图版本 - Tensor Cache】
        """
        y = self.labels[idx].view(1) 
        container_data = Data(y=y)

        feature_dict_global = {}
        all_nodes_list = self.global_graph['all_nodes']
        real_nodes_set = self.global_graph['real_nodes']
        
        for field_name in all_nodes_list:
            tensor_value = None
            is_real_node = field_name in real_nodes_set
            
            if is_real_node and field_name in self.tensor_cache:
                tensor_value = self.tensor_cache[field_name][idx]
                
                # 确保形状 (切片后 1D/0D -> 2D/1D)
                if tensor_value.dim() == 0:
                    tensor_value = tensor_value.unsqueeze(0) 
                elif tensor_value.dim() == 1:
                    config_type = self.config.get(field_name, {}).get('type', 'numerical')
                    if 'address' in config_type:
                        tensor_value = tensor_value.unsqueeze(0)
                
                feature_dict_global[field_name] = tensor_value.clone()
            
            else:
                # 抽象节点 或 真实但缺失的节点
                config_type = self.config.get(field_name, {}).get('type', 'numerical')
                if config_type == 'address_ipv4':
                    tensor_value = torch.zeros((1, 4), dtype=torch.long)
                elif config_type == 'address_mac':
                    tensor_value = torch.zeros((1, 6), dtype=torch.long)
                else:
                    tensor_value = torch.tensor([0], dtype=torch.long)
                feature_dict_global[field_name] = tensor_value

        graph_data = Data(
            edge_index=self.global_graph['edge_index'],
            y=y,
            num_nodes=len(all_nodes_list),
            **feature_dict_global
        )

        if self.use_flow_features:
            flow_stats_list = [self.tensor_cache[f][idx] for f in self.flow_feature_names]
            graph_data.flow_stats = torch.stack(flow_stats_list).view(1, -1)
        
        return graph_data

# class GNNTrafficDataset(Dataset):
#     """
#     一个为GNN模型准备数据的Dataset。
#     它的__getitem__方法将一个数据包（一行DataFrame）转换成一个PyG的图(Data)对象。
#     """
#     def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, node_feature_dim: int=128, 
#                  use_flow_features: bool = False):
#         super().__init__()
#         print(f"\nInitializing Hierarchical GNNTrafficDataset (Flow Features: {use_flow_features})...")
#         self.use_flow_features = use_flow_features

#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)['field_embedding_config']
#         with open(vocab_path, 'r') as f:
#             self.vocab_maps = yaml.safe_load(f)
#         self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
#         self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
#         self.decimal_fields = {'tcp.stream'}

#         # --- 【!! 核心修改：注入端口分箱逻辑 !!】 ---
        
#         # 1. 定义哪些字段是端口
#         self.port_fields = {'tcp.srcport', 'tcp.dstport'}
        
#         # 2. 定义我们的“分箱”词汇表
#         #    注意：键(Key)是*逻辑*分箱号 (0-3)
#         #    值(Value)是*嵌入索引* (0-3)
#         port_bin_vocab = {
#             0: 0, # Bin 0 -> Index 0 (Unknown/NaN/Port 0)
#             1: 1, # Bin 1 -> Index 1 (Well-Known Ports, 1-1023)
#             2: 2, # Bin 2 -> Index 2 (Registered Ports, 1024-49151)
#             3: 3  # Bin 3 -> Index 3 (Ephemeral Ports, 49152-65535)
#         }
        
#         # 3. 将这个词汇表强行注入到 self.vocab_maps
#         #    (这也会覆盖掉 vocab.yaml 中任何旧的、巨大的端口映射)
#         print(" -> Injecting semantic port binning vocabulary...")
#         for field in self.port_fields:
#             self.vocab_maps[field] = port_bin_vocab
            
#         # --- [!! 修复结束 !!] ---

#         # --- 2. 【核心修改点】定义“专家”及其“视野” (Schema) --- 
#         all_available_fields = set(dataframe.columns)
        
#         # 您可以根据需要，像配置表一样，精确地定义这些专家的字段
#         self.expert_definitions = {
#             'eth': {f for f in all_available_fields if f.startswith('eth.')},
#             'ip': {f for f in all_available_fields if f.startswith('ip.')},
#             'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f},
#             'tcp_options': {f for f in all_available_fields if f.startswith('tcp.options.')},
#             'tls_record': {f for f in all_available_fields if f.startswith('tls.record.')},
#             'tls_handshake': {f for f in all_available_fields if f.startswith('tls.handshake.')},
#             'tls_x509': {f for f in all_available_fields if f.startswith('tls.x509')}
#             # ... 您可以根据需要，定义任意多的“专家”
#         }
#         self.flow_feature_names = ['flow_avg_len', 'flow_std_len', 'flow_pkt_count']

#         # --- 3. 【核心修改点】为每个“专家”预先生成图结构 ---
#         print("Pre-calculating graph structures for each expert...")
#         self.expert_graphs = {}
#         # 收集所有需要从DataFrame中读取的字段
#         self.all_feature_cols_to_process = set() 
        
#         for name, expert_fields in self.expert_definitions.items():
#             # 找到这个专家Schema中，实际存在于DataFrame中的字段
#             real_nodes_for_expert = sorted(list(expert_fields.intersection(all_available_fields)))
#             if not real_nodes_for_expert:
#                 print(f" -> 警告: 专家 '{name}' 在数据集中没有任何对应的字段，将跳过。")
#                 continue

#             # 为这个专家的“小世界”构建协议树
#             ptree = protocol_tree(real_nodes_for_expert)
#             add_root_layer(ptree)
            
#             # 找到所有真实节点和抽象节点
#             ptree_nodes = set(ptree.keys())
#             for children in ptree.values(): 
#                 ptree_nodes.update(children)
            
#             all_nodes_for_expert = sorted(list(ptree_nodes))
#             field_to_node_idx = {n: i for i, n in enumerate(all_nodes_for_expert)}
            
#             # 存储这个专家的所有信息
#             self.expert_graphs[name] = {
#                 'real_nodes': real_nodes_for_expert, # 真实存在的字段
#                 'all_nodes': all_nodes_for_expert,  # 真实+抽象节点
#                 'field_to_node_idx': field_to_node_idx,
#                 'edge_index': self._create_edge_index_from_tree(ptree, field_to_node_idx)
#             }
#             # 将这些字段加入到“总处理列表”
#             self.all_feature_cols_to_process.update(real_nodes_for_expert)

#         if self.use_flow_features: 
#             print("Flow features enabled. Adding to processing list.")
#             self.all_feature_cols_to_process.update(self.flow_feature_names)
#         else:
#             print("Flow features disabled.")

#         # --- 4. 【性能飞跃】对整个DataFrame进行一次性预处理 ---
#         print(f"Pre-processing all {len(self.all_feature_cols_to_process)} columns...")
#         self.processed_df = self._preprocess_all(dataframe)
#         print("Dataset initialization complete.")

        
#     def _create_edge_index_from_tree(self, ptree: Dict[str, List[str]], field_to_node_idx: Dict[str, int]) -> torch.Tensor:
#         """【新】辅助函数：根据传入的ptree和idx映射，创建边索引。"""
#         edge_list = []
#         for parent, children in ptree.items(): 
#             if parent in field_to_node_idx:
#                 parent_idx = field_to_node_idx[parent]
#                 for child in children:
#                     if child in field_to_node_idx:
#                         child_idx = field_to_node_idx[child]
#                         edge_list.append([child_idx, parent_idx])
#                         edge_list.append([parent_idx, child_idx])
#         if not edge_list: 
#             return torch.empty((2, 0), dtype=torch.long)
#         return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
#     def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         【新】核心性能优化：
#         在__init__中，对整个DataFrame的所有列，进行一次性的向量化预处理。
#         """
#         processed_data_dict = {}
        
#         # 筛选出我们需要处理的、且存在于df中的列
#         cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
#         for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
#             # 1. 流统计特征 (Flow Stats)
#             if field_name in self.flow_feature_names:
#                 processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
#                 continue
                
#             # 2. 包头特征 (GNN Node Features)
#             if field_name not in self.config:
#                 continue # 跳过未配置的字段
            
#             config = self.config[field_name]
#             col_data = df[field_name]

#             # 应用您之前验证过的、健壮的if/elif逻辑 (现在是向量化版本)
#             if config['type'] in ['address_ipv4', 'address_mac']:
#                 field_type = config['type']
#                 processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
#             elif field_name in self.vocab_maps:
#                 vocab_map = self.vocab_maps[field_name]
#                 oov_index = vocab_map.get('__OOV__', 0) # 假设OOV=0
#                 # processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(int)
#                 processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
            
#             # elif field_name == 'tcp.stream': # 您之前的 decimal_fields
#             #     processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
#             # 【修改】使用 self.decimal_fields 变量
#             elif field_name in self.decimal_fields: 
#                 # processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
#                 processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
            
#             elif config['type'] in ['categorical', 'numerical']:
#                 def robust_hex_to_int(x):
#                     if not pd.notna(x): return 0
#                     try:
#                         return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
#                     except ValueError:
#                         return 0
#                 # processed_data_dict[field_name] = col_data.apply(robust_hex_to_int)
#                 processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
#         return pd.DataFrame(processed_data_dict)

#     # def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
#     #     """
#     #     【新】核心性能优化：
#     #     在__init__中，对整个DataFrame的所有列，进行一次性的向量化预处理。
#     #     """
#     #     processed_data_dict = {}
        
#     #     # 筛选出我们需要处理的、且存在于df中的列
#     #     cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
#     #     for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
#     #         # 1. 流统计特征 (Flow Stats)
#     #         if field_name in self.flow_feature_names:
#     #             processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
#     #             continue
                
#     #         # 2. 包头特征 (GNN Node Features)
#     #         if field_name not in self.config:
#     #             continue # 跳过未配置的字段
            
#     #         config = self.config[field_name]
#     #         col_data = df[field_name]

#     #         # 应用您之前验证过的、健壮的if/elif逻辑 (现在是向量化版本)
#     #         if config['type'] in ['address_ipv4', 'address_mac']:
#     #             field_type = config['type']
#     #             processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
#     #         elif field_name in self.vocab_maps:
#     #             vocab_map = self.vocab_maps[field_name]
#     #             oov_index = vocab_map.get('__OOV__', 0) # 假设OOV=0
#     #             # processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(int)
#     #             processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
            
#     #         # elif field_name == 'tcp.stream': # 您之前的 decimal_fields
#     #         #     processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
#     #         # 【修改】使用 self.decimal_fields 变量
#     #         elif field_name in self.decimal_fields: 
#     #             # processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
#     #             processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
            
#     #         elif config['type'] in ['categorical', 'numerical']:
#     #             def robust_hex_to_int(x):
#     #                 if not pd.notna(x): return 0
#     #                 try:
#     #                     return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
#     #                 except ValueError:
#     #                     return 0
#     #             # processed_data_dict[field_name] = col_data.apply(robust_hex_to_int)
#     #             processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
#     #     return pd.DataFrame(processed_data_dict)

#     def __len__(self):
#         return len(self.labels)


#     def __getitem__(self, idx) -> Dict[str, Any]:
#         """
#         【新 - 已修正 KeyError 和 形状问题】
#         【再修正 - 确保抽象节点也具有特征属性】
#         从“已预处理”的DataFrame中，快速切片，并组装成一个“专家图字典”。
#         """
#         processed_row = self.processed_df.iloc[idx]
#         y = self.labels[idx].view(1) # 保持 [1] 形状
        
#         data_dict = {}

#         for expert_name, graph_info in self.expert_graphs.items():
            
#             feature_dict_for_expert = {}
#             all_nodes_list = graph_info['all_nodes']
#             real_nodes_set = set(graph_info['real_nodes']) # 转换为集合以便快速查找
            
#             # --- 核心修改点：遍历 *所有* 节点，而不仅仅是 real_nodes ---
#             for field_name in all_nodes_list:
                
#                 value = None
#                 is_real_node = field_name in real_nodes_set
                
#                 # 1. 如果是真实节点，尝试从行数据中获取值
#                 if is_real_node:
#                     value = processed_row.get(field_name) # value 可能是真实值，也可能是 None
                
#                 # 2. value 现在的情况:
#                 #    a) 真实值 (list, int, float)
#                 #    b) None (因为是抽象节点，或 真实节点但数据缺失)
                
#                 tensor_value = None
                
#                 if isinstance(value, list):
#                     # --- 针对地址类型 (list) ---
#                     tensor_value = torch.tensor(value, dtype=torch.long).view(1, -1)
                
#                 elif isinstance(value, (int, float, np.number)):
#                     # --- 针对分类/数值类型 (int/float) ---
#                     tensor_value = torch.tensor([value], dtype=torch.long)
                
#                 else:
#                     # --- 针对缺失值 (None) 或 抽象节点 ---
#                     # 检查配置，看它 *应该* 是什么形状
#                     config_type = self.config.get(field_name, {}).get('type', 'numerical')
                    
#                     if config_type == 'address_ipv4':
#                         tensor_value = torch.zeros((1, 4), dtype=torch.long)
#                     elif config_type == 'address_mac':
#                         tensor_value = torch.zeros((1, 6), dtype=torch.long)
#                     else:
#                         # 抽象节点 (如 'ip', 'ROOT') 和 缺失的真实节点 (如 'tcp.flags')
#                         # 都会得到一个 tensor([0]) 作为占位符
#                         tensor_value = torch.tensor([0], dtype=torch.long)

#                 feature_dict_for_expert[field_name] = tensor_value
#                 # =================================================================

#             graph_data = Data(
#                 edge_index=graph_info['edge_index'],
#                 y=y,
#                 num_nodes=len(all_nodes_list), # 使用 all_nodes_list 的长度
#                 **feature_dict_for_expert
#             )
#             data_dict[expert_name] = graph_data

#         if self.use_flow_features:
#             flow_stats_list = [processed_row.get(f, 0.0) for f in self.flow_feature_names]
#             data_dict['flow_stats'] = torch.tensor(flow_stats_list, dtype=torch.float).view(1, -1)
        
#         return data_dict
