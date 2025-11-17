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
    一个为GNN模型准备数据的Dataset。
    它的__getitem__方法将一个数据包（一行DataFrame）转换成一个PyG的图(Data)对象。
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, node_feature_dim: int=128, 
                 use_flow_features: bool = False, use_ip_address: bool = True):
        super().__init__()
        print(f"\nInitializing Hierarchical GNNTrafficDataset (Flow Features: {use_flow_features})...")
        self.use_flow_features = use_flow_features

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
        self.decimal_fields = {'tcp.stream'}

        # --- 【!! 核心：注入端口分箱逻辑 !!】 ---
        
        # 1. 定义哪些字段是端口 (YAML type 必须是 'categorical')
        self.port_fields = {'tcp.srcport', 'tcp.dstport'}
        
        # 2. 定义我们的“分箱”词汇表 (逻辑Bin -> 嵌入Index)
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
            
        # --- 【!! 核心：定义十进制字段 !!】 ---
        # (这假设你的 YAML 中 tcp.stream 的 type 是 'numerical')
        self.decimal_fields = {'tcp.stream'} 
        
        # --- [!! 注入结束 !!] ---

        # --- 2. 【核心修改点】定义“专家”及其“视野” (Schema) --- 
        all_available_fields = set(dataframe.columns)

        ip_fields = {f for f in all_available_fields if f.startswith('ip.')}

        # 【!! 核心修复：移除 IP 噪声 !!】
        if not use_ip_address: 
            ip_fields_to_ignore = {'ip.src', 'ip.dst'} 
        else: 
            ip_fields_to_ignore = {} 
        ip_fields_cleaned = ip_fields - ip_fields_to_ignore
        
        # 您可以根据需要，像配置表一样，精确地定义这些专家的字段
        self.expert_definitions = {
            'eth': {f for f in all_available_fields if f.startswith('eth.')},
            # 'ip': {f for f in all_available_fields if f.startswith('ip.')},
            'ip': ip_fields_cleaned, 
            'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f},
            'tcp_options': {f for f in all_available_fields if f.startswith('tcp.options.')},
            'tls_record': {f for f in all_available_fields if f.startswith('tls.record.')},
            'tls_handshake': {f for f in all_available_fields if f.startswith('tls.handshake.')},
            'tls_x509': {f for f in all_available_fields if f.startswith('tls.x509')}
            # ... 您可以根据需要，定义任意多的“专家”
        }
        self.flow_feature_names = ['flow_avg_len', 'flow_std_len', 'flow_pkt_count']

        # --- 3. 【核心修改点】为每个“专家”预先生成图结构 ---
        print("Pre-calculating graph structures for each expert...")
        self.expert_graphs = {}
        # 收集所有需要从DataFrame中读取的字段
        self.all_feature_cols_to_process = set() 
        
        for name, expert_fields in self.expert_definitions.items():
            # 找到这个专家Schema中，实际存在于DataFrame中的字段
            real_nodes_for_expert = sorted(list(expert_fields.intersection(all_available_fields)))
            if not real_nodes_for_expert:
                print(f" -> 警告: 专家 '{name}' 在数据集中没有任何对应的字段，将跳过。")
                continue

            # 为这个专家的“小世界”构建协议树
            ptree = protocol_tree(real_nodes_for_expert)
            add_root_layer(ptree)
            
            # 找到所有真实节点和抽象节点
            ptree_nodes = set(ptree.keys())
            for children in ptree.values(): 
                ptree_nodes.update(children)
            
            all_nodes_for_expert = sorted(list(ptree_nodes))
            field_to_node_idx = {n: i for i, n in enumerate(all_nodes_for_expert)}
            
            # 存储这个专家的所有信息
            self.expert_graphs[name] = {
                'real_nodes': real_nodes_for_expert, # 真实存在的字段
                'all_nodes': all_nodes_for_expert,  # 真实+抽象节点
                'field_to_node_idx': field_to_node_idx,
                'edge_index': self._create_edge_index_from_tree(ptree, field_to_node_idx)
            }
            # 将这些字段加入到“总处理列表”
            self.all_feature_cols_to_process.update(real_nodes_for_expert)

        if self.use_flow_features: 
            print("Flow features enabled. Adding to processing list.")
            self.all_feature_cols_to_process.update(self.flow_feature_names)
        else:
            print("Flow features disabled.")

        # --- 4. 【性能飞跃】对整个DataFrame进行一次性预处理 ---
        print(f"Pre-processing all {len(self.all_feature_cols_to_process)} columns...")
        self.processed_df = self._preprocess_all(dataframe)
        print("Dataset initialization complete.")

        # 调用我们的一键转换
        self._preload_to_tensors()
        print("Dataset initialization complete.")

        # # ==================== 【!! 核心性能修复 !!】 ====================
        # print("\nPre-caching all items into RAM. This may take several minutes...")
        # # 
        # # 我们将__init__中的所有重活只做一次。
        # #
        # self.data_cache = []
        # for idx in tqdm(range(len(self.labels)), desc="Caching data items"):
        #     # 调用我们刚刚重命名的 "重" 函数
        #     self.data_cache.append(self._create_data_dict(idx))

        # print(f"Caching complete. {len(self.data_cache)} items loaded into RAM.")
        # # ====================================================================

        
    def _create_edge_index_from_tree(self, ptree: Dict[str, List[str]], field_to_node_idx: Dict[str, int]) -> torch.Tensor:
        """【新】辅助函数：根据传入的ptree和idx映射，创建边索引。"""
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
        【新】核心性能优化：
        在__init__中，对整个DataFrame的所有列，进行一次性的向量化预处理。

        """
        def bin_port_from_hex(x): # 【!!】这是你的“外挂”分箱函数
            if not pd.notna(x): return 0 # Bin 0
            try:
                port_int = int(str(x).split('.')[0], 16) 
                if 1 <= port_int <= 1023: return 1 # Bin 1
                elif 1024 <= port_int <= 49151: return 2 # Bin 2
                elif port_int >= 49152: return 3 # Bin 3
                else: return 0 # Bin 0
            except ValueError: return 0

        def robust_hex_to_int(x):
            if not pd.notna(x): return 0
            try: return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
            except ValueError: return 0

        def robust_timestamp_to_tsval(x):
            if not pd.notna(x): return 0
            try:
                s = str(x).lower().replace('0x', '')
                if len(s) != 20 or not s.startswith('080a'): return 0
                tsval_hex = s[4:12]
                return int(tsval_hex, 16)
            except (ValueError, TypeError): return 0
        processed_data_dict = {}
        
        # 筛选出我们需要处理的、且存在于df中的列
        cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
        for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
            # 1. 流统计特征 (Flow Stats)
            if field_name in self.flow_feature_names:
                processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
                continue
                
            # 2. 包头特征 (GNN Node Features)
            if field_name not in self.config:
                continue # 跳过未配置的字段

            config = self.config[field_name]
            col_data = df[field_name]
            field_type = config['type']

            if field_type in ['address_ipv4', 'address_mac']:
                processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
            elif field_type == 'categorical':
                # 【!! 核心修复：劫持端口字段 !!】
                if field_name in self.port_fields:
                    # a. Hex Port -> Bin ID (0, 1, 2, 3)
                    binned_data = col_data.apply(bin_port_from_hex)
                    # b. Map Bin ID -> Embedding Index (使用注入的 {0:0, 1:1, ...})
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = 0 # (0 映射到 0)
                    processed_data_dict[field_name] = binned_data.map(vocab_map).fillna(oov_index).astype(np.int32)
                
                # 【!!】*其他* 分类字段
                elif field_name in self.vocab_maps:
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = vocab_map.get('__OOV__', 0)
                    processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
                else:
                    pass # (跳过高基数ID，例如 serialNumber)

            elif field_type == 'numerical':
                if field_name in self.decimal_fields: # (例如 tcp.stream)
                    processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
                elif field_name == 'tcp.options.timestamp': # (例如 080a...)
                    # processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).astype(np.int64)
                    # 【!! 核心修复 !!】
                    # 1. 转换为大整数
                    # 2. 应用 np.log1p (log(x+1)) 来压缩数值范围
                    # 3. 转换为 float32
                    processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).apply(np.log1p).astype(np.float32)
                else: # (默认: ip.len, tcp.len, tsval, tsecr ...)
                    processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
            else:
                 print(f"警告: 字段 '{field_name}' 的类型 '{field_type}' 无法识别。将跳过。")
            
            
        return pd.DataFrame(processed_data_dict)

    # def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     【新】核心性能优化：
    #     在__init__中，对整个DataFrame的所有列，进行一次性的向量化预处理。
    #     """
    #     processed_data_dict = {}
        
    #     # 筛选出我们需要处理的、且存在于df中的列
    #     cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
    #     for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
    #         # 1. 流统计特征 (Flow Stats)
    #         if field_name in self.flow_feature_names:
    #             processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
    #             continue
                
    #         # 2. 包头特征 (GNN Node Features)
    #         if field_name not in self.config:
    #             continue # 跳过未配置的字段
            
    #         config = self.config[field_name]
    #         col_data = df[field_name]

    #         # 应用您之前验证过的、健壮的if/elif逻辑 (现在是向量化版本)
    #         if config['type'] in ['address_ipv4', 'address_mac']:
    #             field_type = config['type']
    #             processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
    #         elif field_name in self.vocab_maps:
    #             vocab_map = self.vocab_maps[field_name]
    #             oov_index = vocab_map.get('__OOV__', 0) # 假设OOV=0
    #             # processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(int)
    #             processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
            
    #         # elif field_name == 'tcp.stream': # 您之前的 decimal_fields
    #         #     processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
    #         # 【修改】使用 self.decimal_fields 变量
    #         elif field_name in self.decimal_fields: 
    #             # processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
    #             processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
            
    #         elif config['type'] in ['categorical', 'numerical']:
    #             def robust_hex_to_int(x):
    #                 if not pd.notna(x): return 0
    #                 try:
    #                     return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
    #                 except ValueError:
    #                     return 0
    #             # processed_data_dict[field_name] = col_data.apply(robust_hex_to_int)
    #             processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
    #     return pd.DataFrame(processed_data_dict)

    def __len__(self):
        return len(self.labels)


    def _preload_to_tensors(self):
        """
        【新方法】将 Pandas DataFrame 中的列直接转换为紧凑的 PyTorch Tensors。
        这将极大减少 __getitem__ 的 CPU 开销，同时内存占用很低。
        """
        print("Converting dataframe columns to PyTorch tensors (Vectorization)...")
        self.tensor_cache = {} # 结构: {expert_name: {field_name: Tensor}}
        num_samples = len(self.processed_df)
        
        for expert_name, graph_info in self.expert_graphs.items():
            self.tensor_cache[expert_name] = {}
            all_nodes_list = graph_info['all_nodes']
            real_nodes_set = set(graph_info['real_nodes'])
            
            for field_name in all_nodes_list:
                # 1. 确定目标 dtype 和 shape
                config = self.config.get(field_name, {})
                config_type = config.get('type', 'categorical')
                
                # --- 情况 A: 真实存在的节点 ---
                if field_name in real_nodes_set and field_name in self.processed_df.columns:
                    col_data = self.processed_df[field_name]
                    
                    if config_type in ['address_ipv4', 'address_mac']:
                        # 地址类型是 list，需要堆叠
                        # 将 Series of Lists 转换为 Tensor [N, 4] or [N, 6]
                        # 这种写法比手动循环快得多
                        try:
                            # 假设 col_data 里的元素已经是 list 了
                            data_tensor = torch.tensor(col_data.tolist(), dtype=torch.long)
                        except:
                            # 如果数据不干净，回退到全0
                            dim = 4 if config_type == 'address_ipv4' else 6
                            data_tensor = torch.zeros((num_samples, dim), dtype=torch.long)

                    elif config_type == 'numerical':
                        # 数值类型 -> float32 [N]
                        values = col_data.fillna(0).values.astype(np.float32)
                        data_tensor = torch.from_numpy(values)
                        
                    else:
                        # 分类类型 -> long [N]
                        values = col_data.fillna(0).values.astype(np.int64)
                        data_tensor = torch.from_numpy(values)
                
                # --- 情况 B: 抽象节点或缺失节点 (创建全0 Tensor) ---
                else:
                    if config_type == 'address_ipv4':
                        data_tensor = torch.zeros((num_samples, 4), dtype=torch.long)
                    elif config_type == 'address_mac':
                        data_tensor = torch.zeros((num_samples, 6), dtype=torch.long)
                    elif config_type == 'numerical':
                        data_tensor = torch.zeros((num_samples,), dtype=torch.float32)
                    else:
                        data_tensor = torch.zeros((num_samples,), dtype=torch.long)

                # 存入缓存
                self.tensor_cache[expert_name][field_name] = data_tensor
        
        # 还要预加载 Flow Features (如果有)
        if self.use_flow_features:
            self.flow_tensor_cache = {}
            for f in self.flow_feature_names:
                col_data = self.processed_df[f]
                values = col_data.fillna(0).values.astype(np.float32)
                self.flow_tensor_cache[f] = torch.from_numpy(values)
                
        print("Vectorization complete. DataFrame memory can be released.")
        # 可选：释放原始 DataFrame 以节省内存 (我们现在只依赖 Tensor 了)
        del self.processed_df 

    # def __getitem__(self, idx) -> Dict[str, Any]:
    # # def _create_data_dict(self, idx) -> Dict[str, Any]:
    #     """
    #     【新 - 已修正 KeyError 和 形状问题】
    #     【再修正 - 确保抽象节点也具有特征属性】
    #     从“已预处理”的DataFrame中，快速切片，并组装成一个“专家图字典”。
    #     """
    #     processed_row = self.processed_df.iloc[idx]
    #     y = self.labels[idx].view(1) # 保持 [1] 形状
        
    #     data_dict = {}

    #     for expert_name, graph_info in self.expert_graphs.items():
            
    #         feature_dict_for_expert = {}
    #         all_nodes_list = graph_info['all_nodes']
    #         real_nodes_set = set(graph_info['real_nodes']) # 转换为集合以便快速查找
            
    #         # --- 核心修改点：遍历 *所有* 节点，而不仅仅是 real_nodes ---
    #         for field_name in all_nodes_list:
                
    #             value = None
    #             is_real_node = field_name in real_nodes_set
                
    #             # 1. 如果是真实节点，尝试从行数据中获取值
    #             if is_real_node:
    #                 value = processed_row.get(field_name) # value 可能是真实值，也可能是 None
                
    #             # 2. value 现在的情况:
    #             #    a) 真实值 (list, int, float)
    #             #    b) None (因为是抽象节点，或 真实节点但数据缺失)
                
    #             tensor_value = None

    #             # 1. 首先获取该字段的配置类型
    #             config = self.config.get(field_name, {})
    #             config_type = config.get('type', 'categorical') # 默认按 categorical 处理
                
    #             if isinstance(value, list):
    #                 # --- 针对地址类型 (list) ---
    #                 tensor_value = torch.tensor(value, dtype=torch.long).view(1, -1)
                
    #             elif isinstance(value, (int, float, np.number)):
    #                 # --- 针对分类/数值类型 (int/float) ---
    #                 # tensor_value = torch.tensor([value], dtype=torch.long)
    #                 if config_type == 'numerical':
    #                     # 【修复】数值型 -> float32
    #                     tensor_value = torch.tensor([value], dtype=torch.float32)
    #                 else:
    #                     # 分类型 -> long
    #                     tensor_value = torch.tensor([value], dtype=torch.long)
                
    #             else:
    #                 # --- 针对缺失值 (None) 或 抽象节点 ---
    #                 # 检查配置，看它 *应该* 是什么形状
    #                 # config_type = self.config.get(field_name, {}).get('type', 'numerical')
                    
    #                 if config_type == 'address_ipv4':
    #                     tensor_value = torch.zeros((1, 4), dtype=torch.long)
    #                 elif config_type == 'address_mac':
    #                     tensor_value = torch.zeros((1, 6), dtype=torch.long)
    #                 elif config_type == 'numerical':
    #                     # 【修复】数值型的占位符也必须是 float
    #                     tensor_value = torch.tensor([0.0], dtype=torch.float32)
    #                 else:
    #                     # 抽象节点 (如 'ip', 'ROOT') 和 缺失的真实节点 (如 'tcp.flags')
    #                     # 都会得到一个 tensor([0]) 作为占位符
    #                     tensor_value = torch.tensor([0], dtype=torch.long)

    #             feature_dict_for_expert[field_name] = tensor_value
    #             # =================================================================

    #         graph_data = Data(
    #             edge_index=graph_info['edge_index'],
    #             y=y,
    #             num_nodes=len(all_nodes_list), # 使用 all_nodes_list 的长度
    #             **feature_dict_for_expert
    #         )
    #         data_dict[expert_name] = graph_data

    #     if self.use_flow_features:
    #         flow_stats_list = [processed_row.get(f, 0.0) for f in self.flow_feature_names]
    #         data_dict['flow_stats'] = torch.tensor(flow_stats_list, dtype=torch.float).view(1, -1)
        
    #     return data_dict

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        【极速版】直接从预生成的 Tensors 中切片，无需 Pandas 参与。
        """
        # 1. 标签 (Label)
        y = self.labels[idx].view(1)

        data_dict = {}

        # 2. 遍历专家，组装 Data 对象
        for expert_name, graph_info in self.expert_graphs.items():
            feature_dict_for_expert = {}
            all_nodes_list = graph_info['all_nodes']
            
            # 从 tensor_cache 中直接拿数据！速度飞快！
            expert_tensors = self.tensor_cache[expert_name]
            
            for field_name in all_nodes_list:
                # 切片操作：tensor[idx]
                raw_val = expert_tensors[field_name][idx]
                
                # 调整维度：
                # 如果是 IP/MAC [4] -> [1, 4]，如果是标量 [] -> [1]
                if raw_val.dim() == 0:
                    tensor_value = raw_val.view(1)
                else:
                    tensor_value = raw_val.unsqueeze(0)
                
                feature_dict_for_expert[field_name] = tensor_value

            # 创建 Data 对象 (这个开销很小，因为数据已经准备好了)
            graph_data = Data(
                edge_index=graph_info['edge_index'],
                y=y,
                num_nodes=len(all_nodes_list),
                **feature_dict_for_expert
            )
            data_dict[expert_name] = graph_data

        # 3. 流特征 (如果启用)
        if self.use_flow_features:
            flow_stats_list = []
            for f in self.flow_feature_names:
                flow_stats_list.append(self.flow_tensor_cache[f][idx])
            data_dict['flow_stats'] = torch.stack(flow_stats_list).view(1, -1)
        
        return data_dict
    
    # def __getitem__(self, idx) -> Dict[str, Any]:
    #     """
    #     【新】这个函数现在极度轻量。
    #     它只负责从内存缓存列表中索引。
    #     """
    #     return self.data_cache[idx]
