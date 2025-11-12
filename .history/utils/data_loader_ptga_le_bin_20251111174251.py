import pandas as pd
import torch
import yaml
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, Any, List
from utils.dataframe_tools import protocol_tree, add_root_layer
from utils.data_loader import _preprocess_address

# (确保你的辅助函数在这里被定义或导入)
# from your_utils_file import protocol_tree, add_root_layer, _preprocess_address

class GNNTrafficDataset(Dataset):
    """
    【!! MoE 最终版 V3：Tensor Cache + 端口分箱 + 字典输出 !!】
    
    为 HierarchicalMoE (多视角)模型准备数据。
    - __init__: 构建 *每个专家* 的图骨架 (self.expert_graphs)
    - __init__: 使用 V3 (YAML驱动) 的 _preprocess_all 进行预处理
    - __init__: 将预处理后的 df 转换为多进程安全的 "Tensor Cache"
    - __getitem__: 从 Cache 中高速切片，返回一个 *专家字典* (Dict[str, Data])
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, 
                 use_flow_features: bool = False):
        super().__init__()
        print(f"\nInitializing Hierarchical GNNTrafficDataset (Tensor Cache, Port Binning, Flow: {use_flow_features})...")
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
        
        # --- 【!! 核心：注入端口分箱逻辑 !!】 ---
        # (这假设你的 YAML 中 tcp.srcport/dstport 的 type 是 'port_binned')
        self.port_fields = {'tcp.srcport', 'tcp.dstport'}
        
        # --- 【!! 核心：定义十进制字段 !!】 ---
        # (这假设你的 YAML 中 tcp.stream 的 type 是 'numerical')
        self.decimal_fields = {'tcp.stream'} 
        
        # 2. 【MoE 架构】定义“专家”及其“视野”
        all_available_fields = set(dataframe.columns)
        self.expert_definitions = {
            'eth': {f for f in all_available_fields if f.startswith('eth.')},
            'ip': {f for f in all_available_fields if f.startswith('ip.')},
            'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f},
            'tcp_options': {f for f in all_available_fields if f.startswith('tcp.options.')},
            'tls_record': {f for f in all_available_fields if f.startswith('tls.record.')},
            'tls_handshake': {f for f in all_available_fields if f.startswith('tls.handshake.')},
            'tls_x509': {f for f in all_available_fields if f.startswith('tls.x509')}
        }
        
        # 3. 【MoE 架构】为 *每个* 专家预先生成图结构
        print("Pre-calculating graph structures for *each expert*...")
        self.expert_graphs = {}
        self.all_feature_cols_to_process = set() 
        
        for name, expert_fields in self.expert_definitions.items():
            real_nodes_for_expert = sorted(list(expert_fields.intersection(all_available_fields)))
            if not real_nodes_for_expert and name != 'tls_x509': # (允许 tls_x509 为空)
                print(f" -> 警告: 专家 '{name}' 在数据集中没有任何对应的字段，将跳过。")
                continue

            # 【!! 关键 !!】使用你的 "统一图" 版 add_root_layer
            ptree = protocol_tree(real_nodes_for_expert)
            add_root_layer(ptree) # (例如 "UNIFIED" 版本)
            
            ptree_nodes = set(ptree.keys())
            for children in ptree.values(): 
                ptree_nodes.update(children)
            
            all_nodes_for_expert = sorted(list(ptree_nodes))
            field_to_node_idx = {n: i for i, n in enumerate(all_nodes_for_expert)}
            
            self.expert_graphs[name] = {
                'real_nodes': set(real_nodes_for_expert), # <-- 使用 set
                'all_nodes': all_nodes_for_expert,
                'field_to_node_idx': field_to_node_idx,
                'edge_index': self._create_edge_index_from_tree(ptree, field_to_node_idx)
            }
            # 更新 *所有* 需要处理的真实 GNN 节点
            self.all_feature_cols_to_process.update(real_nodes_for_expert)

        if self.use_flow_features: 
            print("Flow features enabled. Adding to processing list.")
            self.all_feature_cols_to_process.update(self.flow_feature_names)
        else:
            print("Flow features disabled.")
            
        # 4. 预处理 (使用 V3 函数)
        print(f"Pre-processing all {len(self.all_feature_cols_to_process)} columns...")
        # (我们必须传入 *所有* 需要的列，包括 GNN 和 Flow)
        processed_df = self._preprocess_all(dataframe, self.all_feature_cols_to_process)
        
        # 5. 【!! 核心：转换为 Tensor Cache !!】
        print("Converting processed DataFrame to tensor cache...")
        self.tensor_cache = {}
        cols_in_df = set(processed_df.columns)
        
        for col_name in tqdm(self.all_feature_cols_to_process, desc="Creating tensor cache"):
            if col_name not in cols_in_df: continue
            col_data = processed_df[col_name]
            
            first_val = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
            
            if isinstance(first_val, (list, tuple)): # 地址
                length = 4 if 'ip' in col_name else (6 if 'mac' in col_name else 4)
                def fill_empty_addr(x, length=length):
                    if not isinstance(x, (list, tuple)) or len(x) == 0: return [0] * length
                    return x
                data_list = col_data.apply(fill_empty_addr).tolist()
                self.tensor_cache[col_name] = torch.tensor(data_list, dtype=torch.long)
            elif col_name in self.flow_feature_names: # 流特征
                self.tensor_cache[col_name] = torch.tensor(col_data.values, dtype=torch.float)
            else: # GNN 特征 (Categorical, Numerical, Binned)
                self.tensor_cache[col_name] = torch.tensor(col_data.values, dtype=torch.long) # (int64/int32 统一为 long)

        # 【!! 关键 !!】我们不再需要 processed_df，释放它！
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

    
    def _preprocess_all(self, df: pd.DataFrame, cols_to_process: set) -> pd.DataFrame:
        """
        【!! 最终修复版 V3 (YAML 驱动 + 端口分箱) !!】
        由 YAML 配置文件中的 'type' 驱动。
        """
        processed_data_dict = {}
        
        # --- 1. 在函数内部定义所有辅助转换器 ---
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
        def bin_port_from_hex(x):
            if not pd.notna(x): return 0 # Bin 0
            try:
                port_int = int(str(x).split('.')[0], 16) 
                if 1 <= port_int <= 1023: return 1 # Bin 1
                elif 1024 <= port_int <= 49151: return 2 # Bin 2
                elif port_int >= 49152: return 3 # Bin 3
                else: return 0 # Bin 0
            except ValueError: return 0
        # --- 辅助函数定义结束 ---
        
        cols_in_df = set(df.columns)
        for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
            
            if field_name not in cols_in_df:
                continue
            
            # --- 分支 1: 流统计特征 (Flow Stats) ---
            if field_name in self.flow_feature_names:
                col_data_numeric = pd.to_numeric(df[field_name], errors='coerce')
                col_data_np = col_data_numeric.values
                col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0) 
                processed_data_dict[field_name] = col_data_cleaned.astype(np.float32)
                continue
                
            # --- 分支 2: GNN 包头特征 ---
            if field_name not in self.config: continue 
            
            config = self.config[field_name]
            col_data = df[field_name]
            field_type = config['type']

            # --- 按类型分派 (YAML 驱动) ---
            
            if field_type in ['address_ipv4', 'address_mac']:
                processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
            elif field_type == 'port_binned':
                binned_data = col_data.apply(bin_port_from_hex)
                processed_data_dict[field_name] = binned_data.astype(np.int32)
            
            elif field_type == 'categorical':
                if field_name in self.vocab_maps:
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = vocab_map.get('__OOV__', 0)
                    processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
                else:
                    pass 

            elif field_type == 'numerical':
                if field_name in self.decimal_fields: # (例如 tcp.stream)
                    processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
                elif field_name == 'tcp.options.timestamp': # (例如 080a...)
                    processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).astype(np.int64)
                else: # (默认: ip.len, tcp.len, tsval, tsecr ...)
                    processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
            else:
                 print(f"警告: 字段 '{field_name}' 的类型 '{field_type}' 无法识别。将跳过。")

        return pd.DataFrame(processed_data_dict)

    
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        【MoE 版 - Tensor Cache - 返回字典】
        从“已预处理”的Cache中，快速切片，并组装成一个“专家图字典”。
        """
        y = self.labels[idx].view(1) # [1] 形状
        
        data_dict = {}

        for expert_name, graph_info in self.expert_graphs.items():
            
            feature_dict_for_expert = {}
            all_nodes_list = graph_info['all_nodes']
            real_nodes_set = graph_info['real_nodes']
            
            for field_name in all_nodes_list:
                
                tensor_value = None
                is_real_node = field_name in real_nodes_set
                
                # 1. 如果是真实节点，尝试从 Cache 中获取值
                if is_real_node and field_name in self.tensor_cache:
                    tensor_value = self.tensor_cache[field_name][idx]
                    
                    # 确保形状 (切片后 1D/0D -> 2D/1D)
                    if tensor_value.dim() == 0:
                        tensor_value = tensor_value.unsqueeze(0) 
                    elif tensor_value.dim() == 1:
                         config_type = self.config.get(field_name, {}).get('type', 'numerical')
                         # (地址 和 非十进制数值 才需要 unsqueeze)
                         if 'address' in config_type or (field_name not in self.decimal_fields and field_name != 'tcp.options.timestamp'):
                            tensor_value = tensor_value.unsqueeze(0)
                    
                    feature_dict_for_expert[field_name] = tensor_value.clone()
                
                else:
                    # 2. 抽象节点 (或 真实但缺失的 - 例如 tls_x509)
                    config_type = self.config.get(field_name, {}).get('type', 'numerical')
                    if config_type == 'address_ipv4':
                        tensor_value = torch.zeros((1, 4), dtype=torch.long)
                    elif config_type == 'address_mac':
                        tensor_value = torch.zeros((1, 6), dtype=torch.long)
                    else:
                        tensor_value = torch.tensor([0], dtype=torch.long)
                    feature_dict_for_expert[field_name] = tensor_value

            graph_data = Data(
                edge_index=graph_info['edge_index'],
                y=y,
                num_nodes=len(all_nodes_list),
                **feature_dict_for_expert
            )
            data_dict[expert_name] = graph_data

        if self.use_flow_features:
            flow_stats_list = [self.tensor_cache[f][idx] for f in self.flow_feature_names]
            data_dict['flow_stats'] = torch.stack(flow_stats_list).view(1, -1)
        
        return data_dict