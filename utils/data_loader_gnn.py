import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data 
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree
from utils.data_loader import _preprocess_address
from collections import defaultdict
import torch.nn as nn
from typing import Dict, List

class GNNTrafficDataset(Dataset):
    """
    一个为GNN模型准备数据的Dataset。
    它的__getitem__方法将一个数据包（一行DataFrame）转换成一个PyG的图(Data)对象。
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, node_feature_dim: int=128):
        super().__init__()
        # 1. __init__只加载配置和数据，不创建任何nn.Module
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
        self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
        # 2. 动态确定节点和图结构
        available_fields = set(dataframe.columns)
        configured_fields = set(self.config.keys())
        self.node_fields = sorted(list(available_fields.intersection(configured_fields)))
        self.field_to_node_idx = {name: i for i, name in enumerate(self.node_fields)}
        
        ptree = protocol_tree(self.node_fields)
        self.edge_index = self._create_edge_index(ptree)
        
        # 3. 准备数据
        self.labels = dataframe['label_id'].values
        self.features_df = dataframe[self.node_fields].reset_index(drop=True)
        self.decimal_fields = {'tcp.stream'}

    # ==================== CORE FIX ====================
    # Add 'ptree' as an argument to the function definition
    def _create_edge_index(self, ptree: Dict[str, List[str]]) -> torch.Tensor:
    # ================================================
        """
        根据协议树，一次性地创建图的边索引张量。
        """
        edge_list = []
        # Use the passed-in 'ptree' dictionary, not self.protocol_tree
        for parent, children in ptree.items():
            if parent in self.field_to_node_idx:
                parent_idx = self.field_to_node_idx[parent]
                for child in children:
                    if child in self.field_to_node_idx:
                        child_idx = self.field_to_node_idx[child]
                        # Add edges in both directions for an undirected graph
                        edge_list.append([child_idx, parent_idx])
                        edge_list.append([parent_idx, child_idx])
        
        if not edge_list:
            # If there are no edges, return an empty 2x0 tensor
            return torch.empty((2, 0), dtype=torch.long)
            
        # .t() is transpose, .contiguous() ensures the memory layout is correct
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取第idx个数据包的原始特征行
        raw_row = self.features_df.iloc[idx]
        processed_features = {}
        for field_name in self.node_fields:
            value = raw_row[field_name]
            
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                processed_features[field_name] = _preprocess_address(value, self.config[field_name]['type'])
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', len(vocab_map))
                processed_features[field_name] = vocab_map.get(str(value).lower().replace('0x',''), oov_index) if pd.notna(value) else oov_index
            elif field_name in self.decimal_fields:
                processed_features[field_name] = int(value) if pd.notna(value) and str(value).isdigit() else 0
            elif self.config[field_name]['type'] in ['categorical', 'numerical']:
                if not pd.notna(value):
                    processed_features[field_name] = 0
                else:
                    str_x = str(value).split('.')[0]
                    try:
                        # 先尝试转换为 potentially huge python integer
                        huge_int = int(str_x, 16)
                        
                        # 检查是否溢出，如果溢出则截断并警告
                        if huge_int > self.TORCH_LONG_MAX:
                            # print(f"\nWarning: Overflow detected for field '{field_name}' with value '{str_x}'. "
                            #       f"Value will be capped. Consider treating this field as high-cardinality categorical.")
                            processed_features[field_name] = self.TORCH_LONG_MAX
                        else:
                            processed_features[field_name] = huge_int
                            
                    except ValueError:
                        processed_features[field_name] = 0
            else: 
                continue

        # --- b) 步骤二：构建图对象，节点特征为【整数索引】 ---
        # 我们将所有字段的索引值，按照node_fields的顺序排列
        node_indices_list = []
        for field_name in self.node_fields:
            # 地址是列表，其他是整数
            value = processed_features.get(field_name)
            if isinstance(value, list):
                # 对于地址，我们暂时取其第一个字节作为代表性索引
                node_indices_list.append(value[0])
            else:
                node_indices_list.append(value)

        # x 张量现在是整数索引，形状为 [num_nodes, 1]
        x = torch.tensor(node_indices_list, dtype=torch.long).unsqueeze(1)
        y = self.labels[idx]
        
        graph_data = Data(x=x, edge_index=self.edge_index, y=y)
        # 我们还可以把完整的特征字典附加到图对象上，供模型内部使用
        graph_data.feature_dict = processed_features
        
        return graph_data