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
from torch_geometric.data import Batch

def gnn_collate_fn(batch: List[Data]):
    """
    一个为PTGA模型定制的、健壮的collate_fn。
    它能正确地将包含原生Python类型的图对象批处理成一个Batch对象。
    """
    # 1. 从批次中分离出图的基本结构和我们自定义的特征
    base_data_list = []
    feature_dicts = []
    for data in batch:
        # 提取特征字典
        features = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes']}
        feature_dicts.append(features)
        
        # 创建一个只包含基本信息的“空壳”图对象
        base_data = Data(edge_index=data.edge_index, y=data.y, num_nodes=data.num_nodes)
        base_data_list.append(base_data)
        
    # 2. 使用PyG的默认方式，高效地合并图结构和标签
    batched_graph = Batch.from_data_list(base_data_list)
    
    # 3. 手动地、正确地将我们的特征字典列表，打包成Tensor字典
    field_names = feature_dicts[0].keys()
    batched_features = {}
    for field in field_names:
        values = [sample[field] for sample in feature_dicts]
        # 这个torch.tensor调用现在可以正确处理整数和整数列表
        try:
             batched_features[field] = torch.tensor(values, dtype=torch.long)
        except ValueError: # 处理变长列表的情况 (虽然不应该发生)
             batched_features[field] = [torch.tensor(v, dtype=torch.long) for v in values]
    
    # 4. 将打包好的特征Tensor字典，作为新属性附加到最终的批处理对象上
    batched_graph.feature_dict = batched_features
    
    return batched_graph

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
        self.decimal_fields = {'tcp.stream'}
        # 2. 动态确定节点和图结构
        available_fields = set(dataframe.columns)
        configured_fields = set(self.config.keys())
        self.node_fields = sorted(list(available_fields.intersection(configured_fields)))
        self.field_to_node_idx = {name: i for i, name in enumerate(self.node_fields)}
        
        ptree = protocol_tree(self.node_fields)
        self.edge_index = self._create_edge_index(ptree)
        
        # 3. 准备数据
        # self.labels = dataframe['label_id'].values
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        # 在__init__中进行一次性的、高效的列式预处理
        self.features_df = self._preprocess_dataframe_to_int(
            dataframe[self.node_fields].reset_index(drop=True)
        )
        

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
            return torch.empty((2, 0), dtype=torch.long) # 这样设计可能使图变成不连通的若干小图            
        # .t() is transpose, .contiguous() ensures the memory layout is correct
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _preprocess_dataframe_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        这个函数正是您询问的、经过修改的函数。
        它接收一个原始的DataFrame，返回一个完全由整数或列表组成的DataFrame。
        """
        processed_data_dict = {}
        TORCH_LONG_MAX = torch.iinfo(torch.long).max

        for field_name in tqdm(df.columns, desc="Pre-processing columns to integers"):
            if field_name not in self.config:
                continue
            
            # 使用您之前版本中经过验证的、健壮的if/elif逻辑
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                field_type = self.config[field_name]['type']
                # .apply的结果是一个Series，其每个元素都是一个list
                processed_column = df[field_name].apply(lambda x: _preprocess_address(x, field_type))
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', len(vocab_map))
                processed_column = df[field_name].apply(
                    lambda x: vocab_map.get(str(x).lower().replace('0x',''), oov_index) if pd.notna(x) else oov_index
                )
            elif field_name in self.decimal_fields:
                processed_column = pd.to_numeric(df[field_name], errors='coerce').fillna(0).astype(int)
            elif self.config[field_name]['type'] in ['categorical', 'numerical']:
                def robust_hex_to_int_with_cap(x):
                    if not pd.notna(x): return 0
                    str_x = str(x).split('.')[0]
                    try:
                        huge_int = int(str_x, 16)
                        # 返回截断后的值，以防止后续溢出
                        return min(huge_int, TORCH_LONG_MAX)
                    except ValueError:
                        return 0
                processed_column = df[field_name].apply(robust_hex_to_int_with_cap)
            else:
                continue
            
            processed_data_dict[field_name] = processed_column
            
        return pd.DataFrame(processed_data_dict)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # a) 从预处理好的DataFrame中获取一行整数索引
        processed_row = self.features_df.iloc[idx]
        
        # b) 将这行数据（一个Pandas Series）转换为一个Python字典
        feature_dict = processed_row.to_dict()
        
        y = self.labels[idx]
        
        # c) 创建图对象，使用 **字典解包** 将特征字典作为【独立属性】附加
        graph_data = Data(edge_index=self.edge_index, y=y, **feature_dict)
        
        # d) 明确地设置节点数
        graph_data.num_nodes = len(self.node_fields)
        
        return graph_data
        # # a) 从预处理好的DataFrame中获取一行整数索引
        # processed_row = self.features_df.iloc[idx]
        
        # # # b) 将这行数据（一个Pandas Series）转换为一个Python字典
        # # feature_dict = processed_row.to_dict()
        
        # # y = self.labels[idx]
        
        # # # c) 创建图对象，将特征字典作为【独立属性】附加
        # # graph_data = Data(edge_index=self.edge_index, y=y, **feature_dict)
        # x = torch.tensor(processed_row.values, dtype=torch.long).unsqueeze(1)
        # y = self.labels[idx]
        
        # graph_data = Data(x=x, edge_index=self.edge_index, y=y)
        
        # # ==================== 核心修改点 开始 ====================
        # # d) 明确地告诉PyG，这个图有多少个节点
        # graph_data.num_nodes = len(self.node_fields)
        # # ==================== 核心修改点 结束 ====================
        
        # return graph_data