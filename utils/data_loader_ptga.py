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
from typing import Dict, List
from torch_geometric.data import Batch

# def gnn_collate_fn(batch: List[Data]):
#     """
#     一个为PTGA模型定制的、健壮的collate_fn。
#     它能正确地将包含原生Python类型的图对象批处理成一个Batch对象。
#     """
#     # 1. 从批次中分离出图的基本结构和我们自定义的特征
#     base_data_list = []
#     feature_dicts = []
#     for data in batch:
#         # 提取特征字典
#         features = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes']}
#         feature_dicts.append(features)
        
#         # 创建一个只包含基本信息的“空壳”图对象
#         base_data = Data(edge_index=data.edge_index, y=data.y, num_nodes=data.num_nodes)
#         base_data_list.append(base_data)
        
#     # 2. 使用PyG的默认方式，高效地合并图结构和标签
#     batched_graph = Batch.from_data_list(base_data_list)
    
#     # 3. 手动地、正确地将我们的特征字典列表，打包成Tensor字典
#     field_names = feature_dicts[0].keys()
#     batched_features = {}
#     for field in field_names:
#         values = [sample[field] for sample in feature_dicts]
#         # 这个torch.tensor调用现在可以正确处理整数和整数列表
#         try:
#              batched_features[field] = torch.tensor(values, dtype=torch.long)
#         except ValueError: # 处理变长列表的情况 (虽然不应该发生)
#              batched_features[field] = [torch.tensor(v, dtype=torch.long) for v in values]
    
#     # 4. 将打包好的特征Tensor字典，作为新属性附加到最终的批处理对象上
#     batched_graph.feature_dict = batched_features
    
#     return batched_graph

class GNNTrafficDataset(Dataset):
    """
    一个为GNN模型准备数据的Dataset。
    它的__getitem__方法将一个数据包（一行DataFrame）转换成一个PyG的图(Data)对象。
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, node_feature_dim: int=128):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
        self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
        self.decimal_fields = {'tcp.stream'}

        # 2. 动态确定节点和图结构
        available_fields = set(dataframe.columns)
        configured_fields = set(self.config.keys())
        self.real_nodes = sorted(list(available_fields.intersection(configured_fields)))
        self.ptree = protocol_tree(self.real_nodes)
        add_root_layer(self.ptree) # Now ptree contains root and different layers
        # ptree_nodes = set(self.ptree.keys()) | set(self.ptree.values()) 
        ptree_nodes = set(self.ptree.keys()) 
        for children in self.ptree.values(): 
            ptree_nodes.update(children)
        self.abstract_nodes = [node for node in ptree_nodes if node not in self.real_nodes]
        self.node_fields = sorted(list(set(self.real_nodes) | set(self.abstract_nodes)))
        self.field_to_node_idx = {name: i for i, name in enumerate(self.node_fields)}
        
        self.edge_index = self._create_edge_index(self.ptree)
        
        # 3. 准备数据
        # self.labels = dataframe['label_id'].values
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        # 在__init__中进行一次性的、高效的列式预处理
        # self.features_df = self._preprocess_dataframe_to_int(
        #     dataframe[self.node_fields].reset_index(drop=True)
        # ) 
        self.features_df = dataframe[self.real_nodes] # 只保留真实存在的列
        

    def _create_edge_index(self, ptree: Dict[str, List[str]]) -> torch.Tensor:
        """
        一个更通用的建边函数，能处理任意层级的抽象节点。
        """
        edge_list = []
        # 直接遍历由新函数生成的、完整的协议树
        for parent, children in ptree.items(): 
            # 只要父节点和子节点在我们的总节点列表中，就为它们建边
            if parent in self.field_to_node_idx:
                parent_idx = self.field_to_node_idx[parent]
                for child in children:
                    if child in self.field_to_node_idx:
                        child_idx = self.field_to_node_idx[child]
                        # 建立双向边
                        edge_list.append([child_idx, parent_idx])
                        edge_list.append([parent_idx, child_idx])
        if not edge_list: return torch.empty((2, 0), dtype=torch.long) # No graph has been established by ptree
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
        # __getitem__ 负责处理一个样本的所有逻辑
        # a) 从原始DataFrame中获取一行数据
        raw_row = self.features_df.iloc[idx]
        
        # ==================== 完整的IF/ELIF预处理逻辑 ====================
        # b) 对单行数据进行预处理，得到【整数索引】字典
        feature_dict = {}
        for field_name in self.real_nodes: # 只处理真实存在的节点
            value = raw_row.get(field_name) # 使用.get()更安全
            
            # 使用我们之前建立的、健壮的if/elif处理逻辑
            if self.config[field_name]['type'] in ['address_ipv4', 'address_mac']:
                feature_dict[field_name] = _preprocess_address(value, self.config[field_name]['type'])
            elif field_name in self.vocab_maps:
                vocab_map = self.vocab_maps[field_name]
                oov_index = vocab_map.get('__OOV__', len(vocab_map))
                feature_dict[field_name] = vocab_map.get(str(value).lower().replace('0x',''), oov_index) if pd.notna(value) else oov_index
            elif field_name in self.decimal_fields:
                feature_dict[field_name] = int(value) if pd.notna(value) and str(value).isdigit() else 0
            elif self.config[field_name]['type'] in ['categorical', 'numerical']:
                if not pd.notna(value):
                    feature_dict[field_name] = 0
                else:
                    str_x = str(value).split('.')[0]
                    try:
                        huge_int = int(str_x, 16)
                        # 返回截断后的值，以防止溢出
                        feature_dict[field_name] = min(huge_int, self.TORCH_LONG_MAX)
                    except ValueError:
                        feature_dict[field_name] = 0
            else:
                continue
        # =================================================================

        y = self.labels[idx]
        
        # c) 创建图对象，将【整数索引】作为独立属性附加
        #    抽象节点的特征将在模型内部处理
        graph_data = Data(edge_index=self.edge_index, y=y, **feature_dict)
        
        # d) 明确地设置节点数
        graph_data.num_nodes = len(self.node_fields)
        
        return graph_data