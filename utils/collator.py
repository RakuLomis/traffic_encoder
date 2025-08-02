import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict
from models.FieldEmbedding import FieldEmbedding, _AddressEmbedding

class PTACollator:
    """
    一个智能的整理函数 (Collate Function)，用于PTA模型。
    它的核心任务是将一批动态的、结构不一的数据样本，转换成
    一个包含规整的、填充过的、并带有掩码的批处理张量的字典，
    以便PTA模型可以进行高效的、完全向量化的前向传播。
    """
    def __init__(self, field_embedder, protocol_tree: Dict[str, List[str]], 
                 subfield_aligned_dim: int = 64, aligned_dim: int = 128):
        """
        初始化Collator。
        
        :param field_embedder: 已经实例化的FieldEmbedding模块，我们将用它来进行嵌入查找。
        :param protocol_tree: 描述协议层次结构的字典。
        :param subfield_aligned_dim: 在阶段一中，所有子字段将被对齐到的统一维度。
        :param aligned_dim: 在阶段二中，所有字段将被对齐到的统一维度。
        """
        self.field_embedder = field_embedder
        self.protocol_tree = protocol_tree
        self.subfield_aligned_dim = subfield_aligned_dim
        self.aligned_dim = aligned_dim

        # --- 1. 预计算最大尺寸和映射，用于确定填充的维度 ---
        
        # a) 确定所有“父”字段 (有子字段的字段) 和 “简单”字段
        self.parent_fields = sorted([p for p, s in self.protocol_tree.items() if s and p in self.field_embedder.config])
        all_fields_in_tree = set()
        for fields in self.protocol_tree.values():
            all_fields_in_tree.update(fields)
        self.simple_fields = sorted([f for f in all_fields_in_tree if f not in self.parent_fields and f in self.field_embedder.config])
        
        # b) 协议层
        self.protocol_layers = sorted(['eth', 'ip', 'tcp', 'tls'])

        # ==================== 核心修改点 开始 ====================
        # 计算并保存 simple_fields_by_layer 属性
        self.simple_fields_by_layer = {}
        for p_name in self.protocol_layers:
            # 找到属于该协议层，并且是“简单字段”的那些字段
            fields_in_layer = [
                f for f in self.protocol_tree.get(p_name, []) 
                if f in self.simple_fields
            ]
            self.simple_fields_by_layer[p_name] = sorted(fields_in_layer)
        # ==================== 核心修改点 结束 ====================
        
        # c) 计算最大填充长度
        self.max_subfields = max(len([sf for sf in self.protocol_tree.get(p, []) if sf in self.field_embedder.config]) for p in self.parent_fields) if self.parent_fields else 0
        self.max_fields_per_layer = max(len([f for f in self.protocol_tree.get(p, []) if f in self.field_embedder.config]) for p in self.protocol_layers) if self.protocol_layers else 0

        # d) 创建从名称到索引位置的映射
        self.parent_to_idx = {name: i for i, name in enumerate(self.parent_fields)}
        self.protocol_to_idx = {name: i for i, name in enumerate(self.protocol_layers)}
        
        # --- 2. 在Collator内部创建所有需要的对齐层 ---
        self.subfield_aligners = self._create_subfield_aligners()
        self.field_aligners = self._create_field_aligners()
            
        print("PTACollator initialized successfully:")
        print(f" - Found {len(self.parent_fields)} parent fields. Max subfields: {self.max_subfields}")
        print(f" - Found {len(self.protocol_layers)} protocol layers. Max fields/layer: {self.max_fields_per_layer}")

    def _create_subfield_aligners(self):
        aligners = nn.ModuleDict()
        for parent_name in self.parent_fields:
            parent_key = parent_name.replace('.', '__')
            aligner_group = nn.ModuleDict()
            valid_subfields = [sf for sf in self.protocol_tree.get(parent_name, []) if sf in self.field_embedder.config]
            for sf_name in valid_subfields:
                start, end = self.field_embedder.embedding_slices[sf_name]
                original_dim = end - start
                if original_dim > 0:
                    aligner_group[sf_name.replace('.', '__')] = nn.Linear(original_dim, self.subfield_aligned_dim)
            aligners[parent_key] = aligner_group
        return aligners

    def _create_field_aligners(self):
        aligners = nn.ModuleDict()
        all_fields = self.parent_fields + self.simple_fields
        for field_name in all_fields:
            if field_name not in self.field_embedder.embedding_slices: continue
            
            input_dim_for_aligner = 0
            if field_name in self.parent_fields:
                input_dim_for_aligner = self.subfield_aligned_dim
            else:
                start, end = self.field_embedder.embedding_slices[field_name]
                input_dim_for_aligner = end - start
            
            if input_dim_for_aligner > 0:
                aligners[field_name.replace('.', '__')] = nn.Linear(input_dim_for_aligner, self.aligned_dim)
        return aligners

    def __call__(self, batch: List[tuple]) -> tuple:
        feature_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        batch_size = len(batch)
        
        embedded_vectors = self._embed_batch(feature_list)

        # --- 准备阶段一 (Subfield -> Field) 的输入 ---
        stage1_tensor = torch.zeros(batch_size, len(self.parent_fields), self.max_subfields, self.subfield_aligned_dim)
        stage1_mask = torch.ones(batch_size, len(self.parent_fields), self.max_subfields, dtype=torch.bool)
        
        for b_idx in range(batch_size):
            for p_name, p_idx in self.parent_to_idx.items():
                p_key = p_name.replace('.', '__')
                subfields = [sf for sf in self.protocol_tree.get(p_name, []) if sf in feature_list[b_idx]]
                for sf_idx, sf_name in enumerate(subfields):
                    sf_key = sf_name.replace('.', '__')
                    if sf_key in self.subfield_aligners[p_key]:
                        original_vec = embedded_vectors[b_idx][sf_name]
                        aligner = self.subfield_aligners[p_key][sf_key]
                        aligned_vec = aligner(original_vec)
                        stage1_tensor[b_idx, p_idx, sf_idx, :] = aligned_vec
                        stage1_mask[b_idx, p_idx, sf_idx] = False

        # --- 准备阶段二 (Field -> Protocol Layer) 的输入 ---
        # 这个阶段的输入是阶段一的输出(聚合后的父字段)和原始的简单字段
        # Collator只准备原始简单字段，模型将在内部组合它们
        simple_fields_by_layer = {p: [] for p in self.protocol_layers}
        simple_fields_tensors = {}
        simple_fields_masks = {}

        for p_name in self.protocol_layers:
            fields_in_layer = [f for f in self.protocol_tree.get(p_name, []) if f in self.simple_fields]
            simple_fields_by_layer[p_name] = fields_in_layer

            layer_tensor = torch.zeros(batch_size, len(fields_in_layer), self.aligned_dim)
            layer_mask = torch.ones(batch_size, len(fields_in_layer), dtype=torch.bool)

            for b_idx in range(batch_size):
                for f_idx, f_name in enumerate(fields_in_layer):
                    if f_name in embedded_vectors[b_idx]:
                        f_key = f_name.replace('.', '__')
                        if f_key in self.field_aligners:
                            original_vec = embedded_vectors[b_idx][f_name]
                            aligner = self.field_aligners[f_key]
                            aligned_vec = aligner(original_vec)
                            layer_tensor[b_idx, f_idx, :] = aligned_vec
                            layer_mask[b_idx, f_idx] = False
            
            simple_fields_tensors[p_name] = layer_tensor
            simple_fields_masks[p_name] = layer_mask

        batched_data = {
            "stage1_tensor": stage1_tensor,
            "stage1_mask": stage1_mask,
            "simple_fields_tensors": simple_fields_tensors,
            "simple_fields_masks": simple_fields_masks,
            "parent_fields_order": self.parent_fields,
            "simple_fields_by_layer": simple_fields_by_layer
        }
        
        return batched_data, labels

    def _embed_batch(self, feature_list: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        batch_embedded_vectors = []
        for sample in feature_list:
            embedded_sample = {}
            for field_name, value in sample.items():
                if field_name not in self.field_embedder.config: continue
                layer_key = self.field_embedder.field_to_key_map.get(field_name)
                if not layer_key or layer_key not in self.field_embedder.embedding_layers: continue
                
                layer = self.field_embedder.embedding_layers[layer_key]
                
                if isinstance(layer, nn.Embedding):
                    input_tensor = torch.tensor(value, dtype=torch.long)
                elif isinstance(layer, nn.Linear):
                    input_tensor = torch.tensor(value).view(-1, 1).float()
                # elif isinstance(layer, _AddressEmbedding):
                #     input_tensor = torch.tensor(value, dtype=torch.long)
                else:
                    continue # Should not happen
                
                embedded_sample[field_name] = layer(input_tensor)
            batch_embedded_vectors.append(embedded_sample)
        return batch_embedded_vectors