import torch 
import torch.nn as nn 
from typing import Dict, List, Tuple
from tqdm import tqdm

class AttentionAggregator(nn.Module): 
    def __init__(self, embed_dim, num_heads=4):
        """
        An Attention aggregator module to get the global information for add [CLS] in tensors. 

        Parameters 
        ---------- 
        embed_dim: 
            The dimension of input tensor (vector). 
        num_heads: 
            The number of headers in Multihead Attention model.  
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # inited with normal distribution
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
        self.layer_norm1 = nn.LayerNorm(embed_dim) 
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(), 
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None): 
        """
        Parameters 
        ---------- 
        x: 
            (batch_size, num_vectors, embed_dim) 

        Returns 
        ------- 
        cls_output: 
            (batch_size, embed_dim)
        """
        batch_size = x.shape[0] 
        # use broadcast mechanism to make cls_token's shape into (batch_size, 1, embed_dim). 
        # -1 means keeping same in this dimension
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # concat, shape (batch_size, num_vectors + 1, embed_dim) 
        x_with_cls = torch.cat([cls_tokens, x], dim=1) 

        # process and use mask 
        attention_mask = None 
        if mask is not None: 
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool) 
            attention_mask = torch.cat([cls_mask, mask], dim=1)

        # 将掩码传递给 key_padding_mask 参数
        attn_output, _ = self.attention(
            query=x_with_cls, 
            key=x_with_cls, 
            value=x_with_cls, 
            key_padding_mask=attention_mask
        )
        
        # --- 后续的Transformer块结构也稍作调整，使其更标准 ---
        x = self.layer_norm1(x_with_cls + attn_output)
        x = self.layer_norm2(x + self.ffn(x))

        # attn_output, _ = self.attention(x_with_cls, x_with_cls, x_with_cls) 
        # x = self.layer_norm(x_with_cls + attn_output) # residual 
        # x = self.layer_norm(x + self.ffn(x)) 

        cls_output = x[:, 0, :] 
        return cls_output 
    
# # ==================== 全新的、重构后的PTA模型 ====================

# class ProtocolTreeAttention(nn.Module):
#     """
#     一个经过重构的、高性能的PTA模型。
#     它的forward函数是完全向量化的，不包含任何Python原生循环。
#     """
#     def __init__(self,
#                  # 模型的核心超参数
#                  subfield_aligned_dim: int = 64,
#                  aligned_dim: int = 128,
#                  num_heads: int = 4,
#                  num_classes: int = 2):
#         """
#         初始化重构后的PTA模型。
#         这个模型不再需要知道具体的字段名，只关心输入的Tensor形状。
#         """
#         super().__init__()
        
#         # --- 1. 阶段一：Subfield -> Field 的聚合器 ---
#         # 这个聚合器将一次性处理所有父字段的子字段
#         self.subfield_aggregator = AttentionAggregator(subfield_aligned_dim, num_heads)
        
#         # --- 2. 阶段二：Field -> Protocol Layer 的聚合器 ---
#         # a) 需要一个对齐层，统一所有字段（聚合后的父字段+简单字段）的维度
#         #    它的输入维度是subfield_aligned_dim(用于聚合后的父字段)和aligned_dim(用于简单字段)中较大的那个
#         #    为了简化，我们假设Collator已经将所有简单字段也对齐到了subfield_aligned_dim
#         self.field_aligner = nn.Linear(subfield_aligned_dim, aligned_dim)
        
#         # b) 这个聚合器将一次性处理所有协议层的字段
#         self.protocol_layer_aggregator = AttentionAggregator(aligned_dim, num_heads)
        
#         # --- 3. 阶段三：Protocol Layer -> Packet 的聚合器 ---
#         self.final_aggregator = AttentionAggregator(aligned_dim, num_heads)
        
#         # --- 4. 分类头 ---
#         self.classifier = nn.Linear(aligned_dim, num_classes)

#     def forward(self, batch: Dict) -> torch.Tensor:
#         """
#         一个完全向量化的前向传播函数。

#         :param batch: 由PTACollator准备好的、包含填充张量和掩码的字典。
#         :return: 最终的分类logits。
#         """
#         # --- 0. 从批处理数据中解包 ---
#         # 我们假设Collator提供了这些键
#         subfield_tensor = batch["subfield_tensor"]
#         subfield_mask = batch["subfield_mask"]
#         simple_fields_tensor = batch["simple_fields_tensor"]
#         simple_fields_mask = batch["simple_fields_mask"]
        
#         # 记录关键维度
#         batch_size = subfield_tensor.shape[0]
#         num_parents = subfield_tensor.shape[1]
#         max_subfields = subfield_tensor.shape[2]
#         subfield_dim = subfield_tensor.shape[3]
        
#         # ==================== 阶段一：Subfield -> Field ====================
        
#         # a) 将父字段维度合并到批处理维度，以便并行处理
#         # (batch, num_parents, max_subfields, dim) -> (batch * num_parents, max_subfields, dim)
#         x_s1 = subfield_tensor.view(batch_size * num_parents, max_subfields, subfield_dim)
#         mask_s1 = subfield_mask.view(batch_size * num_parents, max_subfields)
        
#         # b) 一次性完成所有父字段的聚合！
#         aggregated_fields = self.subfield_aggregator(x_s1, mask=mask_s1)
#         # -> 结果形状: (batch * num_parents, subfield_aligned_dim)
        
#         # c) 将结果重新塑形，得到每个数据包的所有“升级后”的父字段向量
#         aggregated_fields = aggregated_fields.view(batch_size, num_parents, self.subfield_aligned_dim)
#         # -> (batch, num_parents, subfield_aligned_dim)

#         # ==================== 阶段二：Field -> Protocol Layer ====================
        
#         # a) 将聚合后的父字段和原始的简单字段拼接起来
#         #    aggregated_fields: (batch, num_parents, subfield_aligned_dim)
#         #    simple_fields_tensor: (batch, num_simple_fields, subfield_aligned_dim)
#         all_fields_tensor = torch.cat([aggregated_fields, simple_fields_tensor], dim=1)
        
#         # b) 同样，拼接对应的掩码
#         #    父字段都是真实数据，所以掩码全为False
#         parent_masks = torch.zeros(batch_size, num_parents, dtype=torch.bool, device=all_fields_tensor.device)
#         all_fields_mask = torch.cat([parent_masks, simple_fields_mask], dim=1)
        
#         # c) 对齐所有字段到 aligned_dim
#         aligned_fields = self.field_aligner(all_fields_tensor)
        
#         # d) 使用 protocol_layer_aggregator 进行聚合
#         #    注意：这里的实现简化了。一个完整的实现需要将字段按协议层分组再聚合。
#         #    为了保持代码简洁，我们暂时将所有字段视为一个大组进行聚合。
#         #    这仍然能捕获跨协议层的字段关联。
#         protocol_vectors = self.protocol_layer_aggregator(aligned_fields, mask=all_fields_mask)
#         # -> (batch, aligned_dim)
        
#         # ==================== 阶段三：(可选) ====================
#         # 由于阶段二已经得到了一个代表数据包的向量，阶段三可以简化或省略。
#         # 这里我们直接将阶段二的输出视为最终的 packet_vector
#         packet_vector = protocol_vectors
        
#         # ==================== 分类 ====================
#         logits = self.classifier(packet_vector)
        
#         return logits


class ProtocolTreeAttention(nn.Module):
    """
    一个经过重构的、高性能的PTA模型。
    它的forward函数是完全向量化的，不包含任何Python原生循环，
    以实现最大的计算效率。
    """
    def __init__(self,
                 # 我们需要从Collator获取这些维度信息
                 parent_fields_list: List[str],
                 simple_fields_by_layer: Dict[str, List[str]],
                 # 模型的核心超参数
                 subfield_aligned_dim: int = 64,
                 aligned_dim: int = 128,
                 num_heads: int = 4,
                 num_classes: int = 2):
        """
        初始化重构后的PTA模型。

        :param parent_fields_list: 一个包含所有父字段名称的有序列表。
        :param simple_fields_by_layer: 一个字典，键是协议层名，值是该层包含的简单字段列表。
        :param subfield_aligned_dim: 子字段对齐后的维度。
        :param aligned_dim: 字段对齐后的维度。
        :param num_heads: 注意力头数。
        :param num_classes: 最终分类任务的类别数。
        """
        super().__init__()
        
        self.parent_fields = parent_fields_list
        self.simple_fields_by_layer = simple_fields_by_layer
        self.protocol_layers = sorted(simple_fields_by_layer.keys())

        # --- 1. 阶段一：Subfield -> Field 的聚合器 ---
        # 这个聚合器将一次性处理所有父字段的子字段
        self.subfield_aggregator = AttentionAggregator(subfield_aligned_dim, num_heads)
        
        # --- 2. 阶段二：Field -> Protocol Layer 的聚合器 ---
        # a) 需要一个对齐层，统一所有字段（聚合后的父字段+简单字段）的维度
        #    聚合后的父字段维度是 subfield_aligned_dim
        #    简单字段的维度需要从Collator获取，但为了简化，我们假设Collator已将其对齐
        #    因此，这个对齐层现在的作用是统一所有字段到 aligned_dim
        self.field_aligner = nn.Linear(subfield_aligned_dim, aligned_dim) # 这是一个简化的假设
        
        # b) 这个聚合器将一次性处理所有协议层的字段
        self.protocol_layer_aggregator = AttentionAggregator(aligned_dim, num_heads)
        
        # --- 3. 阶段三：Protocol Layer -> Packet 的聚合器 ---
        self.final_aggregator = AttentionAggregator(aligned_dim, num_heads)
        
        # --- 4. 分类头 ---
        self.classifier = nn.Linear(aligned_dim, num_classes)

    def forward(self, batch: Dict) -> torch.Tensor:
        """
        一个完全向量化的前向传播函数。

        :param batch: 由PTACollator准备好的、包含填充张量和掩码的字典。
        :return: 最终的分类logits。
        """
        # 从批处理数据中解包
        stage1_tensor = batch["stage1_tensor"]
        stage1_mask = batch["stage1_mask"]
        simple_fields_tensors = batch["simple_fields_tensors"]
        simple_fields_masks = batch["simple_fields_masks"]
        
        batch_size = stage1_tensor.shape[0]
        
        # ==================== 阶段一：Subfield -> Field ====================
        # stage1_tensor: (batch, num_parents, max_subfields, dim)
        
        # a) 将父字段维度合并到批处理维度，以便并行处理
        num_parents = stage1_tensor.shape[1]
        max_subfields = stage1_tensor.shape[2]
        subfield_dim = stage1_tensor.shape[3]
        
        x_s1 = stage1_tensor.view(batch_size * num_parents, max_subfields, subfield_dim)
        mask_s1 = stage1_mask.view(batch_size * num_parents, max_subfields)
        
        # b) 一次性完成所有父字段的聚合！
        aggregated_fields = self.subfield_aggregator(x_s1, mask=mask_s1)
        # -> 结果形状: (batch * num_parents, subfield_aligned_dim)
        
        # c) 将结果重新塑形，得到每个数据包的所有“升级后”的父字段向量
        aggregated_fields = aggregated_fields.view(batch_size, num_parents, self.subfield_aligned_dim)
        # -> (batch, num_parents, subfield_aligned_dim)

        # ==================== 阶段二：Field -> Protocol Layer ====================
        # 我们需要为每个协议层，收集其所有的字段（包括聚合后的父字段和简单字段）
        
        protocol_layer_inputs = []
        protocol_layer_masks = []
        
        # a) 从Collator获取该层最多有多少个字段
        max_fields_this_layer = max(len(self.simple_fields_by_layer.get(p, [])) + \
                                    len([f for f in self.protocol_tree.get(p, []) if f in self.parent_fields])
                                    for p in self.protocol_layers)
        
        for p_name in self.protocol_layers:
            # i. 收集该层的所有“简单字段”向量和掩码
            simple_tensors = simple_fields_tensors[p_name]
            simple_masks = simple_fields_masks[p_name]
            
            # ii. 收集该层的所有“聚合后父字段”向量和掩码
            parent_indices_in_layer = [self.parent_to_idx[f] for f in self.protocol_tree.get(p_name, []) if f in self.parent_fields]
            parent_tensors = aggregated_fields[:, parent_indices_in_layer, :]
            # 父字段都是真实数据，所以掩码全为False
            parent_masks = torch.zeros(batch_size, parent_tensors.shape[1], dtype=torch.bool, device=parent_tensors.device)
            
            # iii. 将简单字段和父字段拼接起来
            combined_tensors = torch.cat([parent_tensors, simple_tensors], dim=1)
            combined_masks = torch.cat([parent_masks, simple_masks], dim=1)
            
            # iv. 对它们进行填充，以确保所有协议层的输入形状都一致
            num_actual_fields = combined_tensors.shape[1]
            padding_needed = max_fields_this_layer - num_actual_fields
            
            padded_tensors = nn.functional.pad(combined_tensors, (0, 0, 0, padding_needed))
            padded_masks = nn.functional.pad(combined_masks, (0, padding_needed), value=True) # 用True填充
            
            protocol_layer_inputs.append(padded_tensors)
            protocol_layer_masks.append(padded_masks)
            
        # b) 将所有协议层的输入堆叠起来，准备进行并行处理
        x_s2 = torch.stack(protocol_layer_inputs, dim=1)
        mask_s2 = torch.stack(protocol_layer_masks, dim=1)
        # -> x_s2: (batch, num_layers, max_fields_per_layer, dim)
        
        # c) 同样，将协议层维度合并到批处理维度
        num_layers = x_s2.shape[1]
        x_s2 = x_s2.view(batch_size * num_layers, max_fields_this_layer, -1)
        mask_s2 = mask_s2.view(batch_size * num_layers, max_fields_this_layer)
        
        # d) 对齐并聚合
        aligned_fields = self.field_aligner(x_s2)
        aggregated_layers = self.protocol_layer_aggregator(aligned_fields, mask=mask_s2)
        # -> (batch * num_layers, aligned_dim)
        
        # e) 重新塑形，得到每个协议层的代表向量
        protocol_vectors = aggregated_layers.view(batch_size, num_layers, self.aligned_dim)
        # -> (batch, num_layers, aligned_dim)
        
        # ==================== 阶段三：Protocol Layer -> Packet ====================
        # 协议层向量已经是规整的，无需掩码
        packet_vector = self.final_aggregator(protocol_vectors)
        
        # ==================== 分类 ====================
        logits = self.classifier(packet_vector)
        
        return logits
    
# class ProtocolTreeAttention(nn.Module):
#     # __init__ 方法与我们上一版讨论的、解决了对齐层维度问题的版本完全相同
#     def __init__(self, field_embedder, protocol_tree: Dict[str, List[str]], 
#                  subfield_aligned_dim: int = 64,
#                  aligned_dim: int = 128,
#                  num_heads: int = 4, 
#                  num_classes: int = 2):
#         super().__init__()
#         self.field_embedder = field_embedder
#         self.protocol_tree = protocol_tree
#         self.aligned_dim = aligned_dim
        
#         # --- 阶段一初始化 ---
#         self.subfield_aligners = nn.ModuleDict()
#         self.subfield_aggregators = nn.ModuleDict()

#         for parent_field, subfields in self.protocol_tree.items():
#             valid_subfields = [sf for sf in subfields if sf in self.field_embedder.embedding_slices]
#             if valid_subfields and parent_field in self.field_embedder.embedding_slices:
#                 parent_key = parent_field.replace('.', '__')
#                 aligner_group = nn.ModuleDict()
#                 for sf_name in valid_subfields:
#                     start, end = self.field_embedder.embedding_slices[sf_name]
#                     original_dim = end - start
#                     sf_key = sf_name.replace('.', '__')
#                     aligner_group[sf_key] = nn.Linear(original_dim, subfield_aligned_dim)
#                 self.subfield_aligners[parent_key] = aligner_group
#                 self.subfield_aggregators[parent_key] = AttentionAggregator(subfield_aligned_dim, num_heads)

#         # --- 阶段二初始化 ---
#         self.field_aligners = nn.ModuleDict()
#         for field_name in self.field_embedder.embedding_slices.keys():
#             input_dim_for_aligner = 0
#             if field_name.replace('.', '__') in self.subfield_aggregators:
#                 input_dim_for_aligner = subfield_aligned_dim
#             else:
#                 start, end = self.field_embedder.embedding_slices[field_name]
#                 input_dim_for_aligner = end - start
#             if input_dim_for_aligner > 0: # 避免为0维特征创建对齐层
#                 aligner_key = field_name.replace('.', '__')
#                 self.field_aligners[aligner_key] = nn.Linear(input_dim_for_aligner, self.aligned_dim)
        
#         self.protocol_layer_aggregators = nn.ModuleDict({
#             'eth': AttentionAggregator(self.aligned_dim, num_heads),
#             'ip': AttentionAggregator(self.aligned_dim, num_heads),
#             'tcp': AttentionAggregator(self.aligned_dim, num_heads)
#         })
        
#         # --- 阶段三和分类头 ---
#         self.final_aggregator = AttentionAggregator(self.aligned_dim, num_heads)
#         self.classifier = nn.Linear(self.aligned_dim, num_classes)

#     # ==================== 核心修改点：重构后的 forward 方法 ====================
#     def forward(self, batch_data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#         # 0. 初始嵌入
#         x = self.field_embedder(batch_data_dict)
        
#         # 1. 创建“中央数据总线”：一个包含所有初始嵌入向量的字典
#         # field_vectors = {
#         #     name: x[:, start:end]
#         #     for name, (start, end) in self.field_embedder.embedding_slices.items()
#         # }
#         field_vectors = self.field_embedder(batch_data_dict) 

#         # ==================== 阶段一：子字段 -> 字段 ====================
#         # 在这个阶段，我们“升级” field_vectors 字典中父字段的向量
#         for parent_key, aggregator in self.subfield_aggregators.items():
#             parent_field_original = parent_key.replace('__', '.')
#             subfield_names = self.protocol_tree[parent_field_original]
#             # valid_subfields = [sf for sf in subfield_names if sf in self.field_embedder.embedding_slices]
#             valid_subfields = [sf for sf in subfield_names if sf in field_vectors]
            
#             if not valid_subfields:
#                 continue

#             # a) 对齐：从字典中取出子字段向量，并通过各自的对齐层
#             aligned_subfield_vectors = []
#             for sf_name in valid_subfields:
#                 # sf_key = sf_name.replace('.', '__')
#                 # original_vector = field_vectors[sf_name]
#                 # aligner = self.subfield_aligners[parent_key][sf_key]
#                 # aligned_subfield_vectors.append(aligner(original_vector))
#                 aligner = self.subfield_aligners[parent_key][sf_name.replace('.', '__')]
#                 aligned_subfield_vectors.append(aligner(field_vectors[sf_name]))
            
#             # b) 聚合：将对齐后的向量送入聚合器
#             stacked_aligned = torch.stack(aligned_subfield_vectors, dim=1)
#             aggregated_vector = aggregator(stacked_aligned)
            
#             # c) 更新：用聚合后的高级向量，替换字典中原始的父字段向量
#             field_vectors[parent_field_original] = aggregated_vector

#         # ==================== 阶段二：字段 -> 协议层 ====================
#         protocol_vectors = []
#         for protocol_name in ['eth', 'ip', 'tcp']:
#             if protocol_name not in self.protocol_tree:
#                 continue
            
#             field_names_in_layer = self.protocol_tree[protocol_name]
            
#             # a) 对齐：从“升级后”的字典中，取出该层所有字段的向量，并通过各自的对齐层
#             aligned_field_vectors = []
#             for field_name in field_names_in_layer:
#                 if field_name in field_vectors:
#                     # vector_to_align = field_vectors[field_name]
#                     # aligner_key = field_name.replace('.', '__')
#                     # if aligner_key in self.field_aligners:
#                     #     aligner = self.field_aligners[aligner_key]
#                     #     aligned_field_vectors.append(aligner(vector_to_align))
#                     aligner_key = field_name.replace('.', '__')
#                     if aligner_key in self.field_aligners:
#                         aligner = self.field_aligners[aligner_key]
#                         aligned_field_vectors.append(aligner(field_vectors[field_name]))


#             if not aligned_field_vectors:
#                 continue

#             # b) 聚合：将对齐后的向量送入该协议层的聚合器
#             stacked_aligned = torch.stack(aligned_field_vectors, dim=1)
#             protocol_vector = self.protocol_layer_aggregators[protocol_name](stacked_aligned)
#             protocol_vectors.append(protocol_vector)

#         # ==================== 阶段三、分类 （代码不变） ====================
#         if not protocol_vectors:
#              # 如果没有任何协议层被处理，返回一个零向量以避免崩溃
#             batch_size = next(iter(batch_data_dict.values())).shape[0]
#             return torch.zeros(batch_size, self.classifier.out_features)

#         stacked_protocol_vectors = torch.stack(protocol_vectors, dim=1)
#         packet_vector = self.final_aggregator(stacked_protocol_vectors)
#         logits = self.classifier(packet_vector)
        
#         return logits
