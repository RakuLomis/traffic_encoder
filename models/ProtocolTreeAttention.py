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
        self.layer_norm = nn.LayerNorm(embed_dim) 
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(), 
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x): 
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

        attn_output, _ = self.attention(x_with_cls, x_with_cls, x_with_cls) 
        x = self.layer_norm(x_with_cls + attn_output) # residual 
        x = self.layer_norm(x + self.ffn(x)) 

        cls_output = x[:, 0, :] 
        return cls_output 
    
class ProtocolTreeAttention(nn.Module):
    def __init__(self, field_embedder, protocol_tree: Dict[str, List[str]], 
                 subfield_aligned_dim: int = 64, # newly add
                 aligned_dim: int = 128, num_heads: int = 4, num_classes: int = 2):
        """
        Main module for Protocol Tree Attention (PTA). 
        PTA handles and aggregates the subfields information, and sends the Attention results to their parent fields. 
        Then the fields in a same protocol will be aligned and processed to find the correlation. 
        Finally, the different protocols' weights are concatenated and pushed into final steps like Softmax to calculate the scores. 

        Parameters 
        ---------- 
        field_embedder: 
            已经实例化的FieldEmbedding模块。
        protocol_tree: 
            描述协议层次结构的字典。
        aligned_dim: 
            在协议层聚合前，将所有字段向量对齐到的统一维度。
        num_heads: 
            所有注意力模块的头数。
        num_classes: 
            最终分类任务的类别数。
        """
        super().__init__()
        self.field_embedder = field_embedder
        self.protocol_tree = protocol_tree
        self.aligned_dim = aligned_dim
        
        # # --- 阶段一：为“子字段->字段”的聚合创建聚合器 ---
        # self.subfield_aggregators = nn.ModuleDict()
        # for parent_field, subfields in self.protocol_tree.items(): # key(str): value(list)
        #     # 确保父字段有子字段，且它本身也是一个需要被嵌入的特征
        #     if subfields and parent_field in self.field_embedder.embedding_slices: # keys
        #         # 获取该父字段的嵌入维度，作为其聚合器的维度
        #         start, end = self.field_embedder.embedding_slices[parent_field]
        #         parent_embed_dim = end - start
                
        #         aggregator_key = parent_field.replace('.', '__')
        #         """
        #         维度错误, 子向量维度? 
        #         这里传递的是父字段对应的embedding向量维度
        #         既然是aggregator的作用其实是添加CLS标记并提取这个嵌入后的字段的全局信息, 
        #         那这里的作用就是为每个字段添加aggregator
        #         """
        #         self.subfield_aggregators[aggregator_key] = AttentionAggregator(parent_embed_dim, num_heads)

        # --- 阶段一：为“子字段->字段”的聚合创建对齐层和聚合器 ---
        self.subfield_aligners = nn.ModuleDict()
        self.subfield_aggregators = nn.ModuleDict()

        for parent_field, subfields in self.protocol_tree.items():
            valid_subfields = [sf for sf in subfields if sf in self.field_embedder.embedding_slices]
            
            if valid_subfields and parent_field in self.field_embedder.embedding_slices:
                parent_key = parent_field.replace('.', '__')
                
                # a) 为这个父字段下的所有子字段创建对齐层
                aligner_group = nn.ModuleDict()
                for sf_name in valid_subfields:
                    start, end = self.field_embedder.embedding_slices[sf_name]
                    original_dim = end - start
                    sf_key = sf_name.replace('.', '__')
                    aligner_group[sf_key] = nn.Linear(original_dim, subfield_aligned_dim)
                self.subfield_aligners[parent_key] = aligner_group # 对齐, 子字段对齐subfield_aligners[parent_key][sf_key]

                # b) 创建一个输入维度为“对齐后维度”的聚合器
                self.subfield_aggregators[parent_key] = AttentionAggregator(subfield_aligned_dim, num_heads)
        
        """
        What if fields are logical? 
        """

        # --- 阶段二：为“字段->协议层”的聚合创建聚合器和对齐层 ---
        # 1. 创建线性层，将每个字段的嵌入（可能维度不同）对齐到`aligned_dim`
        # self.field_aligners = nn.ModuleDict()
        # for field_name, (start, end) in self.field_embedder.embedding_slices.items():
        #     original_dim = end - start
        #     aligner_key = field_name.replace('.', '__')
        #     self.field_aligners[aligner_key] = nn.Linear(original_dim, self.aligned_dim) 
        
        # --- 阶段二: 字段 -> 协议层 (修正此处的初始化逻辑) ---
        self.field_aligners = nn.ModuleDict()
        
        # ==================== 核心修改点 ====================
        # Iterate through all fields that have an embedding.
        for field_name in self.field_embedder.embedding_slices.keys():
            input_dim_for_aligner = 0
            
            # Check if this field is a "parent" field (i.e., it has an aggregator).
            if field_name.replace('.', '__') in self.subfield_aggregators:
                # If it's a parent, its vector in Stage 2 comes from the aggregator.
                # The aggregator's output dimension is subfield_aligned_dim.
                input_dim_for_aligner = subfield_aligned_dim
            else:
                # If it's a "simple" field, its vector is its original embedding.
                start, end = self.field_embedder.embedding_slices[field_name]
                input_dim_for_aligner = end - start

            # Create the aligner with the CORRECT input dimension.
            aligner_key = field_name.replace('.', '__')
            self.field_aligners[aligner_key] = nn.Linear(input_dim_for_aligner, self.aligned_dim)

        # 2. 为每个协议主干（eth, ip, tcp）创建聚合器
        self.protocol_layer_aggregators = nn.ModuleDict({
            'eth': AttentionAggregator(self.aligned_dim, num_heads),
            'ip': AttentionAggregator(self.aligned_dim, num_heads),
            'tcp': AttentionAggregator(self.aligned_dim, num_heads)
            # 未来可以动态添加'tls'等
        })
        
        # --- 阶段三：为“协议层->最终数据包”创建最终聚合器 ---
        self.final_aggregator = AttentionAggregator(self.aligned_dim, num_heads)
        
        # --- 分类头 ---
        self.classifier = nn.Linear(self.aligned_dim, num_classes)

    def _get_vectors_by_names(self, x, field_names: List[str]) -> torch.Tensor:
        """一个辅助函数，根据字段名列表从大向量中提取并堆叠嵌入向量。"""
        vectors = []
        for name in field_names:
            if name in self.field_embedder.embedding_slices:
                start, end = self.field_embedder.embedding_slices[name]
                vectors.append(x[:, start:end]) # [(batch_size, end-start)]
        return torch.stack(vectors, dim=1) # (batch_size, dim)

    # def forward(self, batch_data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     # 0. 初始嵌入
    #     # 得到 (batch_size, total_embedding_dim) 的扁平化向量
    #     x = self.field_embedder(batch_data_dict)
        
    #     # ==================== 阶段一：子字段 -> 字段 ====================
    #     # 创建一个可修改的副本，用于存放“升级后”的嵌入
    #     upgraded_x = x.clone()
        
    #     for parent_field, aggregator in self.subfield_aggregators.items():
    #         parent_field_original = parent_field.replace('__', '.')
    #         subfield_names = self.protocol_tree[parent_field_original]
            
    #         # 提取所有子字段的向量
    #         subfield_vectors = self._get_vectors_by_names(x, subfield_names)
            
    #         self.subfield_aligners[parent_field][subfield_names]

    #         # 通过注意力聚合得到新的父字段向量
    #         aggregated_parent_vector = aggregator(subfield_vectors)
            
    #         # 用聚合后的向量，替换掉副本中原始的、简单的父字段向量
    #         start, end = self.field_embedder.embedding_slices[parent_field_original]
    #         upgraded_x[:, start:end] = aggregated_parent_vector
            
    #     # ==================== 阶段二：字段 -> 协议层 ====================
    #     protocol_vectors = []
    #     # 按固定顺序处理，保证输出稳定
    #     for protocol_name in ['eth', 'ip', 'tcp']:
    #         if protocol_name in self.protocol_tree:
    #             field_names_in_layer = self.protocol_tree[protocol_name]
                
    #             # 1. 提取该层所有字段的向量（已经是升级过的）
    #             layer_field_vectors_unaligned = []
    #             for field_name in field_names_in_layer:
    #                 if field_name in self.field_embedder.embedding_slices:
    #                     start, end = self.field_embedder.embedding_slices[field_name]
    #                     layer_field_vectors_unaligned.append(upgraded_x[:, start:end])
                
    #             # 2. 将不同维度的字段向量对齐到`aligned_dim`
    #             aligned_vectors = []
    #             for i, field_name in enumerate(field_names_in_layer):
    #                  if field_name in self.field_embedder.embedding_slices:
    #                     aligner = self.field_aligners[field_name.replace('.', '__')]
    #                     aligned_vectors.append(aligner(layer_field_vectors_unaligned[i]))
                
    #             # 3. 将对齐后的向量堆叠并送入该层的聚合器
    #             if aligned_vectors:
    #                 stacked_aligned_vectors = torch.stack(aligned_vectors, dim=1)
    #                 protocol_vector = self.protocol_layer_aggregators[protocol_name](stacked_aligned_vectors)
    #                 protocol_vectors.append(protocol_vector)

    #     # ==================== 阶段三：协议层 -> 最终数据包 ====================
    #     # 将所有协议层的代表向量堆叠起来
    #     stacked_protocol_vectors = torch.stack(protocol_vectors, dim=1)
        
    #     # 进行最终的聚合
    #     packet_vector = self.final_aggregator(stacked_protocol_vectors)
        
    #     # ==================== 分类 ====================
    #     logits = self.classifier(packet_vector)
        
    #     return logits
    def forward(self, batch_data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 0. 初始嵌入
        # 得到 (batch_size, total_embedding_dim) 的扁平化向量
        x = self.field_embedder(batch_data_dict)
        
        # ==================== 阶段一：子字段 -> 字段 (已更新对齐逻辑) ====================
        
        # 创建一个新的字典，用来存放“升级后”的字段向量。
        # 键是字段名，值是对应的向量。
        upgraded_field_vectors = {}
        
        # 首先，将所有“简单”字段（没有孩子的）的原始嵌入直接复制过来
        for field_name, (start, end) in self.field_embedder.embedding_slices.items():
            if field_name not in self.protocol_tree or not self.protocol_tree[field_name]:
                upgraded_field_vectors[field_name] = x[:, start:end]

        # 然后，处理所有“父”字段，用聚合后的向量覆盖它们
        for parent_key, aggregator in self.subfield_aggregators.items():
            parent_field_original = parent_key.replace('__', '.')
            subfield_names = self.protocol_tree[parent_field_original]
            valid_subfields = [sf for sf in subfield_names if sf in self.field_embedder.embedding_slices]
            
            # --- 这是新增的核心对齐逻辑 ---
            
            # 1. 准备一个空列表，用来收集对齐后的子字段向量
            aligned_subfield_vectors = []
            
            # 2. 遍历该父字段下的每一个子字段
            for sf_name in valid_subfields:
                sf_key = sf_name.replace('.', '__')
                
                # 3. 提取该子字段原始的、维度不一的嵌入向量
                start, end = self.field_embedder.embedding_slices[sf_name]
                original_subfield_vector = x[:, start:end]
                
                # 4. 查找并应用它专属的对齐层(nn.Linear)
                aligner = self.subfield_aligners[parent_key][sf_key]
                aligned_vector = aligner(original_subfield_vector)
                
                aligned_subfield_vectors.append(aligned_vector)
                
            # --- 对齐逻辑结束 ---

            if aligned_subfield_vectors:
                # 5. 将所有维度已统一的子字段向量堆叠起来
                stacked_aligned_vectors = torch.stack(aligned_subfield_vectors, dim=1)
                
                # 6. 送入该父字段专属的聚合器
                aggregated_parent_vector = aggregator(stacked_aligned_vectors)
                
                # 7. 将聚合后的、高质量的向量存入我们的新字典
                upgraded_field_vectors[parent_field_original] = aggregated_parent_vector
            
        # ==================== 阶段二：字段 -> 协议层 ====================
        protocol_vectors = []
        for protocol_name in ['eth', 'ip', 'tcp']:
            if protocol_name in self.protocol_tree:
                field_names_in_layer = self.protocol_tree[protocol_name]
                
                # 1. 提取该层所有字段的向量 (现在从upgraded_field_vectors中提取)
                layer_field_vectors_unaligned = []
                for field_name in field_names_in_layer:
                    if field_name in upgraded_field_vectors:
                        layer_field_vectors_unaligned.append(upgraded_field_vectors[field_name])
                
                # 2. 将不同维度的字段向量对齐到`aligned_dim`
                # 注意：这里的对齐逻辑也需要相应调整，以处理新的输入
                aligned_vectors = []
                # 我们需要确保字段顺序和对齐层顺序一致
                valid_fields_in_layer = [f for f in field_names_in_layer if f in upgraded_field_vectors]
                for field_name in valid_fields_in_layer:
                    aligner = self.field_aligners[field_name.replace('.', '__')]
                    # 从upgraded_field_vectors中获取向量
                    vector_to_align = upgraded_field_vectors[field_name]
                    aligned_vectors.append(aligner(vector_to_align))
                
                # 3. 将对齐后的向量堆叠并送入该层的聚合器
                if aligned_vectors:
                    stacked_aligned_vectors = torch.stack(aligned_vectors, dim=1)
                    protocol_vector = self.protocol_layer_aggregators[protocol_name](stacked_aligned_vectors)
                    protocol_vectors.append(protocol_vector)

        # ==================== 阶段三、分类 （代码不变） ====================
        stacked_protocol_vectors = torch.stack(protocol_vectors, dim=1)
        packet_vector = self.final_aggregator(stacked_protocol_vectors)
        logits = self.classifier(packet_vector)
        
        return logits

# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 这是一个演示如何使用PTA模块的伪代码
    
    # 假设我们有实例化的 FieldEmbedding 和 protocol_tree
    # class MockFieldEmbedding(nn.Module): ... # (需要一个模拟的FieldEmbedding类)
    # field_embedder = MockFieldEmbedding(...)
    # protocol_tree = {...} # 您之前提供的协议树
    
    # 实例化PTA模型
    # pta_model = ProtocolTreeAttention(field_embedder, protocol_tree, num_classes=10)
    
    # 创建一批虚拟数据
    # dummy_batch = {...} # (一个符合格式的数据字典)
    
    # 执行前向传播
    # output_logits = pta_model(dummy_batch)
    
    # print(f"Shape of the final output logits: {output_logits.shape}")
    # 期待的输出形状: (batch_size, num_classes)
    print("ProtocolTreeAttention class defined successfully.")
    print("To run this file, you need to instantiate a mock FieldEmbedding class and provide a protocol_tree dictionary.")