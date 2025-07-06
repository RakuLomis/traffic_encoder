import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os
from utils.data_loader import TrafficDataset
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention

raw_df_path = os.path.join('.', 'Data', 'Test', 'merge_tls_test_01', 'discrete', '0.csv') 
config_path = os.path.join('.', 'utils', 'f2v.yaml')
vocab_path = os.path.join('.', 'Data', 'Test', 'categorical_vocabs.yaml')

raw_df = pd.read_csv(raw_df_path)
protocol_tree = protocol_tree(raw_df.columns.tolist())

traffic_dataset = TrafficDataset(raw_df_path, config_path, vocab_path)
print("---------") 
sample_item = traffic_dataset[0]
# 为了简洁，只打印前5个字段
for i, (k, v) in enumerate(sample_item.items()):
    if i >= 5: break
    print(f"'{k}': {v}")
batch_size = 4
traffic_dataloader = DataLoader(
    dataset=traffic_dataset,
    batch_size=batch_size,
    shuffle=True # 在训练时打乱数据顺序
)
first_batch = next(iter(traffic_dataloader))
print(first_batch['eth.dst'])
print("---------") 
field_embedder = FieldEmbedding(config_path) # embedding_slices here
# output_vector = field_embedder(first_batch) 
# print("--------------embedding_slices----------------")
# print(field_embedder.embedding_slices)

# test_tcp_flags_subfields = protocol_tree['tcp.flags'] 
# test_tcp_flags_subfields_to_embed = [f for f in test_tcp_flags_subfields if f in field_embedder.embedding_slices]
# subfield_tensors = [] 

# for subfield_name in test_tcp_flags_subfields_to_embed: 
#     start, end = field_embedder.embedding_slices[subfield_name] 
#     # (batch_size, subfield_embedding_dim)
#     subfield_tensor = output_vector[:, start:end]
#     subfield_tensors.append(subfield_tensor)

# # stack all the subfields in order to send them into Attention Block 
# test_tcp_flags_subfields_att_input = torch.stack(subfield_tensors, dim=1) # (batch_size, num_subfields, subfield_embedding_dim)

# print(f"\nShape of the final model output vector: {output_vector.shape}") 

# 实例化PTA模型
pta_model = ProtocolTreeAttention(field_embedder, protocol_tree, num_classes=10)

# 执行前向传播
output_logits = pta_model(first_batch)

print(f"Shape of the final output logits: {output_logits.shape}")
# 期待的输出形状: (batch_size, num_classes)
print(f"The value of output is: {output_logits}")
print("ProtocolTreeAttention class defined successfully.")
print("To run this file, you need to instantiate a mock FieldEmbedding class and provide a protocol_tree dictionary.")