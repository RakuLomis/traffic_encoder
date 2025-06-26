import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os
from utils.data_loader import TrafficDataset
from models.FieldEmbedding import FieldEmbedding

raw_df_path = os.path.join('.', 'Data', 'Test', 'merge_tls_test_01', 'discrete', '0.csv') 
config_path = os.path.join('.', 'utils', 'f2v.yaml')


# raw_df = pd.read_csv(raw_df_path)
traffic_dataset = TrafficDataset(raw_df_path, config_path)
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
field_embedder = FieldEmbedding(config_path) 
output_vector = field_embedder(first_batch) 
print(f"\nShape of the final model output vector: {output_vector.shape}")