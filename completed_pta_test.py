import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os 
from tqdm import tqdm
from utils.data_loader import TrafficDataset
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention 
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating

CONTINUOUS_BLOCK = 'continuous' 
DISCRETE_BLOCK = 'discrete'

current_path = os.path.dirname(os.path.abspath(__file__)) 
csv_name = 'dataset_29_completed_label'
raw_df_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness')
raw_df_path = os.path.join(raw_df_directory, csv_name + '.csv')
config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
vocab_path = os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml')

print(f"Reading raw_df: {raw_df_path}")
raw_df = pd.read_csv(raw_df_path, low_memory=False)
raw_df.reset_index(inplace=True)
if 'frame_num' in raw_df.columns:
    raw_df.drop(columns=['frame_num'], inplace=True)
protocol_tree = protocol_tree(raw_df.columns.tolist())
# del raw_df 

if not os.path.exists(os.path.join(raw_df_directory, csv_name, DISCRETE_BLOCK)): 
    print("Truncation has not finished. ")
    list_df_block = padding_or_truncating(raw_df, False, DISCRETE_BLOCK) 
    for block_num in tqdm(range(len(list_df_block)), desc= "Truncating..."): 
        output_csv_in_fold(list_df_block[block_num], os.path.join(raw_df_directory, csv_name, DISCRETE_BLOCK), f'{block_num}' + '.csv') 
    print("Truncation finished. ")


del raw_df 
print(f"raw_df: {raw_df_path} has been deleted to release resources. ")
# traffic_dataset = TrafficDataset(raw_df_path, config_path, vocab_path)
# print("---------") 
# sample_item = traffic_dataset[0]
# # 为了简洁，只打印前5个字段
# for i, (k, v) in enumerate(sample_item.items()):
#     if i >= 5: break
#     print(f"'{k}': {v}")
# batch_size = 4
# traffic_dataloader = DataLoader(
#     dataset=traffic_dataset,
#     batch_size=batch_size,
#     shuffle=True # 在训练时打乱数据顺序
# )
# first_batch = next(iter(traffic_dataloader))
# print(first_batch['eth.dst'])
# print("---------") 
# field_embedder = FieldEmbedding(config_path) # embedding_slices here

# # 实例化PTA模型
# pta_model = ProtocolTreeAttention(field_embedder, protocol_tree, num_classes=10)

# # 执行前向传播
# output_logits = pta_model(first_batch)

# print(f"Shape of the final output logits: {output_logits.shape}")
# # 期待的输出形状: (batch_size, num_classes)
# print(f"The value of output is: {output_logits}")
# print("ProtocolTreeAttention class defined successfully.")
# print("To run this file, you need to instantiate a mock FieldEmbedding class and provide a protocol_tree dictionary.")