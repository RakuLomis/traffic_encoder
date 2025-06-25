import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import os
from utils.data_loader import TrafficDataset

raw_df_path = os.path.join('.', 'Data', 'Test', 'merge_tls_test_01', 'discrete', '0.csv') 
config_path = os.path.join('.', 'utils', 'f2v.yaml')

# raw_df = pd.read_csv(raw_df_path)
traffic_dataset = TrafficDataset(raw_df_path, config_path)
print("---------")