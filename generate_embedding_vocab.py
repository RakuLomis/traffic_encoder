import os 
import yaml
from utils.dataframe_tools import generate_vocabulary 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
csv_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label.csv')

with open(config_path, 'r') as f: 
    yaml_config = yaml.safe_load(f)['field_embedding_config']
fields = list(yaml_config.keys())

categorical_fields = [item for item in fields if yaml_config[item]['type'] == 'categorical']

addr_fields = ['eth.dst', 'eth.src', 'ip.src', 'ip.dst']
other_fields = ['tcp.reassembled_segments']
fields_except_addr = [item for item in fields if item not in addr_fields and item not in other_fields]

vocab_reflect = generate_vocabulary(csv_path, categorical_fields, os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs_v1.yaml'))