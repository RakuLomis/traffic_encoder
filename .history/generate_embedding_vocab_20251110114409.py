import os 
import yaml
from utils.dataframe_tools import generate_vocabulary , generate_vocabulary_memory_optimized

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = 'train_set.csv'
STEP = 'datasets_split'
dataset_name = 'ISCX-VPN'
dataset_name2 = 'ISCX-TOR-Acctivity' 
dataset_name3 = 'ISCX-TOR-Application'
# dataset_name4 = 'dataset_29_d1'
dataset_name4 = 'USTC-TFC2016-Benign'
root_path = os.path.join('..', 'TrafficData', 'datasets_csv_add2')
config_path = os.path.join('.', 'Data', 'fields_embedding_configs_v1.yaml')
# csv_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label.csv')
# csv_path = os.path.join('..', 'TrafficData', 'dataset_20_d2_csv', 'dataset_20_d2.csv') 

# csv_path = os.path.join(root_path, STEP, dataset_name, TARGET)
# output_path = os.path.join(root_path, 'categorical_vocabs', dataset_name + '_vocabs.yaml')

# csv_path = os.path.join(root_path, STEP, dataset_name2, TARGET)
# output_path = os.path.join(root_path, 'categorical_vocabs', dataset_name2 + '_vocabs.yaml')


# csv_path = os.path.join(root_path, STEP, dataset_name3, TARGET)
# output_path = os.path.join(root_path, 'categorical_vocabs', dataset_name3 + '_vocabs.yaml')

csv_path = os.path.join(root_path, STEP, dataset_name4, TARGET)
output_path = os.path.join(root_path, 'categorical_vocabs', dataset_name4 + '_vocabs.yaml')

with open(config_path, 'r') as f: 
    yaml_config = yaml.safe_load(f)['field_embedding_config']
fields = list(yaml_config.keys())

categorical_fields = [item for item in fields if yaml_config[item]['type'] == 'categorical']

addr_fields = ['eth.dst', 'eth.src', 'ip.src', 'ip.dst']
other_fields = ['tcp.reassembled_segments']
target_categorical_fields = [item for item in categorical_fields if item not in addr_fields and item not in other_fields]
print(target_categorical_fields)

# vocab_reflect = generate_vocabulary(csv_path, target_categorical_fields, os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml'))
generate_vocabulary_memory_optimized(csv_path, target_categorical_fields, output_path)