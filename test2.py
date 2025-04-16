import pandas as pd
import numpy as np
import os 

path = 'E:\\Program\\VSCode\\MyGit\\traffic_encoder\\Data\\Test\\merge_tls_test_01.csv' 
df = pd.read_csv(path) 

list_col = df.columns.tolist() 

def find_fields_by_prefix(layers: list, layer_prefixes: dict, fields: list, len_prefix: int, init: bool): 
    next_layers = []
    if len(layers) > 0: 
        if init: 
            for field in fields: 
                prefix = field.split('.')[0] 
                if prefix not in layers and len(field.split('.')) - len_prefix == 0: 
                    layer_prefixes['statistics'].append(field) 
                elif field.startswith(prefix) and len(field.split('.')) - len_prefix == 1: 
                    layer_prefixes[prefix].append(field) 
                    next_layers.append(field) 
        else: 
            for prefix in layers: 
                # for prefix in layer_prefixes[layer]: 
                for field in fields: 
                    if field.startswith(prefix) and len(field.split('.')) - len_prefix == 1: 
                        if prefix not in layer_prefixes: 
                            layer_prefixes[prefix] = [] 
                            layer_prefixes[prefix].append(field) 
                        else: 
                            layer_prefixes[prefix].append(field) 
                    else: 
                        continue 
                    next_layers.append(field) 
            # len_prefix += 1 
    init = False
    return layer_prefixes, next_layers, init 

def protocol_tree(list_fields: list, list_layers = ['eth', 'ip', 'tcp', 'tls']): 
    """
    Find the hierarchy structure of protocols by handling the csv columns. 
    """
    dict_protocol_tree = {item: [] for item in list_layers} 
    dict_protocol_tree['statistics'] = [] 
    lens = [len(item.split('.')) for item in list_fields]
    len_prefix = 1 # length of current prefix, i.e. eth 
    max_field_len = max(lens)
    init = True
    while len_prefix < max_field_len: 
        # len_prefix += 1 # xx.xx 
        # list_prefix = [] # xx.xx 
        dict_protocol_tree, list_layers, init = find_fields_by_prefix(list_layers, dict_protocol_tree, list_fields, len_prefix, init) 
        len_prefix += 1
    return dict_protocol_tree 

dict_tree = protocol_tree(list_col) 
print(dict_tree)