import pandas as pd
import os 
import numpy as np 
from typing import Optional 
from tqdm import tqdm 
from typing import Literal 
import yaml

def get_file_path(dir_path: str, prefix: Optional[str] = None, postfix: Optional[str] = None): 
    """
    Get all files' paths and names, which statisfy the prefix and postfix, 
    from the specific directory (folder). 

    Parameters 
    ---------- 
    dir_path: str 
        Path of the directory to search. 
    preifx: str, Optional 
        File prefix filter (e.g., "img_"), by default None (no filter). 
    postfix: str, Optional 
        File suffix filter (e.g., ".png"), by default None (no filter). 
    """
    target_paths = [] 
    file_names = [] 
    if os.path.exists(dir_path): 
        if os.path.isdir(dir_path): 
            with os.scandir(dir_path) as entries: 
                for entry in tqdm(entries, "get_file_path: "):
                    if entry.is_file():
                        file_name = entry.name
                        # Check prefix condition
                        prefix_ok = (prefix is None) or file_name.startswith(prefix)
                        # Check postfix condition
                        postfix_ok = (postfix is None) or file_name.endswith(postfix)

                        if prefix_ok and postfix_ok:
                            # Build full path
                            full_path = entry.path
                            target_paths.append(full_path)

                            # Process base name 
                            if postfix and file_name.endswith(postfix):
                                base_name = file_name[:-len(postfix)] 
                            elif postfix and not file_name.endswith(postfix): 
                                continue 
                            else: # no postfix, record file name
                                base_name = file_name
                            file_names.append(base_name)
    else: 
        print("Invalid directory path") 
    return target_paths, file_names

def filter_out_nan(df: pd.DataFrame): 
    """
    Filter out columns whose values are all NaN. 
    """
    df = df.replace(['', 'nan', 'NaN', None], pd.NA) # for .py NaN maynot be identified automately like in .ipynb
    all_nan_cols = df.columns[df.isna().all()] 
    return df.drop(columns=all_nan_cols) 


def to_integer_code(df: pd.DataFrame, col_name = 'reassembled_segments'): 
    """
    Turn the specific column into the integer code for convenience. 

    Returns
    -------
    df: dataframe
        Transformed by the dict: {'[]': -1, '[1, 2]': 0, '[3, 4, 5]': 1, ...} 
    """
    dict_ori_integer = {} 
    code = 0 
    if col_name in df.columns: 
        for value in tqdm(df[col_name], "to_integer_code: "): 
            if value == '[]': # '[]' represents no reassembled segments
                dict_ori_integer[value] = -1 
            if value not in dict_ori_integer: 
                dict_ori_integer[value] = code 
                code += 1 
        df[col_name] = df[col_name].apply(lambda x: dict_ori_integer[x]) 
    else: 
        print(f"No column named ${col_name}") 
    return df 

def output_csv_in_fold(df: pd.DataFrame, fold_path: str, csv_name: str, index: Optional[str] = False): 
    """
    Output the dataframe as a csv into a specific fold. 
    If the fold is not exist, this function will create a new one. 
    """
    os.makedirs(fold_path, exist_ok=True) 
    file_path = os.path.join(fold_path, csv_name) 
    df.to_csv(file_path, index=index) 
    
def padding_features(df: pd.DataFrame, padding_value='ffff'): 
    """
    Padding the NaN value with 'ffff', in order to represent -1 in decimal number. 
    In fact, almost all values in the dataframe are not numbers, instead, strings. 
    """
    return df.fillna(padding_value) 


def get_not_nan_pos(mask_df: pd.DataFrame): 
    """
    Get the positions of values which are not NaN. 
    For Columns, their positions start from '0'. 
    But for Rows, their positions start from '1' in order to align with packets' number in Wireshark. 
    
    Attention: The '1' is not handled in our functions, but is extracted from the feature 
    'tcp.frame_num' and used to be the index. 

    Returns 
    ------- 
    dict_true: dict, {col_num: [row_num (not NaN)]}
    """
    dict_true = {} 
    for col_num in range(mask_df.shape[1]): 
        list_true_indices = list(mask_df[mask_df.iloc[:, col_num]].index) 
        dict_true[col_num] = list_true_indices 
    return dict_true 

def truncate_to_block(dict_true: dict, block_type: Literal['continuous', 'discrete'] = 'continuous'): 
    """
    Truncate the packets' fields into blocks for different experts. 
    
    Compared with filling NaN values with '-1', 'ffff', or other symbols, this method 
    significantly reduces the number of parameters and has the potential to enhance the feasibility 
    of models in Mixture of Experts (MoE) structures. 
    
    Parameters 
    ---------- 
    dict_true: dict
        The dict which contains the positions of not NaN values. 
    block_type: Literal['continuous', 'discrete'] 
        Generating block with 'continuous' fields or 'discrete' fields. 

    Returns 
    ------- 
    dict_block: dict 
        {'block': [], 'columns': [], 'rows': [] }. 
        Specifically, columns use index and were not changed (add, delete, etc.), which can be located by .iloc. 
        However, for rows, since .iloc works by the position not index, and the non-TCP packets were filtered out in previous 
        work, .loc should be used to handle this situation. 
    """
    if block_type not in ['continuous', 'discrete']: 
        raise ValueError("block_type must be either 'continuous' or 'discrete'") 
    
    dict_block = {
        'block': [], 
        'columns': [], 
        'rows': [] 
    } 

    block_flag = 0 
    last_key = 0 
    last_value = dict_true[last_key] 
    list_col = [] 
    list_record_col = []
    if block_type == 'continuous': 
        for key, value in dict_true.items(): 
            if key == last_key: # init 
                dict_block['block'].append(block_flag) 
                list_col.append(key) 
                dict_block['columns'].append(list_col.copy()) 
                dict_block['rows'].append(dict_true[key]) 
            else: 
                if value == last_value: 
                    dict_block['columns'][block_flag].append(key) 
                if value != last_value: 
                    block_flag += 1 
                    dict_block['block'].append(block_flag) 
                    list_col.clear() 
                    list_col.append(key) 
                    dict_block['columns'].append(list_col.copy()) 
                    dict_block['rows'].append(dict_true[key]) 
                    last_key = key 
                    last_value =value 
    elif block_type == 'discrete': 
        for key, value in dict_true.items(): 
            if key not in list_record_col: 
                for ik, iv in dict_true.items(): 
                    if iv == value: 
                        if block_flag not in dict_block['block']: 
                            dict_block['block'].append(block_flag) 
                            dict_block['rows'].append(dict_true[key]) 
                        list_col.append(ik) 
                    if iv != value: 
                        continue 
                dict_block['columns'].append(list_col.copy()) 
                list_record_col.extend(list_col.copy()) 
                list_col.clear()
                block_flag += 1 
    return dict_block 
        

# def to_block_continuous(dict_true: dict): 
    dict_block = {
        'block': [], 
        'columns': [], 
        'rows': [] 
    } 

    block_flag = 0 
    last_key = 0 
    last_value = dict_true[last_key] 
    list_col = []
    for key, value in dict_true.items(): 
        if key == last_key: # init 
            dict_block['block'].append(block_flag) 
            list_col.append(key) 
            dict_block['columns'].append(list_col.copy()) 
            dict_block['rows'].append(dict_true[key]) 
        else: 
            if value == last_value: 
                dict_block['columns'][block_flag].append(key) 
            if value != last_value: 
                block_flag += 1 
                dict_block['block'].append(block_flag) 
                list_col.clear() 
                list_col.append(key) 
                dict_block['columns'].append(list_col.copy()) 
                dict_block['rows'].append(dict_true[key]) 
                last_key = key 
                last_value =value 
    return dict_block 

def truncating_features(df: pd.DataFrame, block_type: Literal['continuous', 'discrete'] = 'continuous'): 
    """
    Filter out the NaN columns and truncate the entries into different blocks. 
    
    Parameters 
    ---------- 
    df: pd.DataFrame 
        The dataframe has deleted the NaN columns already. 
    block_type: Literal['continuous', 'discrete'] 
        Generating block with 'continuous' fields or 'discrete' fields. 

    Returns 
    ------- 
    dict_block: dict 
        {'block': [], 'columns': [], 'rows': [] }. 
    """
    if block_type not in ['continuous', 'discrete']: 
        raise ValueError("block_type must be either 'continuous' or 'discrete'") 
    mask_df = df.notnull() 
    dict_true = get_not_nan_pos(mask_df) 
    dict_block = truncate_to_block(dict_true, block_type) 
    return dict_block 

def block_to_dataframe(dict_block: dict, df_ori: pd.DataFrame): # delete output_path: str
    """
    Turn the dict concluding block truncating information into a dataframe. 

    Returns 
    ------- 
    list_block: list
        [block0: pd.DataFrame, block1: pd.DataFrame, ...] 
    """ 
    list_block = []
    block_values = dict_block['block'] 
    columns_values = dict_block['columns'] 
    rows_values = dict_block['rows'] 
    for block_name, columns, rows in tqdm(zip(block_values, columns_values, rows_values)): 
        if 0 not in columns: # frame_num must be added as index
            columns.append(0) 
        subset_rows = df_ori.loc[rows] 
        sub_df = subset_rows.iloc[:, columns] 
        # output_csv_in_fold(sub_df, output_path, 'block_' + block_name + '.csv') 
        list_block.append(sub_df) 
    return list_block 

# def truncating_features(dict_true: dict): 
    dict_block = {
        'block': [], 
        'columns': [], 
        'rows': [] 
    } 

    block_flag = 0 
    list_col = [] 
    list_record_col = []
    for key, value in dict_true.items(): 
        if key not in list_record_col: 
            for ik, iv in dict_true.items(): 
                if iv == value: 
                    if block_flag not in dict_block['block']: 
                        dict_block['block'].append(block_flag) 
                        dict_block['rows'].append(dict_true[key]) 
                    list_col.append(ik) 
                if iv != value: 
                    continue 
            dict_block['columns'].append(list_col.copy()) 
            list_record_col.extend(list_col.copy()) 
            list_col.clear()
            block_flag += 1 
    return dict_block 

def padding_or_truncating(df: pd.DataFrame, pon: bool, block_type: Literal['continuous', 'discrete'] = 'continuous'): 
    """
    Padding the NaN values or truncating the dataframe into various blocks. 

    Parameters 
    ---------- 
    df: pd.DataFrame
        The input dataframe has many NaN values in different features. 
    pon: bool
        Padding or Not. True means using .fillna to padding the NaN values, while 
        False represents truncating the dataframe into different blocks. In each block, 
        feature values are clustered. 
    block_type: Literal['continuous', 'discrete'] 
        Generating block with 'continuous' fields or 'discrete' fields. 
    """ 
    if block_type not in ['continuous', 'discrete']: 
        raise ValueError("block_type must be either 'continuous' or 'discrete'") 
    res_list = [] 
    df = to_integer_code(filter_out_nan(df)) 
    if pon: 
        df = padding_features(df) 
        res_list.append(df)
    else: 
        dict_block = truncating_features(df, block_type) 
        res_list = block_to_dataframe(dict_block, df) 
    return res_list 

def find_fields_by_prefix_physically(layers: list, layer_prefixes: dict, fields: list, init: bool): 
    next_layers = []
    if len(layers) > 0: 
        if init: 
            for field in fields: 
                prefix = field.split('.')[0] 
                len_prefix = len(prefix.split('.')) 
                if prefix not in layers and len(field.split('.')) - len_prefix == 0: 
                    layer_prefixes['statistics'].append(field) 
                elif field.startswith(prefix) and len(field.split('.')) - len_prefix == 1: 
                    layer_prefixes[prefix].append(field) 
                    next_layers.append(field) 
                # for tls, it does not have field like tls.xx, only has tls.xx.xx 
            empty_layers = [] 
            extra_len_prefix = len_prefix
            # check up the empty layers in initial, and increase the length of prefix. 
            # We hold an assumption that the emptiness means the length of prefix is not long enough
            for key in layer_prefixes.keys(): 
                if len(layer_prefixes[key]) == 0: 
                    empty_layers.append(key) 
            while len(empty_layers) > 0: 
                extra_len_prefix += 1
                for layer in empty_layers[:]: 
                    for field in fields: 
                        if field.startswith(layer): 
                            prefix = '.'.join(field.split('.')[:extra_len_prefix])
                            if field.startswith(prefix) and len(field.split('.')) - extra_len_prefix == 1: 
                                layer_prefixes[layer].append(field) 
                                next_layers.append(field) 
                    empty_layers.remove(layer)                
        else: 
            for prefix in layers: 
                len_prefix = len(prefix.split('.')) 
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

def find_fields_by_prefix_logically(layers: list, layer_prefixes: dict, fields: list, len_prefix: int): 
    next_layers = [] 
    if len(layers) > 0: 
        for field in fields: 
            prefix = '.'.join(field.split('.')[:len_prefix]) 
            if prefix not in layers and len(field.split('.')) - len_prefix == 0: 
                layer_prefixes['statistics'].append(field) 
            elif field.startswith(prefix) and len(field.split('.')) - len_prefix == 1: # 
                if prefix not in layer_prefixes: 
                    layer_prefixes[prefix] = [] 
                layer_prefixes[prefix].append(field) 
                next_layers.append(field) 
            elif field.startswith(prefix) and len(field.split('.')) - len_prefix >= 2: 
                current_layer = '.'.join(field.split('.')[:len_prefix + 1]) 
                if prefix in layer_prefixes: 
                    if current_layer not in layer_prefixes[prefix]: 
                        layer_prefixes[prefix].append(current_layer) 
                        next_layers.append(current_layer) 
                else: # in order to handle the single structure fields like 'tls.ber.bitstring.padding'
                    layer_prefixes[prefix] = [] 
                    layer_prefixes[prefix].append(current_layer) 
                    next_layers.append(current_layer) 
    return layer_prefixes, next_layers 

def protocol_tree(list_fields: list, list_layers = ['eth', 'ip', 'tcp', 'tls'], logical_tree =True): 
    """
    Find the hierarchy structure of protocols by handling the csv columns. 
    """
    dict_protocol_tree = {item: [] for item in list_layers} 
    dict_protocol_tree['statistics'] = [] 
    lens = [len(item.split('.')) for item in list_fields]
    len_prefix = 1 # length of current prefix, i.e. eth 
    max_field_len = max(lens) 
    if logical_tree: 
        while len_prefix < max_field_len: 
            dict_protocol_tree, list_layers = find_fields_by_prefix_logically(list_layers, dict_protocol_tree, list_fields, len_prefix) 
            len_prefix += 1
    else:  
        init = True
        while len_prefix < max_field_len: 
            dict_protocol_tree, list_layers, init = find_fields_by_prefix_physically(list_layers, dict_protocol_tree, list_fields, init) 
            len_prefix += 1
    return dict_protocol_tree 

def find_fields_in_pta(protocol, dict_protocol_tree, physical_nodes): 
    """
    Return the list of field, subfields and field^{\prime} for Protocol Tree Attention. 
    The field^{\prime} is the field which does not have subfields. 
    
    Parameters 
    ---------- 
    protocol: str 
        The protocol name. 
    dict_protocol_tree: dict 
        The dict concluding protocol tree information. 
    physical_nodes: list 
        The physically existing fields in the dataframe. 

    Returns 
    ------- 
    list_fields_subfields: list 
        The list of fields and their subfields, which are consisted of 'field', 'subfields', and 'is_logical'.  
    list_fields_no_subfields: list 
        The list of the field^{\prime}s.  
    """
    list_fields_subfields = [
        # {'field': str, 'subfields': list, 'is_logical': bool}, 
    ] 
    list_fields_no_subfields = [] 
    for field in dict_protocol_tree[protocol]: 
        if field in physical_nodes: 
            # if dict_protocol_tree[field] does not exist, add to list_fields_no_subfields
            if field not in dict_protocol_tree: # field does not have subfields
                list_fields_no_subfields.append(field) 
            else: 
                temp_list = []
                for subfield in dict_protocol_tree[field]: # exmaine subfields exist physically or not
                    if subfield in physical_nodes: 
                        temp_list.append(subfield) 
                list_fields_subfields.append({
                    'field': field, 
                    'subfields': temp_list, 
                    'is_logical': False
                }) 
        else: # tls.handshake and tls.record are all logical nodes 
            temp_list = []
            for subfield in dict_protocol_tree[field]: 
                if subfield in physical_nodes: 
                    temp_list.append(subfield) 
            list_fields_subfields.append({
                    'field': field, 
                    'subfields': temp_list, 
                    'is_logical': True
                }) 
    return list_fields_subfields, list_fields_no_subfields 

# def generate_vocabulary(csv_path, categorical_fields, output_path):
#     """
#     从CSV文件中为指定的分类字段生成词典映射，并保存为YAML文件。

#     :param csv_path: 输入的CSV文件路径。
#     :param categorical_fields: 需要为其创建词典的字段名称列表。
#     :param output_path: 保存生成的词典映射的YAML文件路径。
#     """
#     print(f"Reading data from: {csv_path}")
#     df = pd.read_csv(csv_path)
    
#     master_vocab = {}
    
#     for field in tqdm(categorical_fields, desc="Processing fields ..."):
#         if field not in df.columns:
#             print(f"Warning: Field '{field}' not found in CSV, skipping.")
#             continue
            
#         # print(f"Processing field: '{field}'") 
        
#         unique_values = df[field].dropna().unique()
        
#         # 将所有值统一为小写的字符串，以便处理
#         unique_str_values = sorted([
#             f'{int(v):x}' if isinstance(v, (int, float)) else str(v).lower().replace('0x','')
#             for v in unique_values
#         ])
        
#         vocab_map = {val: i for i, val in enumerate(unique_str_values)}
#         vocab_map['__OOV__'] = len(vocab_map)
        
#         master_vocab[field] = vocab_map
        
#         print(f"  - Found {len(unique_str_values)} unique values. Vocab size (incl. OOV): {len(vocab_map)}")

#     # ==================== 核心修改点 开始 ====================
#     # 定义一个函数，告诉PyYAML如何表示一个字符串
#     def str_presenter(dumper, data):
#         """
#         如果字符串中包含换行符，则使用'|'风格，否则使用普通标量。
#         这会强制PyYAML将所有str对象视为YAML的字符串类型。
#         """
#         if len(data.splitlines()) > 1:
#             return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
#         return dumper.represent_scalar('tag:yaml.org,2002:str', data)

#     # 在dump之前，将这个表示器添加到PyYAML的默认Dumper中
#     yaml.add_representer(str, str_presenter)
#     # ==================== 核心修改点 结束 ====================    

#     # 2. 将主词典写入YAML文件
#     print(f"\nSaving master vocabulary to: {output_path}")
#     with open(output_path, 'w') as f:
#         # 使用 yaml.dump 来写入文件
#         # default_flow_style=False 使其格式更易读（类似块状），而不是单行
#         yaml.dump(master_vocab, f, default_flow_style=False, sort_keys=False) 
        
#     print("Vocabulary generation complete!")
#     return master_vocab
def generate_vocabulary(csv_path, categorical_fields, output_path):
    """
    从CSV文件中为指定的分类字段生成词典映射，并保存为YAML文件。
    (使用强制风格的自定义Dumper以保证键的类型正确)
    """
    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    master_vocab = {}
    
    for field in tqdm(categorical_fields, desc="Processing fields..."):
        if field not in df.columns:
            continue
        
        unique_values = df[field].dropna().unique()
        
        unique_str_values = sorted([
            f'{int(v):x}' if isinstance(v, (int, float)) else str(v).lower().replace('0x','')
            for v in unique_values
        ])
        
        vocab_map = {val: i for i, val in enumerate(unique_str_values)}
        vocab_map['__OOV__'] = len(vocab_map)
        
        master_vocab[field] = vocab_map

    # ==================== 最终核心修改点 开始 ====================

    # 1. 定义一个强制使用单引号风格的字符串表示器
    def quoted_str_presenter(dumper, data):
        """
        这个表示器会强制将所有字符串用单引号括起来。
        """
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

    # 2. 创建一个我们自己的 Dumper 类
    class QuotedDumper(yaml.Dumper):
        pass

    # 3. 将我们的强制规则只添加到我们自己的 Dumper 类上
    QuotedDumper.add_representer(str, quoted_str_presenter)

    # ==================== 最终核心修改点 结束 ==================== 

    print(f"\nSaving master vocabulary to: {output_path}")
    with open(output_path, 'w') as f:
        # 4. 在调用 yaml.dump 时，明确指定使用我们自定义的 QuotedDumper
        yaml.dump(
            master_vocab, 
            f, 
            Dumper=QuotedDumper,  # 强制使用我们的规则
            default_flow_style=False, 
            sort_keys=False
        ) 
        
    print("Vocabulary generation complete!")
    return master_vocab