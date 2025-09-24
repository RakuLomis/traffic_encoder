import pandas as pd
import os 
import numpy as np 
from typing import Optional 
from tqdm import tqdm 
from typing import Literal 
import yaml
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split

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


def to_integer_code(df: pd.DataFrame, col_name = 'tcp.reassembled_segments'): 
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

    all_column_names = df_ori.columns.tolist()

    for block_name, columns, rows in tqdm(zip(block_values, columns_values, rows_values)): 
        # if 0 not in columns: # frame_num must be added as index
        #     columns.append(0) 
        # after merging, frame_num can not be index any more. 
        try:
            subset_column_names = [all_column_names[i] for i in columns]
        except IndexError:
            print(f"警告：块 {block_name} 的列索引超出了范围，跳过此块。")
            continue

        # 2. 核心修改：确保'index'列始终被包含，以保持联系
        if 'index' not in subset_column_names:
            subset_column_names.insert(0, 'index') # 插入到最前面 

        if 'label' not in subset_column_names: 
            subset_column_names.append('label')

        # 3. 使用行的索引标签（.loc）和列名进行切片
        try:
            subset_rows = df_ori.loc[rows]
            sub_df = subset_rows[subset_column_names]
            list_block.append(sub_df)
        except KeyError:
             print(f"警告：块 {block_name} 的某些行或列标签在DataFrame中不存在，跳过此块。")
             continue
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

def add_root_layer(ptree: Dict[str, List[str]]): 
    protocols = ['eth', 'ip', 'tcp', 'tls'] 
    ptree['root'] = [p for p in protocols if p in ptree] 

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

def generate_vocabulary(csv_path, categorical_fields, output_path):
    """
    从CSV文件中为指定的分类字段生成词典映射，并保存为YAML文件。
    (使用强制风格的自定义Dumper以保证键的类型正确)
    """
    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    
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

def label_and_merge_csvs(root_directory: str, output_directory: str, need_label=False): 
    """
    遍历根目录下的所有子文件夹，将子文件夹名作为标签添加到每个CSV文件中，
    然后按标签合并所有CSV文件。

    :param root_directory: 包含标签子文件夹的根目录路径。
    :param output_directory: 保存合并后CSV文件的输出目录路径。
    """
    # 1. 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    print(f"We get output directory: {output_directory}")

    # 2. 获取所有代表标签的子文件夹名称
    try:
        label_folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]
    except FileNotFoundError:
        print(f"Error: The root directory '{root_directory}' was not found.")
        return

    if not label_folders:
        print(f"No label subdirectories found in '{root_directory}'.")
        return

    print(f"Found {len(label_folders)} labels: {label_folders}")

    # 3. 遍历每一个标签文件夹
    for label in tqdm(label_folders, desc="Processing labels"):
        label_path = os.path.join(root_directory, label)
        
        # 找到该标签文件夹下的所有CSV文件
        try:
            csv_files = [f for f in os.listdir(label_path) if f.endswith('.csv')]
        except FileNotFoundError:
            print(f"Warning: Directory for label '{label}' not found at '{label_path}', skipping.")
            continue

        if not csv_files:
            print(f"Warning: No CSV files found for label '{label}', skipping.")
            continue
        
        # 准备一个列表，用来存放该标签下的所有DataFrame
        list_of_dfs_for_label = []
        
        # 4. 读取每个CSV，添加标签，并存入列表 
        # We decide not to add label in this part. Labels will be added after field block truncation. 
        for csv_file in tqdm(csv_files, desc=f"Reading files for '{label}'", leave=False):
            csv_path = os.path.join(label_path, csv_file)
            try:
                df = pd.read_csv(csv_path, dtype=str)
                # 添加新的'label'列
                if need_label: 
                    df['label'] = label
                list_of_dfs_for_label.append(df)
            except Exception as e:
                print(f"Error reading or processing {csv_path}: {e}")

        # 5. 如果列表不为空，则将所有DataFrame合并成一个
        if list_of_dfs_for_label:
            print(f"\nMerging {len(list_of_dfs_for_label)} CSV files for label '{label}'...")
        # 使用concat进行合并，ignore_index=True会重新生成一个连续的索引
            merged_df = pd.concat(list_of_dfs_for_label, ignore_index=True)

            # 6. 构造输出文件名并保存
            # We do not need `merged' as a postfix.
            if need_label: 
                output_filename = f"{label}_label.csv" 
            else: 
                output_filename = f"{label}.csv" 
            output_path = os.path.join(output_directory, output_filename)

            merged_df.to_csv(output_path, index=False)
            print(f"Successfully saved merged file for '{label}' to '{output_path}'. Shape: {merged_df.shape}")
        else:
            print(f"No dataframes were created for label '{label}'.") 

def merge_csvs_with_different_columns(root_directory: str, output_filepath: str, prefix: Optional[str]=None, postfix: Optional[str]=None):
    """
    遍历目录下的所有CSV文件，将它们合并成一个单一的CSV文件，
    即使它们的列名不完全相同也能正确处理。

    :param root_directory: 包含待合并CSV文件的目录路径。
    :param output_filepath: 最终合并后的CSV文件的保存路径。
    """
    # 1. 检查输入目录是否存在
    if not os.path.isdir(root_directory):
        print(f"错误：目录 '{root_directory}' 不存在。")
        return

    # 2. 找到所有需要合并的CSV文件
    # 假设它们都以 '_label.csv' 结尾
    try:
        # csv_files = [f for f in os.listdir(root_directory) if f.endswith('_label.csv')]
        csv_file_paths, csv_files = get_file_path(root_directory, prefix, postfix)
    except Exception as e:
        print(f"读取目录 '{root_directory}' 时出错: {e}")
        return

    if not csv_files:
        print(f"No target files in '{root_directory}'. ")
        return

    print(f"找到了 {len(csv_files)} 个待合并的CSV文件。")

    # 3. 准备一个列表，用来存放从每个CSV文件中读取出的DataFrame
    list_of_dfs = []

    # 4. 遍历文件列表，读取数据并添加到列表中
    for filename in tqdm(csv_files, desc="正在读取CSV文件"):
        filepath = os.path.join(root_directory, filename + postfix)
        try:
            # 读取时将所有列都当作字符串处理，可以避免因类型推断错误导致的警告
            df = pd.read_csv(filepath, dtype=str)
            list_of_dfs.append(df)
        except Exception as e:
            print(f"\n读取文件 {filepath} 时出错: {e}")

    # 5. 检查是否成功读取了任何数据
    if not list_of_dfs:
        print("未能成功读取任何CSV文件，程序退出。")
        return

    # 6. 使用pd.concat进行合并
    # 这会自动处理列名不同的情况，缺失的列会用NaN填充
    print("\n正在合并所有数据...")
    merged_df = pd.concat(list_of_dfs, ignore_index=True, sort=False)
    print("合并完成！")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 7. 将最终的DataFrame保存到单一的CSV文件中
    print(f"正在将合并后的数据保存到: {output_filepath}")
    merged_df.to_csv(output_filepath, index=False)
    
    print("\n处理完毕！")
    print(f"最终合并文件的形状 (行, 列): {merged_df.shape}")
    print(f"总列数: {len(merged_df.columns)}") 

def truncate_to_block_by_schema(source_csv_path: str, output_dir_path: str): 
    os.makedirs(output_dir_path, exist_ok=True) 
    try: 
        df = pd.read_csv(source_csv_path, dtype=str, low_memory=False) 
    except FileNotFoundError: 
        print(f"Source file is not found: {source_csv_path}") 
        return 
    print(f"Data loading completed, {len(df)} records here. ") 

    df.reset_index(inplace=True) # add index
    # meta_columns = ['index', 'label', 'label_id'] 
    cols_to_drop = []
    if 'frame_num' in df.columns:
        cols_to_drop.append('frame_num') 
    if 'tcp.reassembled_segments' in df.columns: 
        cols_to_drop.append('tcp.reassembled_segments') 
    feature_columns = [col for col in df.columns if col not in cols_to_drop] 

    notna_mask = df[feature_columns].notna() 

    # eth.dst|ip.src|tcp.dstport|...
    fingerprints = notna_mask.apply(
        lambda row: '|'.join(row.index[row]), 
        axis=1
    )
    print("Fingerprints has been built. ") 

    grouped = df.groupby(fingerprints) 
    num_blocks = len(grouped) 
    print(f"There are {num_blocks} Field Block in total. ") 

    block_counter = 0 
    for fingerprint, group_df in tqdm(grouped, desc="Saving blocks"): 
        block_cleaned = group_df.drop(columns=cols_to_drop).dropna(axis=1, how='all')
        
        output_filename = f"{block_counter}.csv"
        output_path = os.path.join(output_dir_path, output_filename)
        
        block_cleaned.to_csv(output_path, index=False)
        block_counter += 1
    print(f"Truncation succeeded! ")


def generate_summary_tables(
    directory_path: str, 
    label_output_path: str, 
    feature_output_path: str
):
    """
    分析Block目录，并生成两张总结表：
    1. 标签分布矩阵 (label vs. block)
    2. 特征存在矩阵 (feature vs. block)
    """
    print(f"开始分析目录中的所有Blocks: {directory_path}")

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 目录不存在 -> {directory_path}")
        return

    all_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
    if not all_files:
        print("错误: 在指定目录中未找到任何CSV文件。")
        return

    # --- 步骤一：数据收集 ---
    # 我们需要一次遍历，收集所有必要信息
    print("正在收集中所有Block的标签和字段信息...")
    
    label_distributions: Dict[str, pd.Series] = {}
    block_schemas: Dict[str, Set[str]] = {}
    all_unique_labels: Set[str] = set()
    all_unique_fields: Set[str] = set()
    
    for filename in tqdm(all_files, desc="Collecting Data"):
        block_name = os.path.splitext(filename)[0]
        block_path = os.path.join(directory_path, filename)
        
        try:
            # 只读取必要的列以提高速度
            # 我们需要所有列来确定schema，但只需要'label'列来统计
            df = pd.read_csv(block_path, dtype=str, usecols=lambda col: col != 'index')
            
            # a) 收集标签分布
            if 'label' in df.columns:
                counts = df['label'].value_counts()
                label_distributions[block_name] = counts
                all_unique_labels.update(counts.index)
            
            # b) 收集字段schema
            meta_columns = {'label', 'label_id'} # 'index'已经被usecols排除了
            feature_columns = set(df.columns) - meta_columns
            block_schemas[block_name] = feature_columns
            all_unique_fields.update(feature_columns)
            
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")

    if not label_distributions or not block_schemas:
        print("未能成功收集到任何有效数据。")
        return

    print("数据收集完成。正在生成表格...")

    # --- 步骤二：构建并保存第一张表 (标签分布矩阵) ---
    try:
        print(f"正在构建标签分布矩阵 ({len(all_unique_labels)} L x {len(label_distributions)} B)...")
        # pd.DataFrame能够非常智能地处理Series字典，自动对齐索引
        label_df = pd.DataFrame(label_distributions)
        
        # 将缺失值（NaN）替换为0，表示该Block中没有该label的样本
        label_df = label_df.fillna(0).astype(int)
        
        # 排序以获得更清晰的视图
        label_df = label_df.sort_index() # 按标签名字母排序
        label_df = label_df.reindex(sorted(label_df.columns), axis=1) # 按Block名排序
        
        label_df.to_csv(label_output_path)
        print(f" -> 标签分布矩阵已成功保存到: {label_output_path}")
        
    except Exception as e:
        print(f"构建或保存标签分布矩阵时出错: {e}")
        
    # --- 步骤三：构建并保存第二张表 (特征存在矩阵) ---
    try:
        print(f"正在构建特征存在矩阵 ({len(all_unique_fields)} F x {len(block_schemas)} B)...")
        
        # 转换为有序列表
        sorted_fields = sorted(list(all_unique_fields))
        sorted_blocks = sorted(list(block_schemas.keys()))
        
        # 创建一个全零的DataFrame作为基础
        feature_df = pd.DataFrame(0, index=sorted_fields, columns=sorted_blocks, dtype=int)
        
        # 遍历我们收集的schema信息，将存在特征的位置填充为1
        for block_name, schema in tqdm(block_schemas.items(), desc="Populating Feature Matrix"):
            # 使用.loc进行高效的批量赋值
            feature_df.loc[list(schema), block_name] = 1
            
        feature_df.to_csv(feature_output_path)
        print(f" -> 特征存在矩阵已成功保存到: {feature_output_path}")

    except Exception as e:
        print(f"构建或保存特征存在矩阵时出错: {e}")


def generate_protocol_tree_and_nodes(columns: List[str]) -> (Dict[str, List[str]], Set[str]): # type: ignore
    """
    从列名中，自动发现所有真实节点和隐含的抽象节点。
    """
    tree = defaultdict(set)
    all_nodes = set(columns)
    for col in columns:
        parts = col.split('.')
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            child = ".".join(parts[:i+1])
            tree[parent].add(child)
            all_nodes.add(parent)
    tree_final = {parent: sorted(list(children)) for parent, children in tree.items()}
    return tree_final, all_nodes

def global_stratified_split(
    source_csv_path: str, 
    output_dir: str, 
    test_size: float = 0.1, 
    validation_size: float = 0.1
):
    """
    对一个大的CSV文件进行全局的、分层的 train-validation-test 分割。

    :param source_csv_path: 包含所有原始数据的总CSV文件路径。
    :param output_dir: 保存分割后的三个CSV文件的目录路径。
    :param test_size: 测试集在总数据中所占的比例。
    :param validation_size: 验证集在总数据中所占的比例。
    """
    print("="*50)
    print("开始执行全局数据集分割...")
    print("="*50)
    
    # --- 1. 创建输出目录 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已准备好: {output_dir}")

    # --- 2. 加载总数据集 ---
    print(f"正在从 {source_csv_path} 加载总数据...")
    try:
        # 始终使用 dtype=str 来确保数据完整性
        df = pd.read_csv(source_csv_path, dtype=str)
    except FileNotFoundError:
        print(f"错误: 源文件未找到 -> {source_csv_path}")
        return

    # 检查'label'列是否存在，这是分层抽样的依据
    if 'label' not in df.columns:
        print("错误: 数据集中缺少 'label' 列，无法进行分层抽样。")
        return
        
    print(f"数据加载完成，共 {len(df)} 条记录。")

    # --- 3. 执行两步分割以得到三份数据集 ---
    
    # a) 第一步：从总数据中分割出【测试集】
    print(f"\n[步骤 1/2] 分割出 {test_size:.0%} 的测试集...")
    remaining_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,  # 确保每次分割结果都一样
        stratify=df['label'] # 【关键】进行分层抽样
    )
    
    # b) 第二步：从剩余数据中分割出【验证集】
    #    注意：这里的test_size需要重新计算，以确保验证集占【原始总数据】的比例
    val_split_ratio = validation_size / (1.0 - test_size)
    print(f"[步骤 2/2] 从剩余数据中分割出 {val_split_ratio:.1%} 的验证集 (相当于总数据的 {validation_size:.0%})...")
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=val_split_ratio,
        random_state=42,
        stratify=remaining_df['label'] # 【关键】再次进行分层抽样
    )
    
    # --- 4. 打印总结并保存文件 ---
    print("\n分割完成！各数据集规模如下:")
    print(f" - 训练集 (Train Set): {len(train_df)} 条 (~{(1-test_size-validation_size):.0%})")
    print(f" - 验证集 (Validation Set): {len(val_df)} 条 (~{validation_size:.0%})")
    print(f" - 测试集 (Test Set): {len(test_df)} 条 (~{test_size:.0%})")
    
    train_path = os.path.join(output_dir, 'train_set.csv')
    val_path = os.path.join(output_dir, 'validation_set.csv')
    test_path = os.path.join(output_dir, 'test_set.csv')
    
    print(f"\n正在保存文件...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f" - 训练集已保存到: {train_path}")
    print(f" - 验证集已保存到: {val_path}")
    print(f" - 测试集已保存到: {test_path}")
    print("\n全局数据集分割步骤已成功完成！")

def augment_main_block(
    block_dir: str, 
    main_block_name: str, 
    output_path: str, 
    min_samples_threshold: int = 3000
):
    """
    实现“靶向数据补充”策略。
    此版本不考虑结构相似性，旨在最大化数据覆盖度。

    :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
    :param main_block_name: 作为基础的Main Block的名称 (例如 '24')。
    :param output_path: 保存增强后的数据集的CSV文件路径。
    :param min_samples_threshold: 定义稀有类别的样本数阈值。
    """
    print("="*50)
    print("开始执行“最大化覆盖”的数据补充策略...")
    print("="*50)
    main_block_path = os.path.join(block_dir, f"{main_block_name}.csv")
    
    # 1. 加载主Block，并确定其目标Schema
    print(f"\n[步骤 1/4] 加载 Main Block '{main_block_name}'...")
    if not os.path.exists(main_block_path):
        print(f"错误: Main Block文件未找到 -> {main_block_path}")
        return
        
    main_df = pd.read_csv(main_block_path, dtype=str)
    # 存储主Block的列顺序，以备后用
    main_df_columns = main_df.columns.tolist()
    
    # 2. 扫描所有Block，建立一个关于类别分布的“情报数据库”
    print("\n[步骤 2/4] 扫描所有Block，建立类别分布情报库...")
    block_info = {}
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    for filename in tqdm(all_files, desc="Scanning Blocks"):
        block_name = os.path.splitext(filename)[0]
        block_path = os.path.join(block_dir, filename)
        try:
            df_label = pd.read_csv(block_path, dtype=str, usecols=['label'])
            if not df_label.empty:
                block_info[block_name] = df_label['label'].value_counts().to_dict()
        except Exception as e:
            print(f"\n扫描 {filename} 时出错: {e}")

    # 3. 找出Main Block中需要补充的“靶向类别”
    main_label_counts = main_df['label'].value_counts()
    target_classes = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    
    all_labels_in_db = set(l for stats in block_info.values() for l in stats)
    missing_classes = list(all_labels_in_db - set(main_label_counts.index))
    target_classes.extend(missing_classes)
    
    print(f"\n[步骤 3/4] 在 Main Block 中找到 {len(target_classes)} 个需要补充的类别:")
    print(sorted(target_classes))

    # 4. 遍历靶向类别，寻找【所有】捐献者并合并数据
    print(f"\n[步骤 4/4] 开始从所有其他Block中寻找并合并补充数据...")
    dfs_to_concat = [main_df]
    
    for target_class in tqdm(target_classes, desc="Augmenting Classes"):
        # 遍历所有其他Block，寻找所有合格的捐献者
        for donor_name, label_counts in block_info.items():
            if donor_name == main_block_name or target_class not in label_counts:
                continue

            # --- 核心修改点：移除了相似度检查 ---
            # 只要这个Block有我们需要的类别，就直接征用
            
            donor_path = os.path.join(block_dir, f"{donor_name}.csv")
            donor_df = pd.read_csv(donor_path, dtype=str)
            supplement_df = donor_df[donor_df['label'] == target_class]
            
            # 特征空间对齐
            aligned_df = pd.DataFrame()
            for col in main_df_columns:
                if col in supplement_df.columns:
                    aligned_df[col] = supplement_df[col]
                else:
                    # 填充缺失的特征列
                    aligned_df[col] = '0'
            
            dfs_to_concat.append(aligned_df)

    # 5. 将所有数据合并成最终的增强版DataFrame
    print("\n正在合并所有数据...")
    if len(dfs_to_concat) > 1:
        augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        augmented_df = main_df
    
    # 6. 保存结果
    augmented_df.to_csv(output_path, index=False)
    print(f"\n数据补充成功！")
    print(f" - 原始 Main Block 样本数: {len(main_df)}")
    print(f" - 补充后总样本数: {len(augmented_df)}")
    print(f" - 最终数据集已保存到: {output_path}")