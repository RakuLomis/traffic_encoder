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