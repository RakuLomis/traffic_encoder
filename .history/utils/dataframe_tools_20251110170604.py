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
from collections import Counter
import hashlib
import json

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

# def add_root_layer(ptree: Dict[str, List[str]]):
#     """
#     一个健壮的 add_root_layer 函数。
#     它只连接 ptree 中已经存在的顶层节点。
#     """
#     if not ptree:
#         # 如果 ptree 为空 (例如，专家没有任何真实字段)
#         ptree['root'] = []
#         return

#     # 1. 找到 ptree 中所有的“父”节点 (e.g., {'eth', 'ip.flags'})
#     parent_nodes = set(ptree.keys())
    
#     # 2. 找到 ptree 中所有的“子”节点 (e.g., {'eth.src', 'ip.flags.df'})
#     child_nodes = set()
#     for children in ptree.values():
#         child_nodes.update(children)
        
#     # 3. 顶层节点 = 那些 *只作为父*，而 *不作为子* 出现的节点
#     #    (例如: 'eth' 是顶层, 但 'ip.flags' 不是, 因为 'ip' 是它的父)
#     #    (在这个例子中，set(['eth', 'ip.flags']) - set(['eth.src', 'ip.flags.df']))
#     top_level_nodes = parent_nodes - child_nodes
    
#     # 4. 将 'root' 连接到所有找到的顶层节点
#     ptree['root'] = sorted(list(top_level_nodes))

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

# ==============================================================================
# 2. 【核心修改点】内存优化版的词典生成函数
# ==============================================================================

def generate_vocabulary_memory_optimized(
    csv_path: str, 
    categorical_fields: List[str], 
    output_path: str, 
    chunk_size: int = 100000 # 每次只从CSV读取10万行
):
    """
    【内存优化版】
    从一个【巨大】的CSV文件中，为指定的分类字段生成词典映射。
    使用“分块读取”策略，以保持恒定且低的内存占用。
    """
    print(f"Starting memory-optimized vocabulary generation from: {csv_path}")
    
    # --- 步骤 1: 预检查，并确定需要读取的列 ---
    try:
        # 只读表头，快速检查
        df_header = pd.read_csv(csv_path, dtype=str, nrows=0)
    except FileNotFoundError:
        print(f"错误: 源文件未找到 -> {csv_path}")
        return None
    except Exception as e:
        print(f"读取 {csv_path} 表头时出错: {e}")
        return None
        
    # 筛选出文件中真正存在的、我们关心的列
    columns_to_read = [col for col in categorical_fields if col in df_header.columns]
    if not columns_to_read:
        print(f"警告: 在 {csv_path} 中未找到任何指定的分类字段。")
        return None
    
    # --- 步骤 2: 初始化“唯一值”收集器 ---
    # 我们使用一个字典，键是字段名，值是一个集合 (set)
    # 集合 (set) 会自动处理重复值，内存效率很高
    master_unique_values = defaultdict(set)

    print(f"Reading file in chunks of {chunk_size} rows...")
    
    # 计算总行数以便显示Tqdm进度条 (这是一个快速的近似方法)
    try:
        total_rows = sum(1 for row in open(csv_path, 'r', encoding='utf-8')) - 1
        num_chunks = (total_rows // chunk_size) + 1
    except Exception:
        num_chunks = None # 如果文件过大或编码有问题，则不显示总进度

    # --- 步骤 3: 分块读取并收集唯一值 ---
    with pd.read_csv(csv_path, dtype=str, usecols=columns_to_read, chunksize=chunk_size) as reader:
        for chunk_df in tqdm(reader, total=num_chunks, desc="Processing Chunks"):
            for field in columns_to_read:
                # 1. 从当前块中获取非空、唯一的值
                # 2. 将这些值更新到我们的主集合中
                master_unique_values[field].update(chunk_df[field].dropna().unique())

    print(f"\nUnique value collection complete. Found {len(master_unique_values)} fields.")

    # --- 步骤 4: 处理收集到的值并构建词典 (逻辑与您之前相同) ---
    master_vocab = {}
    for field in tqdm(master_unique_values.keys(), desc="Building vocabs"):
        unique_values_set = master_unique_values[field]
        
        # 我们使用一个更健壮的方式来处理str/int混合
        unique_str_values = sorted([
            f'{int(float(v)):x}' if v.replace('.','',1).isdigit() else str(v).lower().replace('0x','')
            for v in unique_values_set
        ])
        
        vocab_map = {val: i for i, val in enumerate(unique_str_values)}
        vocab_map['__OOV__'] = len(vocab_map) # OOV索引为0
        
        # 将索引值+1，使0号索引保留给OOV
        vocab_map = {val: i+1 for i, val in enumerate(unique_str_values)}
        vocab_map['__OOV__'] = 0
        
        master_vocab[field] = vocab_map

    # --- 步骤 5: 保存YAML文件 (使用您自定义的Dumper，逻辑不变) ---
    print(f"\nSaving master vocabulary to: {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(
            master_vocab, 
            f, 
            Dumper=QuotedDumper, 
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


def truncate_to_block_by_schema_memory_optimized(
    source_csv_path: str, 
    output_dir_path: str, 
    chunk_size: int = 100000
):
    """
    【Memory-Optimized Version】
    Groups a large CSV by its feature schema and saves each group to a separate file.
    This version reads the source file in chunks to keep memory usage low and constant.

    :param source_csv_path: Path to the large source CSV file.
    :param output_dir_path: Directory to save the partitioned Block CSV files.
    :param chunk_size: The number of rows to process in each chunk.
    """
    print("="*60)
    print("### Starting Memory-Optimized Field Block Truncation ###")
    print("="*60)

    os.makedirs(output_dir_path, exist_ok=True)
    
    try:
        # We use a 'with' statement for robust file handling
        with pd.read_csv(source_csv_path, dtype=str, low_memory=False, chunksize=chunk_size) as reader:
            
            # Use tqdm to track progress over the entire file
            # We need to calculate the total number of chunks first for tqdm
            total_rows = sum(1 for row in open(source_csv_path)) - 1
            
            for chunk_df in tqdm(reader, total=(total_rows // chunk_size) + 1, desc="Processing Chunks"):
                
                # --- This logic is the same as before, but applied only to the chunk ---
                meta_columns = ['index', 'label', 'label_id']
                feature_columns = [col for col in chunk_df.columns if col not in meta_columns]
                
                notna_mask = chunk_df[feature_columns].notna()
                
                fingerprints = notna_mask.apply(
                    lambda row: '|'.join(row.index[row]), 
                    axis=1
                )
                
                grouped = chunk_df.groupby(fingerprints)
                # -------------------------------------------------------------------

                for fingerprint, group_df in grouped:
                    
                    # Clean the group by dropping all-NaN columns specific to this group
                    block_cleaned = group_df.dropna(axis=1, how='all')
                    
                    # --- File Handling Logic ---
                    # Use a hash of the long fingerprint string for a clean, unique filename
                    fp_hash = hashlib.md5(fingerprint.encode()).hexdigest()
                    output_filename = f"block_{fp_hash}.csv"
                    output_path = os.path.join(output_dir_path, output_filename)
                    
                    # Check if the file already exists to decide whether to write headers
                    write_header = not os.path.exists(output_path)
                    
                    # Append the cleaned data to the correct file
                    block_cleaned.to_csv(
                        output_path, 
                        mode='a', # 'a' stands for append
                        header=write_header, 
                        index=False
                    )

    except FileNotFoundError:
        print(f"Source file not found: {source_csv_path}")
        return
        
    print("\nTruncation succeeded!")
    print("Post-processing: Renaming files to simple integers and creating a map...")

    # --- Post-processing: Rename hashed files to 0.csv, 1.csv, etc. for consistency ---
    mapping = {}
    counter = 0
    for filename in sorted(os.listdir(output_dir_path)):
        if filename.startswith('block_') and filename.endswith('.csv'):
            new_name = f"{counter}.csv"
            mapping[filename] = new_name
            os.rename(os.path.join(output_dir_path, filename), os.path.join(output_dir_path, new_name))
            counter += 1
    
    # Save the mapping for future reference
    map_path = os.path.join(output_dir_path, "fingerprint_to_id_map.json")
    with open(map_path, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Renaming complete. {counter} unique blocks were created.")
    print(f"Filename mapping saved to: {map_path}")

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

def global_stratified_split_memory_optimized(
    source_csv_path: str, 
    output_dir: str, 
    test_size: float = 0.1, 
    validation_size: float = 0.1,
    chunk_size: int = 100000 # 每次处理的行数
):
    """
    【内存优化版】对一个超大的CSV文件进行全局的、分层的 train-validation-test 分割。
    """
    print("="*60)
    print("###   开始执行【内存优化版】的全局数据集分割   ###")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)

    # # --- Pass 1: 扫描文件，只收集标签和行号 ---
    # print(f"\n[Pass 1/3] 正在扫描 {source_csv_path} 以收集标签索引...")
    # label_to_indices = defaultdict(list)
    # header = None
    # total_rows = 0
    # with pd.read_csv(source_csv_path, dtype=str, usecols=['label'], chunksize=chunk_size) as reader:
    #     for chunk in tqdm(reader, desc="Scanning labels"):
    #         if header is None:
    #             # 获取完整的表头，以便后续写入
    #             full_header_df = pd.read_csv(source_csv_path, dtype=str, nrows=0)
    #             header = full_header_df.columns.tolist()

    #         for idx, label in chunk['label'].items():
    #             label_to_indices[label].append(idx)
    #         total_rows += len(chunk)
            
    # print(f" -> 扫描完成。共 {total_rows} 条记录，{len(label_to_indices)} 个唯一类别。")

# --- Pass 1: 扫描文件，只收集标签和行号 ---
    print(f"\n[Pass 1/3] 正在扫描 {source_csv_path} 以收集标签索引...")
    label_to_indices = defaultdict(list)
    header = None
    total_rows = 0
    valid_rows_processed = 0 # <-- [新增] 
     
    with pd.read_csv(source_csv_path, dtype=str, usecols=['label'], chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc="Scanning labels"):
            if header is None:
                # 获取完整的表头
                full_header_df = pd.read_csv(source_csv_path, dtype=str, nrows=0)
                header = full_header_df.columns.tolist()
                if 'label' not in header:
                    raise ValueError("错误：源CSV文件中未找到 'label' 列。")

            # [!! 核心修复 !!]
            # 1. 找到那些 'label' 列的值 *不是* 'label' 的有效行
            #    这会一次性过滤掉所有混入的表头行
            clean_chunk = chunk[chunk['label'] != 'label']
            # 2. 只遍历“干净”的行
            for idx, label in clean_chunk['label'].items():
                label_to_indices[label].append(idx)

            # 3. 更新统计
            valid_rows_processed += len(clean_chunk)
            total_rows += len(chunk) # total_rows 仍然跟踪读取的总行数

    # [修改] 更新打印信息
    print(f" -> 扫描完成。共 {total_rows} 条记录被读取，其中 {valid_rows_processed} 条是有效数据。")
    print(f" -> 发现 {len(label_to_indices)} 个唯一类别")
          
    # --- 在内存中，对【索引】进行分层抽样 ---
    print("\n[Pass 2/3] 正在对索引进行分层抽样...")
    train_indices, val_indices, test_indices = [], [], []

    for label, indices in tqdm(label_to_indices.items(), desc="Splitting indices"):
        # 第一次分割：分出测试集索引
        try:
            remaining_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        except ValueError: # 如果某个类别的样本太少，无法分割，则全部放入训练集
            train_indices.extend(indices)
            continue
            
        # 第二次分割：从剩余索引中分出验证集索引
        val_split_ratio = validation_size / (1.0 - test_size)
        try:
            train_idx, val_idx = train_test_split(remaining_idx, test_size=val_split_ratio, random_state=42)
        except ValueError:
            train_indices.extend(remaining_idx)
            test_indices.extend(test_idx)
            continue

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    # 将索引列表转换为集合，以获得O(1)的查找速度
    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)
    test_indices_set = set(test_indices)

    print("\n分割完成！各数据集规模如下:")
    print(f" - 训练集 (Train Set): {len(train_indices_set)} 条")
    print(f" - 验证集 (Validation Set): {len(val_indices_set)} 条")
    print(f" - 测试集 (Test Set): {len(test_indices_set)} 条")

    # --- Pass 2: 逐块读取，并将数据写入对应的文件 ---
    print(f"\n[Pass 3/3] 正在逐块读取并写入分割后的文件...")
    
    train_path = os.path.join(output_dir, 'train_set.csv')
    val_path = os.path.join(output_dir, 'validation_set.csv')
    test_path = os.path.join(output_dir, 'test_set.csv')

    # 首先，为三个输出文件写入表头
    pd.DataFrame(columns=header).to_csv(train_path, index=False)
    pd.DataFrame(columns=header).to_csv(val_path, index=False)
    pd.DataFrame(columns=header).to_csv(test_path, index=False)
    
    with pd.read_csv(source_csv_path, dtype=str, chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc="Writing split files"):
            # 根据索引，筛选出属于每个集合的行
            train_chunk = chunk[chunk.index.isin(train_indices_set)]
            val_chunk = chunk[chunk.index.isin(val_indices_set)]
            test_chunk = chunk[chunk.index.isin(test_indices_set)]
            
            # 以追加模式写入，不写表头
            if not train_chunk.empty:
                train_chunk.to_csv(train_path, mode='a', header=False, index=False)
            if not val_chunk.empty:
                val_chunk.to_csv(val_path, mode='a', header=False, index=False)
            if not test_chunk.empty:
                test_chunk.to_csv(test_path, mode='a', header=False, index=False)

    print(f"\n文件写入完成！")
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

def format_bytes(size_bytes: int) -> str:
    """一个辅助函数，将字节数转换为可读的KB/MB/GB。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

# def augment_main_block_v2(
#     block_dir: str, 
#     output_path: str, 
#     min_samples_threshold: int = 3000
# ):
#     """
#     【全自动版】实现“靶向数据补充”策略。
#     自动发现样本量最大的Block作为Chief Block，然后用其他Block对其进行增强。

#     :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
#     :param output_path: 保存增强后的数据集的CSV文件路径。
#     :param min_samples_threshold: 定义稀有类别的样本数阈值。
#     """
#     print("="*50)
#     print("开始执行【全自动】“最大化覆盖”的数据补充策略...")
#     print("="*50)
    
#     # ==================== 核心修改点 1：自动发现Chief Block ====================
#     print("\n[步骤 1/5] 正在扫描所有Block以确定Chief Block (样本量最大)...")
    
#     all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
#     if not all_files:
#         print(f"错误: 在目录 {block_dir} 中未找到任何CSV文件。")
#         return

#     block_sample_counts = {}
#     for filename in tqdm(all_files, desc="Finding largest block"):
#         try:
#             # 通过快速计算行数来确定样本量，避免加载整个文件
#             row_count = sum(1 for row in open(os.path.join(block_dir, filename))) - 1
#             block_sample_counts[os.path.splitext(filename)[0]] = row_count
#         except Exception as e:
#             print(f"扫描文件 {filename} 时出错: {e}")
            
#     if not block_sample_counts:
#         print("错误：未能成功扫描任何Block。")
#         return
        
#     # 找到样本量最大的那个Block的名称
#     main_block_name = max(block_sample_counts, key=block_sample_counts.get)
#     print(f" -> 自动选定 '{main_block_name}' 作为Chief Block (样本数: {block_sample_counts[main_block_name]})。")
#     # =======================================================================
    
#     main_block_path = os.path.join(block_dir, f"{main_block_name}.csv")
    
#     # 2. 加载主Block，并确定其目标Schema
#     print(f"\n[步骤 2/5] 加载 Chief Block '{main_block_name}'...")
#     main_df = pd.read_csv(main_block_path, dtype=str)
#     main_df_columns = main_df.columns.tolist()
    
#     # 3. 扫描所有Block，建立类别分布的“情报数据库”
#     print("\n[步骤 3/5] 扫描所有Block，建立类别分布情报库...")
#     block_info = {}
#     for filename in tqdm(all_files, desc="Scanning Blocks for labels"):
#         block_name = os.path.splitext(filename)[0]
#         block_path = os.path.join(block_dir, filename)
#         try:
#             df_label = pd.read_csv(block_path, dtype=str, usecols=['label'])
#             if not df_label.empty:
#                 block_info[block_name] = df_label['label'].value_counts().to_dict()
#         except Exception as e:
#             print(f"\n扫描 {filename} 时出错: {e}")

#     # 4. 找出Main Block中需要补充的“靶向类别”
#     main_label_counts = main_df['label'].value_counts()
#     target_classes = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    
#     all_labels_in_db = set(l for stats in block_info.values() for l in stats)
#     missing_classes = list(all_labels_in_db - set(main_label_counts.index))
#     target_classes.extend(missing_classes)
    
#     print(f"\n[步骤 4/5] 在 Chief Block 中找到 {len(target_classes)} 个需要补充的类别:")
#     print(sorted(target_classes))

#     # 5. 遍历靶向类别，寻找【所有】捐献者并合并数据
#     print(f"\n[步骤 5/5] 开始从所有其他Block中寻找并合并补充数据...")
#     dfs_to_concat = [main_df]
    
#     for target_class in tqdm(target_classes, desc="Augmenting Classes"):
#         for donor_name, label_counts in block_info.items():
#             if donor_name == main_block_name or target_class not in label_counts:
#                 continue
            
#             donor_path = os.path.join(block_dir, f"{donor_name}.csv")
#             donor_df = pd.read_csv(donor_path, dtype=str)
#             supplement_df = donor_df[donor_df['label'] == target_class]
            
#             # 特征空间对齐
#             aligned_df = pd.DataFrame()
#             for col in main_df_columns:
#                 if col in supplement_df.columns:
#                     aligned_df[col] = supplement_df[col]
#                 else:
#                     aligned_df[col] = '0'
            
#             dfs_to_concat.append(aligned_df)

#     # 6. 将所有数据合并成最终的增强版DataFrame
#     print("\n正在合并所有数据...")
#     if len(dfs_to_concat) > 1:
#         augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
#     else:
#         augmented_df = main_df
    
#     # 7. 保存结果
#     augmented_df.to_csv(output_path, index=False)
#     print(f"\n数据补充成功！")
#     print(f" - 原始 Chief Block ('{main_block_name}') 样本数: {len(main_df)}")
#     print(f" - 补充后总样本数: {len(augmented_df)}")
#     print(f" - 最终数据集已保存到: {output_path}")

def augment_main_block_v2(
    block_dir: str, 
    output_path: str, 
    min_samples_threshold: int = 3000
):
    """
    【全自动版 v2】实现“靶向数据补充”策略。
    自动发现【文件大小最大】的Block作为Chief Block，然后用其他Block对其进行增强。

    :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
    :param output_path: 保存增强后的数据集的CSV文件路径。
    :param min_samples_threshold: 定义稀有类别的样本数阈值。
    """
    print("="*50)
    print("开始执行【全自动，按文件大小】的数据补充策略...")
    print("="*50)
    
    # ==================== 核心修改点：按“文件大小”自动发现Chief Block ====================
    print("\n[步骤 1/5] 正在扫描所有Block以确定Chief Block (文件大小最大)...")
    
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {block_dir} 中未找到任何CSV文件。")
        return

    block_file_sizes = {}
    for filename in tqdm(all_files, desc="Finding largest block"):
        try:
            # 【关键改动】使用 os.path.getsize 来获取文件的字节大小
            file_path = os.path.join(block_dir, filename)
            file_size = os.path.getsize(file_path)
            block_file_sizes[os.path.splitext(filename)[0]] = file_size
        except Exception as e:
            print(f"扫描文件 {filename} 时出错: {e}")
            
    if not block_file_sizes:
        print("错误：未能成功扫描任何Block。")
        return
        
    # 找到文件大小最大的那个Block的名称
    main_block_name = max(block_file_sizes, key=block_file_sizes.get)
    
    # 【关键改动】更新打印信息，显示文件大小
    max_size_formatted = format_bytes(block_file_sizes[main_block_name])
    print(f" -> 自动选定 '{main_block_name}' 作为Chief Block (文件大小: {max_size_formatted})。")
    # =======================================================================
    
    main_block_path = os.path.join(block_dir, f"{main_block_name}.csv")
    
    # 2. 加载主Block，并确定其目标Schema
    print(f"\n[步骤 2/5] 加载 Chief Block '{main_block_name}'...")
    main_df = pd.read_csv(main_block_path, dtype=str)
    main_df_columns = main_df.columns.tolist()
    
    # 3. 扫描所有Block，建立类别分布的“情报数据库”
    print("\n[步骤 3/5] 扫描所有Block，建立类别分布情报库...")
    block_info = {}
    # 我们可以在这里复用 block_file_sizes 字典的键，避免再次列出文件
    for block_name in tqdm(block_file_sizes.keys(), desc="Scanning Blocks for labels"):
        block_path = os.path.join(block_dir, f"{block_name}.csv")
        try:
            df_label = pd.read_csv(block_path, dtype=str, usecols=['label'])
            if not df_label.empty:
                block_info[block_name] = df_label['label'].value_counts().to_dict()
        except Exception as e:
            print(f"\n扫描 {block_name}.csv 时出错: {e}")

    # 4. 找出Main Block中需要补充的“靶向类别”
    # ... (此步骤及后续所有步骤，与您之前的版本完全相同，无需修改)
    main_label_counts = main_df['label'].value_counts()
    target_classes = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    
    all_labels_in_db = set(l for stats in block_info.values() for l in stats)
    missing_classes = list(all_labels_in_db - set(main_label_counts.index))
    target_classes.extend(missing_classes)
    
    print(f"\n[步骤 4/5] 在 Chief Block 中找到 {len(target_classes)} 个需要补充的类别:")
    print(sorted(target_classes))

    # 5. 遍历靶向类别，寻找【所有】捐献者并合并数据
    print(f"\n[步骤 5/5] 开始从所有其他Block中寻找并合并补充数据...")
    dfs_to_concat = [main_df]
    
    for target_class in tqdm(target_classes, desc="Augmenting Classes"):
        for donor_name, label_counts in block_info.items():
            if donor_name == main_block_name or target_class not in label_counts:
                continue
            
            donor_path = os.path.join(block_dir, f"{donor_name}.csv")
            donor_df = pd.read_csv(donor_path, dtype=str)
            supplement_df = donor_df[donor_df['label'] == target_class]
            
            # 特征空间对齐
            aligned_df = pd.DataFrame()
            for col in main_df_columns:
                if col in supplement_df.columns:
                    aligned_df[col] = supplement_df[col]
                else:
                    aligned_df[col] = '0'
            
            dfs_to_concat.append(aligned_df)

    # 6. 将所有数据合并成最终的增强版DataFrame
    print("\n正在合并所有数据...")
    if len(dfs_to_concat) > 1:
        augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        augmented_df = main_df
    
    # 7. 保存结果
    augmented_df.to_csv(output_path, index=False)
    print(f"\n数据补充成功！")
    print(f" - 原始 Chief Block ('{main_block_name}') 样本数: {len(main_df)}")
    print(f" - 补充后总样本数: {len(augmented_df)}")
    print(f" - 最终数据集已保存到: {output_path}")

def augment_main_block_top_k(
    block_dir: str, 
    output_path: str, 
    min_samples_threshold: int = 3000,
    top_k: int = 3 # <-- 【新】K值，默认为3
):
    """
    【全自动版 v3 - Top-K合并】
    自动发现【文件大小 Top-K】的Block，将它们合并成一个“超级Chief Block”，
    然后再用其他Block对其进行增强。
    """
    print("="*50)
    print(f"开始执行【Top-{top_k}合并】的数据补充策略...")
    print("="*50)
    
    # --- 步骤 1：扫描所有Block，并按文件大小排序 ---
    print(f"\n[步骤 1/6] 正在扫描所有Block以确定Top-{top_k} Chief Blocks...")
    
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {block_dir} 中未找到任何CSV文件。")
        return

    block_file_sizes = {}
    for filename in tqdm(all_files, desc="Finding largest blocks"):
        try:
            file_path = os.path.join(block_dir, filename)
            file_size = os.path.getsize(file_path)
            block_file_sizes[os.path.splitext(filename)[0]] = file_size
        except Exception as e:
            print(f"扫描文件 {filename} 时出错: {e}")
            
    if not block_file_sizes:
        print("错误：未能成功扫描任何Block。")
        return
        
    # 【关键】对所有Block按文件大小进行降序排序
    sorted_blocks = sorted(block_file_sizes.items(), key=lambda item: item[1], reverse=True)
    
    # 选出Top-K个
    top_k_blocks = sorted_blocks[:top_k]
    top_k_names = [name for name, size in top_k_blocks]
    
    print(f" -> 自动选定 Top-{len(top_k_blocks)} Block(s) 作为Chief Block：")
    for name, size in top_k_blocks:
        print(f"    - '{name}' (文件大小: {format_bytes(size)})")

    # --- 步骤 2：加载并合并Top-K Block，创建“超级Chief Block” ---
    print(f"\n[步骤 2/6] 正在加载并合并Top-{len(top_k_blocks)} Block(s)...")
    
    dfs_to_merge = []
    for block_name in top_k_names:
        block_path = os.path.join(block_dir, f"{block_name}.csv")
        dfs_to_merge.append(pd.read_csv(block_path, dtype=str))
        
    # 【关键】使用concat取所有特征的“并集”，并用'0'填充
    #       这会创建出我们需要的、Schema最全的“超级Chief Block”
    main_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False).fillna('0')
    main_df_columns = main_df.columns.tolist() # 这就是我们的“超级Schema”
    
    print(f" -> “超级Chief Block”创建成功。")
    print(f" -> 原始 Top-K 样本数: {len(main_df)}")
    print(f" -> 最终特征（节点）数: {len(main_df_columns) - 2}") # 减去label和label_id
    

    # 3. 扫描所有Block，建立类别分布的“情报数据库”
    print("\n[步骤 3/6] 扫描所有Block，建立类别分布情报库...")
    block_info = {}
    # 我们可以在这里复用 block_file_sizes 字典的键，避免再次列出文件
    for block_name in tqdm(block_file_sizes.keys(), desc="Scanning Blocks for labels"):
        block_path = os.path.join(block_dir, f"{block_name}.csv")
        try:
            df_label = pd.read_csv(block_path, dtype=str, usecols=['label'])
            if not df_label.empty:
                block_info[block_name] = df_label['label'].value_counts().to_dict()
        except Exception as e:
            print(f"\n扫描 {block_name}.csv 时出错: {e}")

    
    # --- 步骤 4：找出Main Block中需要补充的“靶向类别” ---
    # (这个逻辑与之前完全相同)
    main_label_counts = main_df['label'].value_counts()
    target_classes = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    all_labels_in_db = set(l for stats in block_info.values() for l in stats)
    missing_classes = list(all_labels_in_db - set(main_label_counts.index))
    target_classes.extend(missing_classes)
    
    print(f"\n[步骤 4/6] 在 “超级Chief Block” 中找到 {len(target_classes)} 个需要补充的类别:")
    print(sorted(target_classes))

    # --- 步骤 5：遍历靶向类别，寻找【所有】捐献者并合并数据 ---
    print(f"\n[步骤 5/6] 开始从所有其他Block中寻找并合并补充数据...")
    dfs_to_concat = [main_df] # 这一次，我们从“超级Chief Block”开始
    
    for target_class in tqdm(target_classes, desc="Augmenting Classes"):
        for donor_name, label_counts in block_info.items():
            # 【关键】确保我们不会从Top-K Block中重复添加数据
            if donor_name in top_k_names or target_class not in label_counts:
                continue
            
            donor_path = os.path.join(block_dir, f"{donor_name}.csv")
            donor_df = pd.read_csv(donor_path, dtype=str)
            supplement_df = donor_df[donor_df['label'] == target_class]
            
            # 特征空间对齐 (使用“超级Schema”)
            aligned_df = pd.DataFrame()
            for col in main_df_columns:
                if col in supplement_df.columns:
                    aligned_df[col] = supplement_df[col]
                else:
                    aligned_df[col] = '0'
            
            dfs_to_concat.append(aligned_df)

    # 6. 将所有数据合并成最终的增强版DataFrame
    print("\n正在合并所有数据...")
    if len(dfs_to_concat) > 1:
        augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        augmented_df = main_df
    
    # 7. 保存结果
    augmented_df.to_csv(output_path, index=False)
    print(f"\n数据补充成功！")
    print(f" - 原始 Chief Block ('{top_k_names}') 样本数: {len(main_df)}")
    print(f" - 补充后总样本数: {len(augmented_df)}")
    print(f" - 最终数据集已保存到: {output_path}")

def get_top_n_features(df: pd.DataFrame, n: int, existing_schema: Set[str]) -> List[str]:
    """从DataFrame中，找出Top N个最常见的、且不在现有Schema中的特征。"""
    feature_counts = Counter()
    meta_columns = {'index', 'label', 'label_id'}
    
    # 只统计非空值的特征
    for col in df.columns:
        if col not in meta_columns:
            feature_counts[col] = df[col].notna().sum()
            
    # 找出不在现有Schema中的Top N特征
    top_features = []
    # .most_common() 返回 (feature, count) 元组的列表
    for feature, count in feature_counts.most_common():
        if feature not in existing_schema:
            top_features.append(feature)
        if len(top_features) == n:
            break
            
    return top_features

def augment_main_block_with_features(
    block_dir: str, 
    main_block_name: str, 
    output_path: str, 
    min_samples_threshold: int = 2000
):
    """
    实现“动态特征空间扩展”策略的最终、最健壮版本。
    基于单循环的、经过验证的逻辑。
    """
    print("="*50)
    print("开始执行最终的“数据补充与特征扩展”策略...")
    print("="*50)
    main_block_path = os.path.join(block_dir, f"{main_block_name}.csv")
    
    # 1. 加载主Block，并确定其【初始】Schema
    print(f"\n[步骤 1/4] 加载 Main Block '{main_block_name}'...")
    main_df = pd.read_csv(main_block_path, dtype=str)
    meta_columns = {'index', 'label', 'label_id'}
    initial_main_schema = set(main_df.columns)
    
    # 2. 扫描所有Block，建立情报数据库
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
        except Exception:
            pass # 忽略无法读取的文件

    # 3. 找出需要补充的“靶向类别”
    main_label_counts = main_df['label'].value_counts()
    rare_classes_in_main = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    all_labels_in_db = set(l for stats in block_info.values() for l in stats)
    missing_classes_from_main = list(all_labels_in_db - set(main_label_counts.index))
    target_classes = rare_classes_in_main + missing_classes_from_main
    
    print(f"\n[步骤 3/4] 找到 {len(target_classes)} 个需要补充的类别...")

    # ==================== 核心修改点：单循环 + 动态Schema ====================
    
    print("\n[步骤 4/4] 开始寻找补充数据并动态扩展Schema...")
    
    # a) 初始化
    expanded_schema = set(initial_main_schema)
    dfs_to_concat = [main_df] # 先把自己放进去
    
    # b) 开始单循环增强
    for target_class in tqdm(target_classes, desc="Augmenting Classes"):
        
        # 收集该类别在所有其他Block中的所有数据
        supplement_dfs = []
        for donor_name, label_counts in block_info.items():
            if donor_name == main_block_name or target_class not in label_counts:
                continue
            
            # 只要这个Block有我们需要的类别，就直接加载并征用
            donor_path = os.path.join(block_dir, f"{donor_name}.csv")
            donor_df = pd.read_csv(donor_path, dtype=str)
            supplement_dfs.append(donor_df[donor_df['label'] == target_class])
        
        # 安全检查
        if not supplement_dfs:
            print(f"\n  - 注意: 未能为类别 '{target_class}' 找到任何补充数据源。")
            continue
            
        full_supplement_df = pd.concat(supplement_dfs, ignore_index=True)
        
        # 根据类别是否存在于Main Block，决定引入Top N个特征
        n_top = 2 if target_class in missing_classes_from_main else 1
        new_features = get_top_n_features(full_supplement_df, n=n_top, existing_schema=expanded_schema)
        
        if new_features:
            print(f"\n  -> 为类别 '{target_class}' 引入新特征: {new_features}")
            expanded_schema.update(new_features)
            
        # 将这个【未经对齐】的补充数据，暂时存起来
        dfs_to_concat.append(full_supplement_df)
        
    # =======================================================================
    
    # 5. 使用最终扩展后的Schema，对所有收集到的DataFrame进行对齐并合并
    final_schema_list = sorted(list(expanded_schema))
    print(f"\nSchema扩展完成！最终特征数: {len(final_schema_list) - len(meta_columns)}")
    print("正在对所有数据进行最终对齐与合并...")
    
    aligned_dfs = []
    for i, df in enumerate(tqdm(dfs_to_concat, desc="Aligning all DataFrames")):
        aligned_df = pd.DataFrame()
        # 按照最终的Schema来构建
        for col in final_schema_list:
            if col in df.columns:
                aligned_df[col] = df[col]
            else:
                # 只有特征列需要填充
                if col not in meta_columns:
                    aligned_df[col] = '0'
        # 补回元数据列
        for col in meta_columns:
            if col in df.columns:
                 aligned_df[col] = df[col]
        aligned_dfs.append(aligned_df[final_schema_list]) # 确保列顺序

    # 6. 一次性合并所有对齐后的DataFrame
    augmented_df = pd.concat(aligned_dfs, ignore_index=True).fillna('0')
    
    # 7. 保存结果
    augmented_df.to_csv(output_path, index=False)
    print(f"\n数据补充与特征扩展成功！")
    print(f" - 原始 Main Block 样本数: {main_df.shape[0]}")
    print(f" - 增强后总样本数: {augmented_df.shape[0]}")
    print(f" - 最终数据集已保存到: {output_path}")

def create_specialist_dataset(
    block_dir: str, 
    chief_block_name: str, 
    output_path: str,
    target_classes: List[str]
):
    """
    为一个或多个特定类别，创建一个【只包含其独有特征】的、用于验证实验的训练数据集。

    :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
    :param chief_block_name: 作为基础参照的Chief Block的名称 (例如 '24')。
    :param output_path: 保存专家训练集的CSV文件路径。
    :param target_classes: 需要进行专门分析的目标类别列表 (例如 ['google', 'twitter'])。
    """
    print("="*50)
    print("### 创建“独有特征专家模型”专用训练集 ###")
    print(f"### 目标类别: {target_classes} ###")
    print("="*50)
    
    chief_block_path = os.path.join(block_dir, f"{chief_block_name}.csv")
    meta_columns = {'index', 'label', 'label_id'}
    
    # 1. 从所有Block中，收集目标类别的数据
    print(f"\n[步骤 1/4] 正在从所有Block中收集 {', '.join(target_classes)} 的数据...")
    supplement_dfs = []
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    for filename in tqdm(all_files, desc="Collecting specialist data"):
        # 我们也需要从Chief Block自身收集目标类别的样本（如果存在的话）
        # if os.path.splitext(filename)[0] == chief_block_name:
        #     continue
            
        df = pd.read_csv(os.path.join(block_dir, filename), dtype=str)
        specialist_data = df[df['label'].isin(target_classes)]
        if not specialist_data.empty:
            supplement_dfs.append(specialist_data)
    
    if not supplement_dfs:
        print(f"错误: 未能找到任何关于 {target_classes} 的样本。")
        return

    specialist_df = pd.concat(supplement_dfs, ignore_index=True)
    print(f" -> 数据收集完成，共找到 {len(specialist_df)} 个相关样本。")

    # 2. 识别出“独有特征” (Exclusive Features)
    print("\n[步骤 2/4] 正在识别独有特征集...")
    try:
        chief_df_header = pd.read_csv(chief_block_path, dtype=str, nrows=0)
    except FileNotFoundError:
        print(f"错误: Chief Block文件未找到 -> {chief_block_path}")
        return
        
    chief_schema = set(chief_df_header.columns)
    
    # a) 找到专家数据中存在的所有特征
    specialist_schema = set(specialist_df.columns)
    
    # b) 通过集合运算，得到“独有特征”
    exclusive_features = sorted(list(specialist_schema - chief_schema))
    
    # 确保元数据列也被包含
    final_columns = sorted(list(meta_columns.intersection(specialist_df.columns))) + exclusive_features
    
    print(f" -> 发现 {len(exclusive_features)} 个独有特征: {exclusive_features}")
    if not exclusive_features:
        print("警告: 未发现任何独有特征，无法创建专家数据集。")
        return

    # 3. 创建只包含独有特征和元数据的新DataFrame
    print("\n[步骤 3/4] 正在创建只包含独有特征的数据集...")
    
    # 从专家数据中，只选取我们需要的列
    # 使用 .reindex 来确保即使某些行在某些独有特征上没有值，列也会被创建
    exclusive_df = specialist_df.reindex(columns=final_columns)
    
    # 对于独有特征列中的NaN值（代表这个样本虽然属于目标类别，但没有这个特定的独有特征），进行填充
    for col in exclusive_features:
        if exclusive_df[col].hasnans:
            exclusive_df[col].fillna('0', inplace=True)

    # 4. 保存结果
    print("\n[步骤 4/4] 正在保存最终的专家训练集...")
    exclusive_df.to_csv(output_path, index=False)
    print(f"\n专家训练集创建成功！")
    print(f" - 总样本数: {len(exclusive_df)}")
    print(f" - 总特征数: {len(exclusive_features)}")
    print(f" - 最终数据集已保存到: {output_path}")

def create_specialist_dataset_intersect(
    block_dir: str, 
    chief_block_name: str, 
    output_path: str,
    target_classes: List[str]
):
    """
    为一个或多个特定类别，创建一个【只包含其“共同的独有特征”】的、用于验证实验的训练数据集。

    :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
    :param chief_block_name: 作为基础参照的Chief Block的名称 (例如 '24')。
    :param output_path: 保存专家训练集的CSV文件路径。
    :param target_classes: 需要进行专门分析的目标类别列表。
    """
    print("="*50)
    print("### 创建“共同独有特征专家模型”专用训练集 ###")
    print(f"### 目标类别: {target_classes} ###")
    print("="*50)
    
    meta_columns = {'index', 'label', 'label_id'}
    
    # 1. 从所有Block中，收集目标类别的数据，并记录“捐献者”
    print(f"\n[步骤 1/5] 正在从所有Block中收集 {', '.join(target_classes)} 的数据...")
    supplement_dfs = []
    donor_block_names = set() # 用来记录哪些block贡献了数据
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    
    for filename in tqdm(all_files, desc="Collecting specialist data"):
        df = pd.read_csv(os.path.join(block_dir, filename), dtype=str)
        specialist_data = df[df['label'].isin(target_classes)]
        if not specialist_data.empty:
            supplement_dfs.append(specialist_data)
            donor_block_names.add(os.path.splitext(filename)[0])
    
    if not supplement_dfs:
        print(f"错误: 未能找到任何关于 {target_classes} 的样本。")
        return

    specialist_df = pd.concat(supplement_dfs, ignore_index=True)
    print(f" -> 数据收集完成，共找到 {len(specialist_df)} 个相关样本。")
    print(f" -> 数据来源于以下 {len(donor_block_names)} 个'捐献者'Block: {donor_block_names}")

    # ==================== 核心修改点 开始 ====================
    #
    # 2. 识别出“共同的独有特征” (Common Exclusive Features)
    #
    print("\n[步骤 2/5] 正在计算“共同的独有特征”...")
    try:
        chief_df_header = pd.read_csv(os.path.join(block_dir, f"{chief_block_name}.csv"), dtype=str, nrows=0)
        chief_schema = set(chief_df_header.columns)
    except FileNotFoundError:
        print(f"错误: Chief Block文件未找到 -> {chief_block_name}.csv")
        return
        
    # a) 找到所有“捐献者”Block的Schema
    donor_schemas = []
    for donor_name in donor_block_names:
        donor_header = pd.read_csv(os.path.join(block_dir, f"{donor_name}.csv"), dtype=str, nrows=0)
        donor_schemas.append(set(donor_header.columns))

    if not donor_schemas:
        print("警告: 未找到任何捐献者Block的Schema。")
        return

    # b) 计算所有“捐献者”Schema的【交集】
    common_donor_features = donor_schemas[0]
    for schema in donor_schemas[1:]:
        common_donor_features.intersection_update(schema)
        
    # c) 从交集中，减去Chief Block的特征，得到最终的“共同独有特征”
    exclusive_features = sorted(list(common_donor_features - chief_schema))
    
    # ==================== 核心修改点 结束 ====================
    
    final_columns = sorted(list(meta_columns.intersection(specialist_df.columns))) + exclusive_features
    
    print(f" -> 计算完成，找到 {len(exclusive_features)} 个共同独有特征: {exclusive_features}")
    if not exclusive_features:
        print("警告: 未发现任何共同的独有特征，无法创建专家数据集。")
        return

    # 3. 创建只包含【共同独有特征】和元数据的新DataFrame
    print("\n[步骤 3/5] 正在创建只包含独有特征的数据集...")
    exclusive_df = specialist_df.reindex(columns=final_columns)
    for col in exclusive_features:
        if exclusive_df[col].hasnans:
            exclusive_df[col].fillna('0', inplace=True)

    # 4. 保存结果
    print("\n[步骤 4/5] 正在保存最终的专家训练集...")
    exclusive_df.to_csv(output_path, index=False)
    print(f"\n专家训练集创建成功！")
    print(f" - 总样本数: {len(exclusive_df)}")
    print(f" - 总特征数: {len(exclusive_features)}")
    print(f" - 最终数据集已保存到: {output_path}") 

def consolidate_raw_csvs(input_dir: str, output_path: str):
    """
    整合一个目录中的所有原始CSV文件，执行字段筛选、标签合并和对齐。

    :param input_dir: 包含所有原始(raw)CSV文件的目录。
    :param output_path: 保存最终整合后的总数据集的CSV文件路径。
    """
    print("="*60)
    print("###   开始整合与清洗原始CSV文件   ###")
    print("="*60)

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        return
        
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {input_dir} 中未找到任何CSV文件。")
        return

    # 定义我们想要保留的协议层前缀
    PREFIXES_TO_KEEP = ('eth.', 'ip.', 'tcp.', 'tls.')
    
    all_dfs: List[pd.DataFrame] = []

    print(f"\n[步骤 1/3] 正在加载、筛选和清洗 {len(all_files)} 个CSV文件...")
    for filename in tqdm(all_files, desc="Processing raw CSVs"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            # --- 步骤 3: 标签合并 ---
            # 假设标签来源于文件名，例如 'baidu_1.csv' -> 'baidu'
            base_label = os.path.splitext(filename)[0].split('_')[0]
            df['label'] = base_label
            
            # --- 步骤 1: 字段筛选 ---
            original_cols = df.columns.tolist()
            # 我们总是保留 'label' 列，以及所有符合前缀要求的特征列
            cols_to_keep = ['label']
            for col in original_cols:
                if col.startswith(PREFIXES_TO_KEEP):
                    cols_to_keep.append(col)
            
            # 应用筛选
            df_filtered = df[cols_to_keep]
            
            all_dfs.append(df_filtered)

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")
            
    if not all_dfs:
        print("未能成功处理任何CSV文件。")
        return

    # --- 步骤 4: 合并对齐一个总的CSV数据集 ---
    print("\n[步骤 2/3] 正在将所有数据合并为一个总数据集...")
    # pd.concat 会自动取所有列的并集，并在不存在值的地方填充NaN (步骤2)
    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    print(f" -> 合并完成，总数据集包含 {len(consolidated_df)} 条记录和 {len(consolidated_df.columns)} 个字段。")

    # --- 步骤 5: 创建新的frame_num作为索引 ---
    print("\n[步骤 3/3] 正在创建新的唯一索引 'frame_num'...")
    # 先删除可能存在的旧的、不连续的 'frame_num' 列
    if 'frame_num' in consolidated_df.columns:
        consolidated_df = consolidated_df.drop(columns=['frame_num'])
        
    # 创建一个从1开始的新索引
    consolidated_df.insert(0, 'frame_num', range(1, len(consolidated_df) + 1))
    
    # 调整列顺序，将label也放在前面
    if 'label' in consolidated_df.columns:
        cols = consolidated_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('label')))
        consolidated_df = consolidated_df[cols]

    # --- 保存最终结果 ---
    try:
        consolidated_df.to_csv(output_path, index=False)
        print(f"\n总数据集已成功保存到: {output_path}")
    except Exception as e:
        print(f"\n保存总数据集时失败: {e}")

def consolidate_raw_csvs_memory_optimized(input_dir: str, output_path: str):
    """
    【内存优化版】整合一个目录中的所有原始CSV文件。
    采用“两遍扫描，流式写入”的策略，以极低的内存开销处理大量数据。
    """
    print("="*60)
    print("###   开始【内存优化版】的整合与清洗流程   ###")
    print("="*60)

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        return
        
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {input_dir} 中未找到任何CSV文件。")
        return

    PREFIXES_TO_KEEP = ('eth.', 'ip.', 'tcp.', 'tls.')
    
    # --- 第一遍扫描：确定全局Schema ---
    print(f"\n[Pass 1/2] 正在扫描 {len(all_files)} 个文件的表头以确定全局Schema...")
    universal_schema = {'label'} # 始终包含label列
    for filename in tqdm(all_files, desc="Scanning Schemas"):
        file_path = os.path.join(input_dir, filename)
        try:
            # 只读第一行（表头）来获取列名，速度极快，内存占用极小
            df_header = pd.read_csv(file_path, dtype=str, nrows=0)
            
            # 筛选符合前缀的字段
            filtered_cols = {col for col in df_header.columns if col.startswith(PREFIXES_TO_KEEP)}
            universal_schema.update(filtered_cols)
        except Exception as e:
            print(f"\n扫描文件 {filename} 表头时出错: {e}")

    # 确定最终的、有序的列名列表
    final_columns = ['frame_num', 'label'] + sorted(list(universal_schema - {'label'}))
    print(f" -> 全局Schema确定，共 {len(final_columns)} 个最终字段。")

    # --- 第二遍扫描：逐块处理并流式写入 ---
    print(f"\n[Pass 2/2] 正在逐个处理文件并写入到 {output_path}...")
    
    # a) 首先，创建输出文件并写入表头
    pd.DataFrame(columns=final_columns).to_csv(output_path, index=False)
    
    global_row_counter = 0
    for filename in tqdm(all_files, desc="Processing and Appending"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            # 1. 标签合并
            base_label = os.path.splitext(filename)[0].split('_')[0]
            df['label'] = base_label
            
            # 2. 特征空间对齐
            #    - reindex 会自动添加缺失列（值为NaN），并删除多余列
            df_aligned = df.reindex(columns=final_columns)
            
            # 3. 创建新的frame_num (因为是追加模式，需要手动计算)
            num_rows = len(df)
            df_aligned['frame_num'] = range(global_row_counter + 1, global_row_counter + 1 + num_rows)
            global_row_counter += num_rows
            
            # 4. 【核心】将处理好的数据【追加】到输出文件中，不写表头
            df_aligned.to_csv(output_path, mode='a', header=False, index=False)

        except Exception as e:
            print(f"\n处理并写入文件 {filename} 时发生错误: {e}")
    
    print(f"\n总数据集已成功保存到: {output_path}")
    print(f"总计处理了 {global_row_counter} 条记录。")

def stratified_sample_dataframe(
    df: pd.DataFrame, 
    label_column: str = 'label', 
    proportion: float = 0.1, 
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    对一个 DataFrame 进行按比例分层抽样，以缩小其规模。

    此函数确保每个类别（stratum）至少被抽样一次（只要该类别在
    原始数据中至少有一个样本）。

    参数:
    ----------
    df : pd.DataFrame
        需要抽样的原始 DataFrame。
    label_column : str
        用于分层的列名 (例如 'label' 或 'label_id')。
    proportion : float
        要抽样的比例 (例如 0.1 代表 10%)。
    random_state : Optional[int], 默认=42
        用于确保抽样可复现的随机种子。

    返回:
    -------
    pd.DataFrame
        一个新的、按比例分层抽样后的 DataFrame。
    """
    
    print(f"开始分层抽样，目标比例: {proportion * 100:.1f}%...")

    if not (0.0 <= proportion <= 1.0):
        raise ValueError("抽样比例 (proportion) 必须在 0.0 和 1.0 之间。")
        
    if proportion == 1.0:
        print("比例为 100%，返回打乱顺序的原始 DataFrame。")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    if proportion == 0.0:
        print("比例为 0%，返回一个空的 DataFrame。")
        return pd.DataFrame(columns=df.columns)

    # ----------------------------------------------------------------------
    # 核心抽样逻辑
    # ----------------------------------------------------------------------
    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        """
        应用于 groupby 的辅助函数。
        """
        # 1. 计算目标样本数 (浮点数)
        target_n = len(group) * proportion
        
        # 2. 四舍五入到最近的整数
        n_to_sample = int(round(target_n))
        
        # 3. 【!! 关键约束 !!】
        #    如果四舍五入为 0，但该组至少有1个样本，
        #    我们必须强制抽样 1 个，以满足“每种类型都要抽到”的要求。
        if n_to_sample == 0 and len(group) > 0:
            n_to_sample = 1
            
        # (安全检查：如果 n_to_sample > len(group)，group.sample 会自动处理，
        #  因为它默认 replace=False)
        # (但为了防止 np.ceil 导致 n_to_sample > len(group) 的罕见情况，
        #  我们明确一下，尽管 round() 已经使其不太可能)
        n_to_sample = min(n_to_sample, len(group))

        return group.sample(n=n_to_sample, random_state=random_state)
    # ----------------------------------------------------------------------

    # 1. 按标签列分组，并应用我们的自定义抽样函数
    print(f"按 '{label_column}' 列进行分组并应用抽样...")
    sampled_df = df.groupby(label_column, group_keys=False).apply(sample_group)
    
    # 2. (可选但推荐) 最后打乱一次
    #    因为 groupby().apply() 的输出是按组（标签）排序的
    print("抽样完成，正在打乱最终结果...")
    final_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"抽样成功：原始数据 {len(df)} 行 -> 抽样后 {len(final_df)} 行。")
    return final_df

def stratified_hybrid_sample_dataframe(
    df: pd.DataFrame, 
    label_column: str, 
    proportion: float, 
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    【改进版】对 DataFrame 进行按比例分层抽样，并为少数类设置一个“下限”。

    此函数实现了你的三条规则：
    1. 计算一个“平均抽样数” n_avg。
    2. 多数类 (n_A >= n_avg) -> 按比例 p 抽样。
    3. 少数类 (n_A < n_avg) -> 过采样到 n_avg。
    4. 极少数类 (N_A <= n_avg) -> 抽取所有 N_A。

    参数:
    ----------
    df : pd.DataFrame
        需要抽样的原始 DataFrame。
    label_column : str
        用于分层的列名 (例如 'label' 或 'label_id')。
    proportion : float
        要抽样的比例 (例如 0.1 代表 10%)。
    random_state : Optional[int], 默认=42
        用于确保抽样可复现的随机种子。

    返回:
    -------
    pd.DataFrame
        一个新的、按比例且“下限均衡”的抽样后 DataFrame。
    """
    
    print(f"开始混合分层抽样 (Hybrid Stratified Sampling)，目标比例: {proportion * 100:.1f}%...")

    if not (0.0 < proportion <= 1.0):
        raise ValueError("抽样比例 (proportion) 必须在 (0.0, 1.0] 之间。")
        
    if proportion == 1.0:
        print("比例为 100%，返回打乱顺序的原始 DataFrame。")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 核心抽样逻辑
    # ----------------------------------------------------------------------

    # 1. 【规则1】计算 n_avg
    N_total = len(df)
    N_classes = df[label_column].nunique()
    
    if N_classes == 0:
        print("警告：数据集中没有类别，返回空 DataFrame。")
        return pd.DataFrame(columns=df.columns)
        
    n_avg = int(round((proportion * N_total) / N_classes))
    print(f" -> 总样本: {N_total}, 类别数: {N_classes}")
    print(f" -> 平均抽样数 (n_avg) 已设为: {n_avg}")

    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        """
        应用于 groupby 的辅助函数。
        """
        N_A = len(group) # A类总数
        
        # 【规则4】若存在A的总数 N_A <= n_avg，则将所有A类样本添加
        if N_A <= n_avg:
            # 抽样 N_A (即全部)，无需替换
            n_sample = N_A
            replace = False
        
        # 否则 (N_A > n_avg)
        else:
            n_A_prop = int(round(N_A * proportion)) # 按比例应抽
            
            # 【规则3】若 n_A < n_avg，则抽取 n_avg 个
            if n_A_prop < n_avg:
                n_sample = n_avg
                # 【!! 关键 !!】如果 n_avg > N_A (理论上不会，
                # 因为被第一个if捕获了，但为了安全)，
                # 我们需要允许替换。但根据我们的逻辑，
                # 此时 N_A > n_avg，所以 n_sample <= N_A。
                replace = False 
            
            # 【规则2】若 n_A >= n_avg，则正常抽样 n_A 个
            else: # n_A_prop >= n_avg
                n_sample = n_A_prop
                replace = False

        return group.sample(n=n_sample, random_state=random_state, replace=replace)
    # ----------------------------------------------------------------------

    # 1. 按标签列分组，并应用我们的自定义抽样函数
    print(f"按 '{label_column}' 列进行分组并应用混合抽样...")
    sampled_df = df.groupby(label_column, group_keys=False).apply(sample_group)
    
    # 2. (可选但推荐) 最后打乱一次
    print("抽样完成，正在打乱最终结果...")
    final_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"抽样成功：原始数据 {len(df)} 行 -> 抽样后 {len(final_df)} 行。")
    return final_df

def stratified_aggressive_balancing(
    df: pd.DataFrame, 
    label_column: str, 
    proportion: float, 
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    【!! 激进均衡版 !!】
    对 DataFrame 进行分层抽样，并 *强制* 将 *所有* 类别
    （通过过采样和欠采样）统一拉到“平均抽样数” (n_avg)。

    这实现了你的提议：“将大样本和小样本的采样数量统一拉到抽样平均数”

    参数:
    ----------
    df : pd.DataFrame
        需要抽样的原始 DataFrame。
    label_column : str
        用于分层的列名 (例如 'label' 或 'label_id')。
    proportion : float
        要抽样的比例 (例如 0.1 代表 10%)。
    random_state : Optional[int], 默认=42
        用于确保抽样可复现的随机种子。

    返回:
    -------
    pd.DataFrame
        一个新的、*完全均衡*的抽样后 DataFrame。
    """
    
    print(f"开始 *激进均衡* 分层抽样，目标比例: {proportion * 100:.1f}%...")

    if not (0.0 < proportion <= 1.0):
        raise ValueError("抽样比例 (proportion) 必须在 (0.0, 1.0] 之间。")
        
    N_total = len(df)
    N_classes = df[label_column].nunique()
    
    if N_classes == 0:
        print("警告：数据集中没有类别，返回空 DataFrame。")
        return pd.DataFrame(columns=df.columns)
        
    # 【规则1】计算 n_avg
    n_avg = int(round((proportion * N_total) / N_classes))
    if n_avg == 0: n_avg = 1 # 至少为1
    
    print(f" -> 总样本: {N_total}, 类别数: {N_classes}")
    print(f" -> 平均抽样数 (n_avg) 已设为: {n_avg}")
    print(f" -> 所有类别都将被 *强制* 采样到 {n_avg} 个样本。")

    # ----------------------------------------------------------------------
    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        N_A = len(group) # A类总数
        
        # 【!! 核心逻辑 !!】
        # 目标样本数 *总是* n_avg
        n_sample = n_avg
        
        # 如果 n_avg > N_A (原始样本不足), 必须 *过采样*
        if n_sample > N_A:
            replace = True
        else:
        # 如果 n_avg <= N_A (原始样本充足), 进行 *欠采样*
            replace = False

        return group.sample(n=n_sample, random_state=random_state, replace=replace)
    # ----------------------------------------------------------------------

    print(f"按 '{label_column}' 列进行分组并应用 *激进均衡*...")
    sampled_df = df.groupby(label_column, group_keys=False).apply(sample_group)
    
    print("抽样完成，正在打乱最终结果...")
    final_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"抽样成功：原始数据 {len(df)} 行 -> 抽样后 {len(final_df)} 行 (约 {n_avg * N_classes})。")
    return final_df