import pandas as pd
import os 
import numpy as np 
from typing import Optional 
from tqdm import tqdm 
from typing import Literal 

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
    all_nan_cols = df.columns[df.isna().all()] 
    return df.drop(columns=all_nan_cols) 


def to_integer_code(df: pd.DataFrame, col_name = 'reassembled_segments'): 
    """
    Turn the specific column into the integer code for convenience. 

    Returns
    -------
    value_to_code: dict
        {'[]': -1, '[1, 2]': 0, '[3, 4, 5]': 1, ...}
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
    dict_true = {} 
    for col_num in range(mask_df.shape[1]): 
        list_true_indices = list(mask_df[mask_df.iloc[:, col_num]].index) 
        dict_true[col_num] = list_true_indices 
    return dict_true 

def truncate_to_block(dict_true: dict, block_type: Literal['continuous', 'discrete'] = 'continuous'): 
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
    if block_type not in ['continuous', 'discrete']: 
        raise ValueError("block_type must be either 'continuous' or 'discrete'") 
    mask_df = df.notnull() 
    dict_true = get_not_nan_pos(mask_df) 
    dict_block = truncate_to_block(dict_true, block_type) 
    return dict_block 

def block_to_dataframe(dict_block: dict, df_ori: pd.DataFrame, output_path: str): 
    block_values = dict_block['block'] 
    columns_values = dict_block['columns'] 
    rows_values = dict_block['rows'] 
    for block_name, columns, rows in tqdm(zip(block_values, columns_values, rows_values)): 
        subset_rows = df_ori.loc[rows] 
        sub_df = subset_rows.iloc[:, columns] 
        output_csv_in_fold(sub_df, output_path, 'block_' + block_name + '.csv')

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

def padding_or_truncating(df: pd.DataFrame, pon: bool, continuous_truncating=True): 
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
    """
    if pon: 
        return padding_features(df) 
    else: 
        pass
    pass 