import pandas as pd
import os 
import numpy as np 
from typing import Optional 
from tqdm import tqdm 

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
    #         for root, _, files in tqdm(os.walk(dir_path), "get_file_path: "): 
    #             for file_name in files: 
    #             # Check prefix condition
    #                 prefix_ok = (prefix is None) or file_name.startswith(prefix)
    #                 # Check postfix condition
    #                 postfix_ok = (postfix is None) or file_name.endswith(postfix)

    #                 if prefix_ok and postfix_ok:
    #                     # Build full path
    #                     full_path = os.path.join(root, file_name)
    #                     target_paths.append(full_path)

    #                 # Process base name 
    #                 if postfix and file_name.endswith(postfix):
    #                     base_name = file_name[:-len(postfix)] 
    #                 if postfix and not file_name.endswith(postfix): 
    #                     continue 
    #                 else: # no postfix, record file name
    #                     base_name = file_name
    #                 file_names.append(base_name)
    # else: 
    #     print("Invalid directory path") 
    # return target_paths, file_names 
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
    os.makedirs(fold_path, exist_ok=True) 
    file_path = os.path.join(fold_path, csv_name) 
    df.to_csv(file_path, index=index) 
    
def padding_features(df: pd.DataFrame, padding_value='ffff'): 
    return df.fillna(padding_value) 

def truncating_features(df: pd.DataFrame): 
    mask_df = df.notnull() 
    
    pass 

def padding_or_truncating(df: pd.DataFrame, pon: bool): 
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
    pass 