import pandas as pd
import os 
from utils.dataframe_tools import filter_out_nan 
from utils.dataframe_tools import to_integer_code
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating
from utils.dataframe_tools import label_and_merge_csvs

CONTINUOUS_BLOCK = 'continuous' 
DISCRETE_BLOCK = 'discrete'

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 

csv_path_list, csv_name_list = get_file_path(directory_path, postfix='.csv') 
if csv_path_list is not None: 
    for csv_path, csv_name in zip(csv_path_list, csv_name_list): 
        df = pd.read_csv(csv_path) 
        
        list_df_block = padding_or_truncating(df, False, DISCRETE_BLOCK) 
        for block_num in range(len(list_df_block)): 
            output_csv_in_fold(list_df_block[block_num], os.path.join(directory_path, csv_name, DISCRETE_BLOCK), f'{block_num}' + '.csv') 
        # df_fill_nan = df.fillna('ffff') 
        # output_csv_in_fold(df_fill_nan, os.path.join(directory_path, 'fill_nan'), 'fill_nan_' + csv_name + '.csv') 
