import pandas as pd
import os 
from utils.dataframe_tools import filter_out_nan 
from utils.dataframe_tools import to_integer_code
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 

csv_path_list, csv_name_list = get_file_path(directory_path, postfix='.csv') 
if csv_path_list is not None: 
    for csv_path, csv_name in zip(csv_path_list, csv_name_list): 
        df = pd.read_csv(csv_path) 
        filter_out_nan(df) 
        to_integer_code(df) 
        df_fill_nan = df.fillna('ffff') 
        output_csv_in_fold(df_fill_nan, os.path.join(directory_path, 'fill_nan'), 'fill_nan_' + csv_name + '.csv') 
