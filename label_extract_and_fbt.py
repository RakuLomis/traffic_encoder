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

input_directory_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv') 
output_merged_directory_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged')

label_and_merge_csvs(input_directory_path, output_merged_directory_path) 
