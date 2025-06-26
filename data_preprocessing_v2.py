import pandas as pd
import os 
from utils.dataframe_tools import filter_out_nan 
from utils.dataframe_tools import to_integer_code
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating
import argparse 

CONTINUOUS_BLOCK = 'continuous' 
DISCRETE_BLOCK = 'discrete'

# current_path = os.path.dirname(os.path.abspath(__file__))
# directory_path = os.path.join(current_path, 'Data', 'Test') 

def main(): 
    parser = argparse.ArgumentParser(description="Truncate the CSV into different Field Blocks. ") 
    parser.add_argument(
        '-d', '--directory', 
        type=str, 
        required=True, 
        help="The path to the directory containing original csv files. "
    )
    parser.add_argument(
        '-m', '--mode', 
        type=str, 
        default=DISCRETE_BLOCK, 
        choices=[DISCRETE_BLOCK, CONTINUOUS_BLOCK], 
        help="The type of the field block to generate. Defaults to 'discrete'."
    )
    args = parser.parse_args()
    directory_path = args.directory
    block_type = args.mode 
    print(f"Running in '{block_type}' mode.") 
    csv_path_list, csv_name_list = get_file_path(directory_path, postfix='.csv') 
    if csv_path_list is not None: 
        for csv_path, csv_name in zip(csv_path_list, csv_name_list): 
            df = pd.read_csv(csv_path) 

            list_df_block = padding_or_truncating(df, False, block_type) 
            for block_num in range(len(list_df_block)): 
                output_csv_in_fold(list_df_block[block_num], os.path.join(directory_path, csv_name, block_type), f'{block_num}' + '.csv') 
            # df_fill_nan = df.fillna('ffff') 
            # output_csv_in_fold(df_fill_nan, os.path.join(directory_path, 'fill_nan'), 'fill_nan_' + csv_name + '.csv') 
if __name__ == '__main__': 
    main()
