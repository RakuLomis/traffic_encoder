import os 
from utils.pcap_tools import pcap_to_csv 
from tqdm import tqdm 
from utils.dataframe_tools import merge_csvs_with_different_columns 
from utils.dataframe_tools import label_and_merge_csvs


raw_csv_directory_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv') 
csv_directory_merged = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged')
csv_directory_merged_completed = os.path.join(csv_directory_merged, 'completeness')
completed_csv_path = os.path.join(csv_directory_merged_completed, 'dataset_29_completed_label.csv') 

# 将pcap生成的csv文件按照类别合并
label_and_merge_csvs(raw_csv_directory_path, csv_directory_merged, need_label=True) 
# 将所有csv文件合并成一个总的csv
merge_csvs_with_different_columns(csv_directory_merged, completed_csv_path, postfix='_label.csv') 

