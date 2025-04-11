import pyshark
import os
import pandas as pd 
from tqdm import tqdm 
from utils.pcap_tools import get_fields_over_layers 
from utils.pcap_tools import get_pcap_path 
from utils.pcap_tools import get_reasemmble_info 
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import filter_out_nan

""" 
Get fields hexvalues into two dataframes (fields_values, reassemble information) and merge them into one. 
Pay attention to the index of packet in python and wireshark. 
The index in pyshark starts with 0, and in wireshark (including exported .xml) starts from 1. 
"""

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 

pcap_path_list, file_name_list = get_pcap_path(directory_path) 
# pcap_path_list, file_name_list = get_file_path(dir_path=directory_path, postfix=['pcap, pcapng'])
if pcap_path_list is not None: 
    for pcap_path, file_name in zip(pcap_path_list, file_name_list): 
        pcap_file = pyshark.FileCapture(pcap_path) 
        list_fields = get_fields_over_layers(pcap_file) 
        dict_reassemble = get_reasemmble_info(pcap_file) 
        df_reassemble = pd.DataFrame({
            "frame_num": list(dict_reassemble.keys()), 
            "reassembled_segments": list(dict_reassemble.values()) 
        }) 
        # df_reassemble.to_csv(os.path.join(directory_path, 'reassemble_' + file_name + '.csv'), index=False) 
        df_fields = pd.DataFrame(list_fields) # frame_num
        # df_fields['index'] = range(1, len(df_reassemble) + 1) 
        df_merge_tls = pd.merge(df_fields, df_reassemble, on=["frame_num"], how="outer") 
        print(f'original shape: ${df_merge_tls.shape}')
        # df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'), index=False) 
        # df_merge_tls = filter_out_nan(df_merge_tls) 
        all_nan_cols = df_merge_tls.columns[df_merge_tls.isna().all()] 
        df_merge_tls = df_merge_tls.drop(columns=all_nan_cols)
        print(f'fill out NaN shape: ${df_merge_tls.shape}')
        df_merge_tls.to_csv(os.path.join(directory_path, 'merge_' + file_name + '.csv'), index=False)
        pcap_file.close() 
