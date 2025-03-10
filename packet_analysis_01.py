import pyshark
import os
import pandas as pd 
from tqdm import tqdm 
from utils.pcap_tools import get_fields_over_layers 
from utils.pcap_tools import get_pcap_path 
from utils.pcap_tools import get_reasemmble_info 

"""
Pay attention to the index of packet in python and wireshark. 
The index in pyshark starts with 0, and in wireshark (including exported .xml) starts from 1. 
"""

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 

pcap_path_list, file_name_list = get_pcap_path(directory_path) 
if pcap_path_list is not None: 
    for pcap_path, file_name in zip(pcap_path_list, file_name_list): 
        pcap_file = pyshark.FileCapture(pcap_path) 
        list_fields = get_fields_over_layers(pcap_file) 
        dict_reassemble = get_reasemmble_info(pcap_file) 
        df_reassemble = pd.DataFrame({
            "index": list(dict_reassemble.keys()), 
            "reassembled_segments": list(dict_reassemble.values()) 
        }) 
        df_reassemble.to_csv(os.path.join(directory_path, 'reassemble_' + file_name + '.csv'), index=False) 
        # value_list_reassemble = list(dict_reassemble.values()) 
        # print(value_list_reassemble) 
        # df_reassemble = pd.DataFrame(value_list_reassemble) 
        df_fields = pd.DataFrame(list_fields) 
        df_fields['index'] = range(1, len(df_reassemble) + 1) 
        df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'), index=False)
        pcap_file.close() 
