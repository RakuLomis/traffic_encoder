import pyshark
import os
import pandas as pd 
from tqdm import tqdm 
from utils.pcap_tools import get_fields_over_layers 
from utils.pcap_tools import get_pcap_path 

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 

pcap_path_list, file_name_list = get_pcap_path(directory_path) 
if pcap_path_list is not None: 
    for pcap_path, file_name in zip(pcap_path_list, file_name_list): 
        pcap_file = pyshark.FileCapture(pcap_path) 
        # list_fields = get_fields_over_tcp(pcap_file) 
        # list_fields = get_fields_over_all(pcap_file) 
        list_fields = get_fields_over_layers(pcap_file)
        print(list_fields.__len__) 
        df_fields = pd.DataFrame(list_fields) 
        df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'))
        pcap_file.close() 


# for pcap_file in pcap_file_list: 
#     # print(pcap_file[0])     
#     list_fields = get_fields_over_tcp(pcap_file) 
#     print(list_fields.__len__) 
#     df_fields = pd.DataFrame(list_fields) 
#     df_fields.to_csv(os.path.join(directory_path, ))
#     pcap_file.close() 
#     break
 