import pyshark
import os
import pandas as pd 
from tqdm import tqdm 

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(current_path, 'Data', 'Test') 
# print(os.curdir) 
# print(os.path.exists(directory_path)) 

def get_pcap_path(dir_path: str): 
    """
    Get all 'pcap' and 'pcapng' paths from the specific directory (folder). 
    """
    pcap_paths = [] 
    file_names = []
    if os.path.exists(dir_path): 
        if os.path.isdir(dir_path): 
            for root, _, file_paths in os.walk(dir_path): 
                for file_path in file_paths: 
                    if file_path.endswith(('pcap', 'pcapng')): 
                        pcap_paths.append(os.path.join(root, file_path)) 
                        file_names.append(file_path)
    else: 
        print("Invalid directory path") 
    return pcap_paths, file_names 

def packet_count(file, display_filter=None):
    """
    Count the number of packets in the given .pcap file, possible display filter may be applied.
    """
    # cap = pyshark.FileCapture(input_file=file, display_filter=display_filter, only_summaries=True, keep_packets=False)
    cap = file 
    cnt = 0
    for _ in cap:
        cnt += 1
    cap.close()
    return cnt

def get_fields_over_tcp(pcap: pyshark.FileCapture): 
    """
    Extract fields over TCP (including TCP and TLS, excluding other proxy protocols' layers) 
    of the input pcap file and return a dictionary. 

    Parameters
    ----------
    pcap: pyshark.FileCapture
        The pcap file read by pyshark. 

    Returns 
    ------- 
    res_list: [{K: V}, ...] 
        in which K is the name of fields in the specific layer, and V is 
        its corresponding value. 
    """
    payload_field = ['payload', 'segment_data', 'tcp_reassembled_data', 'ech_enc', 'ech_payload'] 
    res_list = []
    # for i in [250, 251]: # Test 
    for i in tqdm(range(packet_count(pcap))): 
        tcp_and_above_fields = {} 
        tcp_found, tls_found = False, False 
        for layer in pcap[i].layers: 
            if layer.layer_name == 'tcp': 
                tcp_found = True 
            if layer.layer_name == 'tls': 
                tls_found = True 
            # Only handle layers over TCP
            if tcp_found or tls_found: 
                # 遍历当前层的所有字段
                for field in layer.field_names: 
                    field_obj = layer.get_field(field) 
                    if field not in payload_field: 
                        # Do not use .show, although it may describe the briefer and more readable information 
                        # .value will display the hexadcimal of ascii code 
                        hex_value = field.raw_value
                        tcp_and_above_fields[field] = field_obj.raw_value
        res_list.append(tcp_and_above_fields) 
    return res_list 


def get_fields_over_all(pcap: pyshark.FileCapture): 
    """
    Extract all layers fields' values. 
    """
    payload_field = ['payload', 'segment_data', 'tcp_reassembled_data', 'ech_enc', 'ech_payload'] 
    layers_all = ['eth', 'ip', 'tcp', 'tls']
    res_list = []
    # for i in [250, 251]: # Test 
    for i in tqdm(range(packet_count(pcap))): 
        all_fields = {} 
        for layer in pcap[i].layers: 
            if layer.layer_name in layers_all: 
                for field in layer.field_names: 
                    # print(f'${layer}: ${layer.filed_names}') 
                    field_obj = layer.get_field(field) 
                    if field not in payload_field:  
                        # Do not use .show, although it may describe the briefer and more readable information 
                        # .value will display the hexadcimal of ascii code 
                        hex_value = field_obj.raw_value
                        # value_size = field_obj.size 
                        if hex_value is not None: 
                            all_fields[field] = hex_value 
        res_list.append(all_fields) 
    return res_list 


# pcap_file_list = [] 
pcap_path_list, file_name_list = get_pcap_path(directory_path) 
if pcap_path_list is not None: 
    for pcap_path, file_name in zip(pcap_path_list, file_name_list): 
        pcap_file = pyshark.FileCapture(pcap_path) 
        # list_fields = get_fields_over_tcp(pcap_file) 
        list_fields = get_fields_over_all(pcap_file) 
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
 