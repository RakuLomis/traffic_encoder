import pyshark 
from tqdm import tqdm 
import os 

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

def get_fields_over_layers(pcap: pyshark.FileCapture, given_layers = ['eth', 'ip', 'tcp', 'tls']): 
    """
    Extract fields over specific layers of the input pcap file and return a dictionary. 

    Parameters
    ----------
    pcap: pyshark.FileCapture
        The pcap file read by pyshark. 
    given_layers: list 
        A list of the specific layers you want to extract fields from, 
        its default value is ['eth', 'ip', 'tcp', 'tls']. 

    Returns 
    ------- 
    res_list: [{K: V}, ...] 
        in which K is the name of fields in the specific layer, and V is 
        its corresponding value. 
    """
    payload_field = ['payload', 'segment_data', 'tcp_reassembled_data', 'ech_enc', 'ech_payload'] 
    # layers_all = ['eth', 'ip', 'tcp', 'tls']
    res_list = []
    # for i in [250, 251]: # Test 
    for i in tqdm(range(packet_count(pcap))): 
        all_fields = {} 
        for layer in pcap[i].layers: 
            if layer.layer_name in given_layers: 
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
