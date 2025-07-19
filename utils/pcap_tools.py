import pyshark 
from tqdm import tqdm 
import os 
import re 
import pandas as pd 
from .dataframe_tools import filter_out_nan 
import json 

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
            # for root, _, file_paths in os.walk(dir_path): 
            #     for file_path in file_paths: 
            #         if file_path.endswith(('.pcap', '.pcapng')): 
            #             pcap_paths.append(os.path.join(root, file_path)) 
            #             file_names.append(file_path[:-len('.pcap')] if file_path.endswith('.pcap') else file_path[:-len('.pcapng')]) 
            with os.scandir(dir_path) as entries: 
                for entry in tqdm(entries, "get_pcap_path: "): 
                    if entry.is_file(): 
                        file_name = entry.name 
                        if file_name.endswith(('.pcap', '.pcapng')): 
                            pcap_paths.append(entry.path) 
                            file_names.append(file_name[:-len('.pcap')] if file_name.endswith('.pcap') else file_name[:-len('.pcapng')]) 
    else: 
        print("Invalid directory path") 
    return pcap_paths, file_names 

def delete_prefix_for_list_item(l: list, prefix: str, replace_dot=True): 
    """
    Delete the prefix for the specific list elements and exchange the '.' and '-' with '_'. 

    Parameters 
    ---------- 
    l: list 
        The list contains target strings. 
    prefix: str
        The prefix you want to delete. 
    replace_dot: bool, default True 
        Replace the '.' and '-' in strings or not. 

    Returns 
    ------- 
    res: list 
        The handled list. 
    """
    res = []
    for item in l: 
        if item.startswith(prefix): 
            item = item.split(prefix, 1)[1] # extract the content after prefix 
        if replace_dot: 
            item = item.replace('.', '_').replace('-', '_').lower() 
        res.append(item) 
    return res

def get_fields_over_layers(pcap: pyshark.FileCapture, given_layers = ['eth', 'ip', 'tcp', 'tls']): 
    """
    Extract fields over specific layers of the input pcap file and return a dictionary, 
    including tcp.stream, excluding payload. 

    Parameters
    ----------
    pcap: pyshark.FileCapture
        The pcap file read by pyshark. 
    given_layers: list 
        A list of the specific layers you want to extract fields from, 
        its default value is ['eth', 'ip', 'tcp', 'tls']. 

    Returns 
    ------- 
    res_list: list, [{K: V}, ...] 
        K is the name of fields in the specific layer, and V is 
        its corresponding value. 
    """
    res_list = []
    long_field = [] 
    # for i in [250, 251]: # Test 
    for i in tqdm(range(packet_count(pcap)), "get_fields_over_layers"): 
        if pcap[i].transport_layer == 'TCP': # ignore the UDP based protocols
            all_fields = {} 
            special_fields = ['stream', 'len'] # fields use 'show' and not hex 
            frame_num = int(pcap[i].frame_info.get_field('number')) 
            all_fields['frame_num'] = frame_num # Add frame number
            for layer in pcap[i].layers: 
                if layer.layer_name in given_layers: 
                    list_field_ori = list(layer._all_fields.keys()) # eth.dst, eth.dst_resolved 
                    list_field_no_layer_field = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.', False)
                    list_field_replaced = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.') 
                    for field in layer.field_names: 
                        try: 
                            field_index = list_field_replaced.index(field) 
                        except ValueError: 
                            print('No matched field. ') 
                        field_dot_split = list_field_no_layer_field[field_index]
                        field_obj = layer.get_field(field) 
                        if getattr(field_obj.main_field, 'hide'): # skip the hidden attributes
                            continue 
                        # Do not use .show, although it may describe the briefer and more readable information 
                        # .value will display the hexadcimal of ascii code 
                        hex_value = field_obj.raw_value
                        if hex_value is not None: 
                            if field == '': 
                                continue
                            if len(hex_value) >= 64: # skip the too long features, such as payload. 
                                long_field.append(field) 
                            if field not in long_field: 
                                all_fields[layer.layer_name + '.' + field_dot_split] = hex_value 
                                # all_fields[layer.layer_name + '_' + field] = hex_value 
                                # pyshark uses '_' to split fields in protocol tree, 
                                # which is also used to represent a field name combined by several words. 
                                # So we need to distinguish these field names. 

                        if layer.layer_name == 'tcp' and field in special_fields: 
                            # the attribute 'show' does not display the hex value, instead, dec value 
                            # stream: tcp stream id 
                            # len: payload length 
                            all_fields[layer.layer_name + '.' + field_dot_split] = field_obj.show 
                            # all_fields[layer.layer_name + '_' + field] = field_obj.show 
            res_list.append(all_fields) 
    pcap.close() 
    return res_list 

def match_segment_number(s: str): 
    """
    Extract numbers after symbol '#'.  
    """
    pattern = r'#(\d+)'
    numbers = re.findall(pattern, s)
    res = [int(num) for num in numbers]
    return res

def get_reasemmble_info(pcap: pyshark.FileCapture): 
    """
    Extract the reassemble information for each packet. 

    Parameters 
    ----------
    pcap: pyshark.FileCapture

    Returns 
    ------- 
    res_dict: dict, {K: [v1, ...], ...} 
        K is the packet index in the same form of Wireshark, namely, starts from 1. 
        [v1, ...] denotes the reassembled indices, whose values will be K in turn and have the same reassembled list. 
        For example, {1: [1, 2], 2: [1, 2]}. 
    """
    res_dict = {} # {index: [reassemble packets]}
    for i in tqdm(range(packet_count(pcap)), "get reassemble info"): 
        if pcap[i].transport_layer == 'TCP': # ignore the UDP based protocols 
            frame_num = int(pcap[i].frame_info.get_field('number')) # get the number of frame
            res_dict[frame_num] = [] # init i-th position as empty 
            segment_index = [] 
            # print(f'${i}$: ${pcap[i].layers}')
            for layer in pcap[i].layers: 
                if layer.layer_name == 'DATA': # fake-field-wrapper is renamed to data in pyshark
                    for field in layer.field_names: 
                        if field == 'tcp_segments': # reassemble will appearance in the last packet
                            field_obj = layer.get_field(field) 
                            content = field_obj.main_field.get_default_value() 
                            segment_index.extend(match_segment_number(content)) 
            for index in segment_index: # cover related values with its reassemble info
                res_dict[index] = segment_index 
    pcap.close() 
    return res_dict


def pcap_to_csv(directory_path, output_directory_path): 
    """
    Transform the pcap into csv and add some extra features. 

    Parameters 
    ---------- 
    directory_path: 
        The path of pcap files' directory. 
    output_directory_path: 
        The path of output csvs' directory.
    """
    pcap_path_list, file_name_list = get_pcap_path(directory_path) 
    # pcap_path_list, file_name_list = get_file_path(dir_path=directory_path, postfix=['pcap, pcapng'])
    if pcap_path_list is not None: 
        for pcap_path, file_name in tqdm(zip(pcap_path_list, file_name_list), desc=f"Handling pcaps from {directory_path}"): 
            pcap_file = pyshark.FileCapture(pcap_path) 
            list_fields = get_fields_over_layers(pcap_file) 
            dict_reassemble = get_reasemmble_info(pcap_file) 
            df_reassemble = pd.DataFrame({
                "frame_num": list(dict_reassemble.keys()), 
                "tcp.reassembled_segments": list(dict_reassemble.values()) 
            }) 
            # df_reassemble.to_csv(os.path.join(directory_path, 'reassemble_' + file_name + '.csv'), index=False) 
            df_fields = pd.DataFrame(list_fields) # frame_num
            # df_fields['index'] = range(1, len(df_reassemble) + 1) 
            df_merge_tls = pd.merge(df_fields, df_reassemble, on=["frame_num"], how="outer") 
            print(f'original shape: ${df_merge_tls.shape}')
            # df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'), index=False) 
            df_merge_tls = filter_out_nan(df_merge_tls) 
            # all_nan_cols = df_merge_tls.columns[df_merge_tls.isna().all()] 
            # df_merge_tls = df_merge_tls.drop(columns=all_nan_cols)
            print(f'fill out NaN shape: ${df_merge_tls.shape}') 
            df_merge_tls.to_csv(os.path.join(output_directory_path, 'merge_' + file_name + '.csv'), index=False)
            pcap_file.close() 



def get_fields_and_reassembled_info(pcap_path: str, given_layers = ['eth', 'ip', 'tcp', 'tls']): 
    """
    Extract fields over specific layers of the input pcap file and return a dictionary, 
    including tcp.stream, excluding payload. 

    Parameters
    ----------
    pcap_path: str
        The path of target pcap file. 
    given_layers: list 
        A list of the specific layers you want to extract fields from, 
        its default value is ['eth', 'ip', 'tcp', 'tls']. /

    Returns 
    ------- 
    res_list: list, [{K: V}, ...] 
        K is the name of fields in the specific layer, and V is 
        its corresponding value. 
    """
    res_list = []
    long_field = [] 
    reassembled_info = {} 
    pcaps = pyshark.FileCapture(pcap_path, keep_packets=False)
    # for i in [250, 251]: # Test 
    try: 
        for pcap in tqdm(pcaps, "get_fields_and_reassembled_info"): 
            if pcap.transport_layer == 'TCP': # ignore the UDP based protocols
                all_fields = {} 
                special_fields = ['stream', 'len'] # fields use 'show' and not hex 
                frame_num = int(pcap.frame_info.get_field('number')) 
                all_fields['frame_num'] = frame_num # Add frame number 
                # Get reassembled info
                reassembled_info[frame_num] = [] 
                segment_index = [] 
                for layer in pcap.layers: 
                    if layer.layer_name in given_layers: 
                        list_field_ori = list(layer._all_fields.keys()) # eth.dst, eth.dst_resolved 
                        list_field_no_layer_field = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.', False)
                        list_field_replaced = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.') 
                        for field in layer.field_names: 
                            try: 
                                field_index = list_field_replaced.index(field) 
                            except ValueError: 
                                print('No matched field. ') 
                            field_dot_split = list_field_no_layer_field[field_index]
                            field_obj = layer.get_field(field) 
                            if getattr(field_obj.main_field, 'hide'): # skip the hidden attributes
                                continue 
                            # Do not use .show, although it may describe the briefer and more readable information 
                            # .value will display the hexadcimal of ascii code 
                            hex_value = field_obj.raw_value
                            if hex_value is not None: 
                                if field == '': 
                                    continue
                                if len(hex_value) >= 64: # skip the too long features, such as payload. 
                                    long_field.append(field) 
                                if field not in long_field: 
                                    all_fields[layer.layer_name + '.' + field_dot_split] = hex_value 
                                    # all_fields[layer.layer_name + '_' + field] = hex_value 
                                    # pyshark uses '_' to split fields in protocol tree, 
                                    # which is also used to represent a field name combined by several words. 
                                    # So we need to distinguish these field names. 

                            if layer.layer_name == 'tcp' and field in special_fields: 
                                # the attribute 'show' does not display the hex value, instead, dec value 
                                # stream: tcp stream id 
                                # len: payload length 
                                all_fields[layer.layer_name + '.' + field_dot_split] = field_obj.show 
                                # all_fields[layer.layer_name + '_' + field] = field_obj.show 
                    # Get reassembled info
                    if layer.layer_name == 'DATA': 
                        for field in layer.field_names: 
                            if field == 'tcp_segments': # reassemble will appearance in the last packet
                                field_obj = layer.get_field(field) 
                                content = field_obj.main_field.get_default_value() 
                                segment_index.extend(match_segment_number(content)) 
                for index in segment_index: # cover related values with its reassemble info
                    reassembled_info[index] = segment_index 

                res_list.append(all_fields)         
        # pcap.close() 
        return res_list, reassembled_info 
    finally: 
        pcaps.close()

def pcap_to_csv_v2(directory_path, output_directory_path): 
    """
    Transform the pcap into csv and add some extra features. 

    Parameters 
    ---------- 
    directory_path: 
        The path of pcap files' directory. 
    output_directory_path: 
        The path of output csvs' directory.
    """
    pcap_path_list, file_name_list = get_pcap_path(directory_path) 
    # pcap_path_list, file_name_list = get_file_path(dir_path=directory_path, postfix=['pcap, pcapng'])
    if pcap_path_list is not None: 
        for pcap_path, file_name in tqdm(zip(pcap_path_list, file_name_list), desc=f"Handling pcaps from {directory_path}"): 
            # pcap_file = pyshark.FileCapture(pcap_path) 
            print(f" File: {file_name}")
            list_fields, dict_reassemble = get_fields_and_reassembled_info(pcap_path) 
            if not list_fields and not dict_reassemble: 
                print(f"No TCP packets in {file_name}. ")
            else: 
                df_reassemble = pd.DataFrame({
                    "frame_num": list(dict_reassemble.keys()), 
                    "tcp.reassembled_segments": list(dict_reassemble.values()) 
                }) 
                # df_reassemble.to_csv(os.path.join(directory_path, 'reassemble_' + file_name + '.csv'), index=False) 
                df_fields = pd.DataFrame(list_fields) # frame_num
                # df_fields['index'] = range(1, len(df_reassemble) + 1) 
                df_merge_tls = pd.merge(df_fields, df_reassemble, on=["frame_num"], how="outer") 
                print(f'original shape: ${df_merge_tls.shape}')
                # df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'), index=False) 
                df_merge_tls = filter_out_nan(df_merge_tls) 
                # all_nan_cols = df_merge_tls.columns[df_merge_tls.isna().all()] 
                # df_merge_tls = df_merge_tls.drop(columns=all_nan_cols)
                print(f'fill out NaN shape: ${df_merge_tls.shape}') 
                df_merge_tls.to_csv(os.path.join(output_directory_path, 'merge_' + file_name + '.csv'), index=False)
                # pcap_file.close() 