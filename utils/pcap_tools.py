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
        pcap.close()

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
            list_fields, dict_reassemble = get_fields_and_reassembled_info(pcap_path)
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

# --------------pcap processing optimizations--------------
# def packet_count_v2(pcap):
#     # 一个快速获取包总数的方法
#     try:
#         return len(pcap)
#     except TypeError: # 如果文件很大，pyshark可能不会预先加载
#         print("Large file, counting packets...")
#         count = 0
#         for _ in pcap:
#             count += 1
#         pcap.close() # 必须关闭并重新打开
#         return count
    

# def get_fields_over_layers_v2(pcap_path: str, given_layers = ['eth', 'ip', 'tcp', 'tls']): 
#     """
#     Extract fields over specific layers of the input pcap file and return a dictionary, 
#     including tcp.stream, excluding payload. 

#     Parameters
#     ----------
#     pcap_path: str
#         Path of pcap file. 
#     given_layers: list 
#         A list of the specific layers you want to extract fields from, 
#         its default value is ['eth', 'ip', 'tcp', 'tls']. 

#     Returns 
#     ------- 
#     res_list: list, [{K: V}, ...] 
#         K is the name of fields in the specific layer, and V is 
#         its corresponding value. 
#     """ 
#     pcaps = pyshark.FileCapture(pcap_path, keep_packets=False)
#     res_list = []
#     long_field = [] 
#     # for i in [250, 251]: # Test 
#     for pcap in tqdm(pcaps, "get_fields_over_layers"): 
#         if pcap.transport_layer == 'TCP': # ignore the UDP based protocols
#             all_fields = {} 
#             special_fields = ['stream', 'len'] # fields use 'show' and not hex 
#             frame_num = int(pcap.frame_info.get_field('number')) 
#             all_fields['frame_num'] = frame_num # Add frame number
#             for layer in pcap.layers: 
#                 if layer.layer_name in given_layers: 
#                     list_field_ori = list(layer._all_fields.keys()) # eth.dst, eth.dst_resolved 
#                     list_field_no_layer_field = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.', False)
#                     list_field_replaced = delete_prefix_for_list_item(list_field_ori, layer.layer_name + '.') 
#                     for field in layer.field_names: 
#                         try: 
#                             field_index = list_field_replaced.index(field) 
#                         except ValueError: 
#                             print('No matched field. ') 
#                         field_dot_split = list_field_no_layer_field[field_index]
#                         field_obj = layer.get_field(field) 
#                         if getattr(field_obj.main_field, 'hide'): # skip the hidden attributes
#                             continue 
#                         # Do not use .show, although it may describe the briefer and more readable information 
#                         # .value will display the hexadcimal of ascii code 
#                         hex_value = field_obj.raw_value
#                         if hex_value is not None: 
#                             if field == '': 
#                                 continue
#                             if len(hex_value) >= 64: # skip the too long features, such as payload. 
#                                 long_field.append(field) 
#                             if field not in long_field: 
#                                 all_fields[layer.layer_name + '.' + field_dot_split] = hex_value 
#                                 # all_fields[layer.layer_name + '_' + field] = hex_value 
#                                 # pyshark uses '_' to split fields in protocol tree, 
#                                 # which is also used to represent a field name combined by several words. 
#                                 # So we need to distinguish these field names. 

#                         if layer.layer_name == 'tcp' and field in special_fields: 
#                             # the attribute 'show' does not display the hex value, instead, dec value 
#                             # stream: tcp stream id 
#                             # len: payload length 
#                             all_fields[layer.layer_name + '.' + field_dot_split] = field_obj.show 
#                             # all_fields[layer.layer_name + '_' + field] = field_obj.show 
#             res_list.append(all_fields) 
#     pcaps.close() 
#     return res_list 

# def get_reasemmble_info_v2(pcap_path: pyshark.FileCapture): 
#     """
#     Extract the reassemble information for each packet. 

#     Parameters 
#     ----------
#     pcap: pyshark.FileCapture

#     Returns 
#     ------- 
#     res_dict: dict, {K: [v1, ...], ...} 
#         K is the packet index in the same form of Wireshark, namely, starts from 1. 
#         [v1, ...] denotes the reassembled indices, whose values will be K in turn and have the same reassembled list. 
#         For example, {1: [1, 2], 2: [1, 2]}. 
#     """
#     pcaps = pyshark.FileCapture(pcap_path, keep_packets=False)
#     res_dict = {} # {index: [reassemble packets]}
#     for pcap in tqdm(pcaps, "get reassemble info"): 
#         if pcap.transport_layer == 'TCP': # ignore the UDP based protocols 
#             frame_num = int(pcap.frame_info.get_field('number')) # get the number of frame
#             res_dict[frame_num] = [] # init i-th position as empty 
#             segment_index = [] 
#             # print(f'${i}$: ${pcap[i].layers}')
#             for layer in pcap.layers: 
#                 if layer.layer_name == 'DATA': # fake-field-wrapper is renamed to data in pyshark
#                     for field in layer.field_names: 
#                         if field == 'tcp_segments': # reassemble will appearance in the last packet
#                             field_obj = layer.get_field(field) 
#                             content = field_obj.main_field.get_default_value() 
#                             segment_index.extend(match_segment_number(content)) 
#             for index in segment_index: # cover related values with its reassemble info
#                 res_dict[index] = segment_index 
#     pcaps.close() 
#     return res_dict


# def pcap_to_csv_v2(directory_path, output_directory_path): 
#     """
#     Transform the pcap into csv and add some extra features. 

#     Parameters 
#     ---------- 
#     directory_path: 
#         The path of pcap files' directory. 
#     output_directory_path: 
#         The path of output csvs' directory.
#     """
#     pcap_path_list, file_name_list = get_pcap_path(directory_path) 
#     # pcap_path_list, file_name_list = get_file_path(dir_path=directory_path, postfix=['pcap, pcapng'])
#     if pcap_path_list is not None: 
#         for pcap_path, file_name in tqdm(zip(pcap_path_list, file_name_list), desc=f"Handling pcaps from {directory_path}"): 
#             # pcap_file = pyshark.FileCapture(pcap_path) 
#             list_fields = get_fields_over_layers_v2(pcap_path) 
#             dict_reassemble = get_reasemmble_info_v2(pcap_path) 
#             df_reassemble = pd.DataFrame({
#                 "frame_num": list(dict_reassemble.keys()), 
#                 "tcp.reassembled_segments": list(dict_reassemble.values()) 
#             }) 
#             # df_reassemble.to_csv(os.path.join(directory_path, 'reassemble_' + file_name + '.csv'), index=False) 
#             df_fields = pd.DataFrame(list_fields) # frame_num
#             # df_fields['index'] = range(1, len(df_reassemble) + 1) 
#             df_merge_tls = pd.merge(df_fields, df_reassemble, on=["frame_num"], how="outer") 
#             print(f'original shape: ${df_merge_tls.shape}')
#             # df_fields.to_csv(os.path.join(directory_path, file_name + '.csv'), index=False) 
#             df_merge_tls = filter_out_nan(df_merge_tls) 
#             # all_nan_cols = df_merge_tls.columns[df_merge_tls.isna().all()] 
#             # df_merge_tls = df_merge_tls.drop(columns=all_nan_cols)
#             print(f'fill out NaN shape: ${df_merge_tls.shape}') 
#             df_merge_tls.to_csv(os.path.join(output_directory_path, 'merge_' + file_name + '.csv'), index=False)
#             # pcap_file.close() 

# def extract_all_data_single_pass(pcap_path: str, given_layers=['eth', 'ip', 'tcp', 'tls']):
#     """
#     在单次高效遍历中提取所有字段信息。
#     此版本移除了所有对tcp.reassembled_segments的额外处理，
#     完全忠实于pyshark提取的原始数据。

#     :param pcap_path: pcap文件的路径。
#     :param given_layers: 需要提取的协议层列表。
#     :return: 一个包含所有数据包信息的DataFrame。
#     """
#     pcap_capture = pyshark.FileCapture(pcap_path, keep_packets=False)
    
#     try:
#         all_packets_list = []
        
#         for packet in tqdm(pcap_capture, desc=f"Processing {os.path.basename(pcap_path)}"):
            
#             if not hasattr(packet, 'tcp'):
#                 continue
                
#             current_fields = {'frame_num': int(packet.frame_info.number)}
            
#             # --- 字段提取逻辑 (无变化) ---
#             for layer in packet.layers:
#                 if layer.layer_name not in given_layers:
#                     continue
                
#                 for field_name in layer.get_field_names():
#                     field_obj = getattr(layer, field_name)
#                     hex_value = field_obj.raw_value
                    
#                     if hex_value is not None and len(hex_value) < 64:
#                         full_field_name = f"{layer.layer_name}.{field_name}"
#                         current_fields[full_field_name] = hex_value
            
#             # --- 特殊字段处理 (无变化) ---
#             current_fields['tcp.stream'] = packet.tcp.stream
#             current_fields['tcp.len'] = packet.tcp.len
            
#             # ==================== 核心修改点 开始 ====================
#             # --- 简化并忠实于原始的重组信息记录 ---
#             # 我们只记录 pyshark 直接提供的信息，不做任何额外的推断和处理。
            
#             reassembled_in_frame = -1  # 默认值为 -1，代表它不是一个已知重组段
#             if hasattr(packet.tcp, 'reassembled_in'):
#                 # 如果pyshark告诉我们这个包被重组了，我们就记录它被重组到的那个包的帧号。
#                 reassembled_in_frame = int(packet.tcp.reassembled_in)
            
#             current_fields['tcp.reassembled_segments'] = reassembled_in_frame
#             # ==================== 核心修改点 结束 ====================

#             all_packets_list.append(current_fields)

#         if not all_packets_list:
#             print(f"Warning: No processable TCP packets found in {pcap_path}")
#             return pd.DataFrame()
            
#         # 直接从收集到的列表中创建DataFrame，不做任何关于重组的后处理
#         df = pd.DataFrame(all_packets_list)
        
#         return df

#     finally:
#         pcap_capture.close()



# def pcap_to_csv_v3(directory_path, output_directory_path): 
#     """
#     Transforms pcap files into CSVs using the optimized single-pass function.
#     """
#     pcap_path_list, file_name_list = get_pcap_path(directory_path)
    
#     if pcap_path_list is not None: 
#         for pcap_path, file_name in zip(pcap_path_list, file_name_list):
            
#             # === SINGLE, EFFICIENT CALL ===
#             df_merged = extract_all_data_single_pass(pcap_path)
            
#             if not df_merged.empty:
#                 print(f'\nOriginal shape for {file_name}: {df_merged.shape}')
                
#                 df_cleaned = filter_out_nan(df_merged) # Your existing cleaning function
                
#                 print(f'Final shape after cleaning: {df_cleaned.shape}')
#                 df_cleaned.to_csv(os.path.join(output_directory_path, 'merge_' + file_name + '.csv'), index=False)