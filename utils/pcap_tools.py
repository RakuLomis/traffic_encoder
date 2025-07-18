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


# --------------pcap processing optimizations--------------
def packet_count_v2(pcap):
    # 一个快速获取包总数的方法
    try:
        return len(pcap)
    except TypeError: # 如果文件很大，pyshark可能不会预先加载
        print("Large file, counting packets...")
        count = 0
        for _ in pcap:
            count += 1
        pcap.close() # 必须关闭并重新打开
        return count
    

def get_fields_over_layers_v2(pcap_path: str, given_layers = ['eth', 'ip', 'tcp', 'tls']): 
    """
    Extract fields over specific layers of the input pcap file and return a dictionary, 
    including tcp.stream, excluding payload. 

    Parameters
    ----------
    pcap_path: str
        Path of pcap file. 
    given_layers: list 
        A list of the specific layers you want to extract fields from, 
        its default value is ['eth', 'ip', 'tcp', 'tls']. 

    Returns 
    ------- 
    res_list: list, [{K: V}, ...] 
        K is the name of fields in the specific layer, and V is 
        its corresponding value. 
    """ 
    pcaps = pyshark.FileCapture(pcap_path, keep_packets=False)
    res_list = []
    long_field = [] 
    # for i in [250, 251]: # Test 
    for pcap in tqdm(pcaps, "get_fields_over_layers"): 
        if pcap.transport_layer == 'TCP': # ignore the UDP based protocols
            all_fields = {} 
            special_fields = ['stream', 'len'] # fields use 'show' and not hex 
            frame_num = int(pcap.frame_info.get_field('number')) 
            all_fields['frame_num'] = frame_num # Add frame number
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
            res_list.append(all_fields) 
    pcaps.close() 
    return res_list 

def get_reasemmble_info_v2(pcap_path: pyshark.FileCapture): 
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
    pcaps = pyshark.FileCapture(pcap_path, keep_packets=False)
    res_dict = {} # {index: [reassemble packets]}
    for pcap in tqdm(pcaps, "get reassemble info"): 
        if pcap.transport_layer == 'TCP': # ignore the UDP based protocols 
            frame_num = int(pcap.frame_info.get_field('number')) # get the number of frame
            res_dict[frame_num] = [] # init i-th position as empty 
            segment_index = [] 
            # print(f'${i}$: ${pcap[i].layers}')
            for layer in pcap.layers: 
                if layer.layer_name == 'DATA': # fake-field-wrapper is renamed to data in pyshark
                    for field in layer.field_names: 
                        if field == 'tcp_segments': # reassemble will appearance in the last packet
                            field_obj = layer.get_field(field) 
                            content = field_obj.main_field.get_default_value() 
                            segment_index.extend(match_segment_number(content)) 
            for index in segment_index: # cover related values with its reassemble info
                res_dict[index] = segment_index 
    pcaps.close() 
    return res_dict

# def get_fields_over_layers_v2(pcap_path: str, given_layers=['eth', 'ip', 'tcp', 'tls']):
#     """
#     使用高效的pyshark迭代模式，在单次遍历中提取所有字段和重组信息。
#     这个函数旨在替换 get_fields_over_layers 和 get_reasemmble_info。

#     :param pcap_path: pcap文件的路径。
#     :param given_layers: 需要提取的协议层列表。
#     :return: 一个包含所有数据包信息的DataFrame。
#     """
    
#     # 推荐使用 keep_packets=False 处理大文件，这可以防止pyshark将所有包缓存在内存中
#     pcap_file = pyshark.FileCapture(pcap_path, keep_packets=False)
    
#     all_packets_list = []
    
#     # 使用高效的 for...in 循环进行单次遍历
#     for packet in tqdm(pcap_file, desc=f"Processing {os.path.basename(pcap_path)}"):
        
#         # 我们只关心TCP包
#         if not hasattr(packet, 'tcp'):
#             continue
            
#         current_fields = {}
        
#         # 1. 添加 frame_num
#         current_fields['frame_num'] = int(packet.frame_info.number)
        
#         # 2. 遍历packet中的所有层，提取字段
#         for layer in packet.layers:
#             if layer.layer_name not in given_layers:
#                 continue
            
#             # layer.get_field_names() 是获取该层所有字段的直接方式
#             for field_name in layer.get_field_names():
#                 # getattr(layer, field_name) 是获取字段对象的安全方式
#                 field_obj = getattr(layer, field_name)
                
#                 # field_obj.raw_value 提供了我们需要的十六进制原始值
#                 hex_value = field_obj.raw_value
                
#                 if hex_value is None or len(hex_value) >= 64:
#                     continue
                
#                 # 构建pyshark风格的完整字段名
#                 full_field_name = f"{layer.layer_name}.{field_name}"
#                 current_fields[full_field_name] = hex_value

#         # 3. 处理特殊的十进制显示的字段
#         current_fields['tcp.stream'] = packet.tcp.stream
#         current_fields['tcp.len'] = packet.tcp.len
        
#         # 4. 在同一次循环中，处理重组信息
#         # pyshark将重组信息直接附加在TCP层对象上
#         reassembled_in = -1 # 默认为-1，表示不重组
#         if hasattr(packet.tcp, 'reassembled_in'):
#             reassembled_in = int(packet.tcp.reassembled_in)
#         current_fields['tcp.reassembled_segments'] = reassembled_in

#         all_packets_list.append(current_fields)

#     # 循环结束后，一次性创建DataFrame
#     if not all_packets_list:
#         print(f"Warning: No processable TCP packets found in {pcap_path}")
#         return pd.DataFrame()
        
#     df = pd.DataFrame(all_packets_list)

#     # --- 后处理重组逻辑 ---
#     # 这一部分可以在DataFrame层面高效完成
#     if 'tcp.reassembled_segments' in df.columns:
#         df = df.set_index('frame_num')
#         # 创建一个映射，键是作为重组一部分的数据包帧号，值是它们最终被重组到的那个数据包的帧号
#         reassembly_map = df[df['tcp.reassembled_segments'] != -1]['tcp.reassembled_segments'].to_dict()
#         df['group_id'] = df.index.map(reassembly_map)
        
#         # 在同一个tcp.stream内，向前填充group_id，以标记属于同一重组事件的所有包
#         # 因为只有最后一个包才有 reassembled_in 记录
#         if 'tcp.stream' in df.columns:
#             df['group_id'] = df.groupby('tcp.stream')['group_id'].backfill()
        
#         # 用每个组的最小帧号作为该重组块的唯一ID
#         df['final_reassembled_id'] = df.groupby('group_id')['group_id'].transform('min')
#         df['tcp.reassembled_segments'] = df['final_reassembled_id'].fillna(-1).astype(int)
#         df = df.drop(columns=['group_id', 'final_reassembled_id']).reset_index()

#     return df 

# def pcap_to_csv_v2(directory_path, output_directory_path): 
#     pcap_path_list, file_name_list = get_pcap_path(directory_path)
    
#     if pcap_path_list is not None: 
#         for pcap_path, file_name in tqdm(zip(pcap_path_list, file_name_list), desc=f"Processing pcaps in {directory_path}"): 
#             # 只调用这一个优化后的函数
#             df_merged = get_fields_over_layers_v2(pcap_path)
            
#             if not df_merged.empty:
#                 print(f'Original shape: {df_merged.shape}')
#                 df_cleaned = filter_out_nan(df_merged) # 调用您原来的清理函数
#                 print(f'Final shape after cleaning: {df_cleaned.shape}')
#                 df_cleaned.to_csv(os.path.join(output_directory_path, 'merge_' + file_name + '.csv'), index=False)


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
            list_fields = get_fields_over_layers_v2(pcap_path) 
            dict_reassemble = get_reasemmble_info_v2(pcap_path) 
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
