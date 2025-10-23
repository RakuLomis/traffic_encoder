import pyshark 
from tqdm import tqdm 
import os 
import re 
import pandas as pd 
from .dataframe_tools import filter_out_nan 
import json 
from typing import List, Dict
import xml.etree.ElementTree as ET
from collections import defaultdict
import subprocess 

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

def pcap_to_pdml_bulk(pcap_path: str, output_xml_path: str) -> bool:
    """使用 tshark -T pdml 将pcap导出为XML，并进行错误处理。"""
    command = ['tshark', '-r', pcap_path, '-T', 'pdml']
    
    print(f"  -> Running tshark bulk export (mode: pdml) for {os.path.basename(pcap_path)}...")
    try:
        with open(output_xml_path, 'w', encoding='utf-8') as f_out:
            result = subprocess.run(command, stdout=f_out, stderr=subprocess.PIPE, text=True, errors='ignore', check=False)
        
        if result.returncode != 0:
            print(f"  -> 警告: tshark在处理 {os.path.basename(pcap_path)} 时出错。Stderr:\n{result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\n" + "="*60)
        print("!!! 致命错误: 'tshark' 命令未找到 !!!")
        print("请确保您已经安装了 Wireshark，并且 tshark 的路径已经添加到了系统的环境变量(PATH)中。")
        print("您可以在命令行中运行 'tshark -v' 来进行验证。")
        print("="*60)
        return False
        
    print("  -> Bulk export to PDML/XML complete.")
    return True

# def is_field_valid(field_element: ET.Element, max_value_len: int = 64) -> bool:
#     """
#     【核心规则函数】根据一系列规则，判断一个字段是否有效。
#     您可以在这里轻松地添加、删除或修改规则。
    
#     :param field_element: 一个 <field> XML元素。
#     :param max_value_len: 字段'value'属性的最大允许长度。
#     :return: 如果字段有效则返回True，否则返回False。
#     """
#     # 规则1: 'name' 属性必须存在，作为我们的列名
#     if 'name' not in field_element.attrib:
#         return False

#     # 规则2: 'hide' 属性值不能是 'yes'
#     if field_element.get('hide') == 'yes':
#         return False
        
#     # 规则3: 'value' 属性必须存在
#     value = field_element.get('value')
#     if value is None:
#         return False
        
#     # 规则4: 'value' 的长度必须小于 max_value_len
#     if len(value) >= max_value_len:
#         return False
        
#     # 如果所有检查都通过，则该字段有效
#     return True

# def extract_fields_from_packet(packet_element: ET.Element) -> Dict[str, str]:
#     """
#     从一个PDML的 <packet> XML元素中，提取所有【有效】的扁平化字段。
#     """
#     packet_fields = {}
#     for field in packet_element.findall('.//field'):
#         if is_field_valid(field):
#             name = field.get('name')
#             value = field.get('value')
#             packet_fields[name] = value
            
#     # 单独处理并添加frame.number
#     geninfo = packet_element.find("proto[@name='geninfo']")
#     if geninfo is not None:
#         num_field = geninfo.find("field[@name='num']")
#         if num_field is not None:
#             packet_fields['frame.number'] = num_field.get('show')

#     return packet_fields

def is_field_valid(field_element: ET.Element) -> bool:
    """
    【核心规则函数 - 已修正】
    根据一系列规则，判断一个字段是否值得被【考虑】。
    
    :param field_element: 一个 <field> XML元素。
    :return: 如果字段值得被考虑则返回True，否则返回False。
    """
    # 规则1: 'name' 属性必须存在，作为我们的列名
    if 'name' not in field_element.attrib:
        return False

    # 规则2: 'hide' 属性值不能是 'yes'
    if field_element.get('hide') == 'yes':
        return False
        
    # 如果通过了基本检查，就值得被进一步考虑
    return True

def extract_fields_from_packet(packet_element: ET.Element, max_value_len: int = 64) -> Dict[str, str]:
    """
    【修正版】
    从一个PDML的 <packet> XML元素中，提取所有【有效】的扁平化字段。
    实现了“value优先，show备选”的逻辑。
    """
    packet_fields = {}
    for field in packet_element.findall('.//field'):
        
        # 1. 先进行基本的“一票否决”检查
        if is_field_valid(field):
            name = field.get('name')
            
            # ==================== 核心修改点：值获取逻辑 ====================
            
            # 2. 优先尝试获取 'value' (原始/十六进制值)
            value_to_use = field.get('value')
            
            if value_to_use is None:
                # 3. 如果 'value' 不存在, 回退到 'show' (可读值)
                value_to_use = field.get('show')
                
            # =================================================================
            
            # 4. 对我们【最终选择】的值，进行最后的验证
            
            # 规则 3 (新): 必须成功获取到一个值 (既不是None)
            if value_to_use is None:
                continue
                
            # 规则 4 (新): 值的长度必须小于 max_value_len
            if len(value_to_use) >= max_value_len:
                continue

            # 5. 如果所有检查都通过，则该字段有效
            packet_fields[name] = value_to_use
            
    # 单独处理并添加frame.number (此逻辑保持不变)
    geninfo = packet_element.find("proto[@name='geninfo']")
    if geninfo is not None:
        num_field = geninfo.find("field[@name='num']")
        if num_field is not None:
            # frame.number 也总是使用 'show' 属性
            packet_fields['frame.number'] = num_field.get('show')

    return packet_fields

# ==============================================================================
# 2. 主功能函数
# ==============================================================================

def convert_pcap_to_raw_csv(pcap_dir: str, output_dir: str, debug: bool = False):
    """
    遍历目录，将每个pcap文件转换为一个独立的、未对齐的CSV文件。
    """
    print("="*50)
    print("### Flexible PCAP to RAW CSV Converter ###")
    print("="*50)
    
    pcap_files = [f for f in os.listdir(pcap_dir) if f.lower().endswith(('.pcap', '.pcapng'))]
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(pcap_files, desc="Converting pcaps to raw CSVs"):
        print(f"\n--- Processing {filename} ---")
        pcap_path = os.path.join(pcap_dir, filename)
        output_csv_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.csv')
        temp_xml_path = output_csv_path.replace('.csv', '.xml')

        if not pcap_to_pdml_bulk(pcap_path, temp_xml_path):
            continue
        if not os.path.exists(temp_xml_path) or os.path.getsize(temp_xml_path) == 0:
            if os.path.exists(temp_xml_path): os.remove(temp_xml_path)
            continue

        packets_to_load = []
        try:
            # 使用流式解析，高效处理大文件
            context = ET.iterparse(temp_xml_path, events=('end',))
            for _, elem in tqdm(context, desc=f"Parsing {os.path.basename(temp_xml_path)}"):
                if elem.tag == 'packet':
                    packet_fields = extract_fields_from_packet(elem)
                    if packet_fields:
                        packets_to_load.append(packet_fields)
                    elem.clear() # 释放内存

            if not packets_to_load:
                print(f"  -> 警告: 在 {filename} 中未能根据规则提取出任何有效的数据包。")
                continue
            
            # 直接从提取的字段创建DataFrame，不进行对齐
            df = pd.DataFrame(packets_to_load)
            df['label'] = os.path.splitext(filename)[0]
            
            df.to_csv(output_csv_path, index=False)
            print(f"  -> Successfully saved {len(df)} packets with {len(df.columns)} columns to {output_csv_path}")

        except Exception as e:
            print(f"  -> Error during XML parsing or DataFrame processing: {e}")
        finally:
            if not debug and os.path.exists(temp_xml_path):
                os.remove(temp_xml_path)
            elif debug:
                print(f"  -> [DEBUG MODE] Intermediate XML file saved to: {temp_xml_path}")

# ==============================================================================
# 2. 缺失的关键“衔接”函数 (用于低内存解析)
# ==============================================================================

def extract_packets_from_pdml(xml_path: str) -> List[Dict[str, str]]:
    """
    从一个PDML/XML文件中，以流式、低内存的方式，提取所有有效的数据包。
    """
    packets_to_load = []
    try:
        context = ET.iterparse(xml_path, events=('end',))
        # 使用tqdm来显示XML解析进度
        for event, elem in tqdm(context, desc=f"  -> Parsing {os.path.basename(xml_path)}"):
            if elem.tag == 'packet':
                packet_fields = extract_fields_from_packet(elem)
                if packet_fields:
                    packets_to_load.append(packet_fields)
                # 【关键】处理完一个<packet>元素后，立即清除它和它的子元素，释放内存
                elem.clear()
    except ET.ParseError as e:
        print(f"  -> XML解析错误: {e} in file {xml_path}")
    return packets_to_load

# ==============================================================================
# 3. 主流程“编排”函数
# ==============================================================================

def convert_pcap_to_raw_csv_v2(pcap_dir: str, output_dir: str, debug: bool = False):
    """
    主函数，执行“智能发现、按标签分组、统一处理”的完整流程。
    """
    print("="*60)
    print("###   规范化的 PCAP to Labeled CSV 转换器   ###")
    print("="*60)

    # --- 步骤一：智能文件发现与分组 ---
    print(f"\n[1/3] 正在从 {pcap_dir} 及其子目录中发现并分组pcap文件...")
    pcap_groups = defaultdict(list)
    
    # os.walk 会深度遍历所有子目录
    for root, dirs, files in os.walk(pcap_dir):
        for filename in files:
            if filename.lower().endswith(('.pcap', '.pcapng')):
                pcap_path = os.path.join(root, filename)
                
                # --- 智能判断Label ---
                parent_dir_name = os.path.basename(root)
                # 如果pcap文件在一个与根目录不同的子目录中，则认为子目录名是label
                if root != pcap_dir:
                    label = parent_dir_name
                # 否则，从文件名中猜测label (例如 'baidu_1.pcap' -> 'baidu')
                else:
                    label = os.path.splitext(filename)[0].split('_')[0]
                
                pcap_groups[label].append(pcap_path)

    if not pcap_groups:
        print("错误: 未在指定目录或其子目录中找到任何pcap(ng)文件。")
        return
        
    print(f" -> 发现完成，共找到 {len(pcap_groups)} 个标签组。")

    # --- 步骤二：按标签进行处理与合并 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[2/3] 正在按标签组，逐个处理pcap并生成CSV...")

    for label, pcap_paths in tqdm(pcap_groups.items(), desc="Processing Labels"):
        print(f"\n--- 正在处理 Label: '{label}' ({len(pcap_paths)}个pcap文件) ---")
        
        all_packets_for_label = []
        
        for pcap_path in pcap_paths:
            # 为每个pcap生成临时的XML文件
            temp_xml_path = os.path.join(output_dir, f"_temp_{os.path.basename(pcap_path)}.xml")
            
            if pcap_to_pdml_bulk(pcap_path, temp_xml_path):
                if os.path.exists(temp_xml_path) and os.path.getsize(temp_xml_path) > 0:
                    packets = extract_packets_from_pdml(temp_xml_path)
                    all_packets_for_label.extend(packets)
            
            if not debug and os.path.exists(temp_xml_path):
                os.remove(temp_xml_path)

        if not all_packets_for_label:
            print(f" -> 警告: 未能为标签 '{label}' 提取出任何有效的数据包。")
            continue
            
        # --- 步骤三：为每个标签生成最终的CSV文件 ---
        df = pd.DataFrame(all_packets_for_label)
        df['label'] = label # 统一打上正确的标签
        
        output_csv_path = os.path.join(output_dir, f"{label}.csv")
        df.to_csv(output_csv_path, index=False)
        print(f" -> 成功为标签 '{label}' 生成CSV，包含 {len(df)} 个数据包和 {len(df.columns)} 个字段。")
        print(f" -> 文件已保存到: {output_csv_path}")
        
    print("\n[3/3] 所有pcap文件已处理完毕！")
