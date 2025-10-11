import pandas as pd
import os
import json
import subprocess
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from collections import defaultdict


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

def is_field_valid(field_element: ET.Element, max_value_len: int = 64) -> bool:
    """
    【核心规则函数】根据一系列规则，判断一个字段是否有效。
    您可以在这里轻松地添加、删除或修改规则。
    
    :param field_element: 一个 <field> XML元素。
    :param max_value_len: 字段'value'属性的最大允许长度。
    :return: 如果字段有效则返回True，否则返回False。
    """
    # 规则1: 'name' 属性必须存在，作为我们的列名
    if 'name' not in field_element.attrib:
        return False

    # 规则2: 'hide' 属性值不能是 'yes'
    if field_element.get('hide') == 'yes':
        return False
        
    # 规则3: 'value' 属性必须存在
    value = field_element.get('value')
    if value is None:
        return False
        
    # 规则4: 'value' 的长度必须小于 max_value_len
    if len(value) >= max_value_len:
        return False
        
    # 如果所有检查都通过，则该字段有效
    return True

def extract_fields_from_packet(packet_element: ET.Element) -> Dict[str, str]:
    """
    从一个PDML的 <packet> XML元素中，提取所有【有效】的扁平化字段。
    """
    packet_fields = {}
    for field in packet_element.findall('.//field'):
        if is_field_valid(field):
            name = field.get('name')
            value = field.get('value')
            packet_fields[name] = value
            
    # 单独处理并添加frame.number
    geninfo = packet_element.find("proto[@name='geninfo']")
    if geninfo is not None:
        num_field = geninfo.find("field[@name='num']")
        if num_field is not None:
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


if __name__ == '__main__':
    # input_dir = os.path.join('..', 'TrafficData', 'dataset_20_d2')
    # output_dir = os.path.join('..', 'TrafficData', 'dataset_20_d2_csv')

    # convert_pcap_to_raw_csv(input_dir, output_dir)
    input_dir = os.path.join('..', 'TrafficData', 'datasets_raw','dataset_29_d1')
    output_dir = os.path.join('..', 'TrafficData', 'datasets_csv','dataset_29_d1')

    convert_pcap_to_raw_csv_v2(input_dir, output_dir)