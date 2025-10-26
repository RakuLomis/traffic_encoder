import pandas as pd
import os
import subprocess
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional, TextIO
import xml.etree.ElementTree as ET
from collections import defaultdict
import gc

# ==============================================================================
# 1. 您提供的核心工具函数 (pcap_to_pdml_bulk, is_field_valid, extract_fields_from_packet)
#    我们保持它们不变，因为它们的逻辑是正确的。
# ==============================================================================

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
    """【核心规则函数】根据一系列规则，判断一个字段是否有效。"""
    if 'name' not in field_element.attrib:
        return False
    if field_element.get('hide') == 'yes':
        return False
    value = field_element.get('value')
    if value is None:
        # 【修正】允许'show'属性
        value = field_element.get('show')
        if value is None:
            return False
    if len(value) >= max_value_len:
        return False
    return True

def extract_fields_from_packet(packet_element: ET.Element, max_value_len: int = 64) -> Dict[str, str]:
    """【修正版】从PDML元素中提取字段，实现“value优先，show备选”的逻辑。"""
    packet_fields = {}
    for field in packet_element.findall('.//field'):
        if 'name' not in field.attrib:
            continue
        if field.get('hide') == 'yes':
            continue
            
        name = field.get('name')
        value_to_use = field.get('value')
        
        if value_to_use is None:
            value_to_use = field.get('show')
            
        if value_to_use is None or len(value_to_use) >= max_value_len:
            continue
            
        packet_fields[name] = value_to_use
            
    geninfo = packet_element.find("proto[@name='geninfo']")
    if geninfo is not None:
        num_field = geninfo.find("field[@name='num']")
        if num_field is not None:
            packet_fields['frame.number'] = num_field.get('show')

    return packet_fields

# ==============================================================================
# 2. 【核心修改点】全新的、流式处理的XML解析/写入函数
# ==============================================================================

def stream_xml_to_csv(
    xml_path: str, 
    output_csv_file: TextIO, 
    label: str, 
    is_first_write_for_label: bool,
    chunk_size: int = 50000 # 每次在内存中处理的包数量
):
    """
    【真正低内存版】
    以流式解析XML，分块（chunk）转换为DataFrame，并追加（append）写入到
    一个已经打开的CSV文件中。

    :param xml_path: 临时的PDML/XML文件路径。
    :param output_csv_file: 一个已打开的、用于写入CSV的文件句柄。
    :param label: 要为这批数据打上的标签。
    :param is_first_write_for_label: 这是否是为这个label写入的第一个pcap。
    :param chunk_size: 批处理大小。
    :return: (是否成功, 是否是第一次写入)
    """
    packets_chunk: List[Dict[str, str]] = []
    
    try:
        context = ET.iterparse(xml_path, events=('end',))
        
        for _, elem in tqdm(context, desc=f"  -> Parsing {os.path.basename(xml_path)}"):
            if elem.tag == 'packet':
                packet_fields = extract_fields_from_packet(elem)
                if packet_fields:
                    packets_chunk.append(packet_fields)
                
                # 当累积的包达到批次大小时，就处理并写入
                if len(packets_chunk) >= chunk_size:
                    df_chunk = pd.DataFrame(packets_chunk)
                    df_chunk['label'] = label
                    
                    if is_first_write_for_label:
                        df_chunk.to_csv(output_csv_file, mode='w', header=True, index=False)
                        is_first_write_for_label = False # 关键：只在第一次写入时写表头
                    else:
                        # 后续追加时，必须确保列一致
                        header_list = pd.read_csv(output_csv_file.name, dtype=str, nrows=0).columns.tolist()
                        df_chunk = df_chunk.reindex(columns=header_list, fill_value='0')
                        df_chunk.to_csv(output_csv_file, mode='a', header=False, index=False)
                    
                    packets_chunk = [] # 清空批次列表
                
                elem.clear() # 释放XML元素的内存
        
        # 处理最后一个不满批次的“尾巴”数据
        if packets_chunk:
            df_chunk = pd.DataFrame(packets_chunk)
            df_chunk['label'] = label
            
            if is_first_write_for_label:
                df_chunk.to_csv(output_csv_file, mode='w', header=True, index=False)
                is_first_write_for_label = False
            else:
                header_list = pd.read_csv(output_csv_file.name, dtype=str, nrows=0).columns.tolist()
                df_chunk = df_chunk.reindex(columns=header_list, fill_value='0')
                df_chunk.to_csv(output_csv_file, mode='a', header=False, index=False)
            
            del df_chunk # 显式清理
            
    except ET.ParseError as e:
        print(f"  -> XML解析错误: {e} in file {xml_path}")
        return False, is_first_write_for_label
    
    return True, is_first_write_for_label

# 3. 【核心修改点】主流程“编排”函数
# ==============================================================================

def stream_pcap_to_csv_chunked(
    pcap_path: str, 
    output_csv_file: TextIO, # 这是一个已打开的文件句柄
    label: str, 
    header_list: Optional[List[str]], # 接收一个表头列表
    chunk_size: int = 50000,
    debug: bool = False
) -> (bool, List[str]): # type: ignore # 返回是否成功，以及最终的表头
    """
    【真正低内存、低磁盘占用版】
    通过内存管道，将tshark的PDML输出实时流式传输给XML解析器，
    并分块写入CSV，不产生大型临时文件。
    """
    print(f"  -> Starting real-time stream processing for {os.path.basename(pcap_path)}...")
    command = ['tshark', '-r', pcap_path, '-T', 'pdml']
    process = None
    
    try:
        # 1. 【关键】使用Popen启动tshark进程，并将stdout重定向到管道
        # 我们【不】指定 text=True 或 encoding，直接读取原始字节流
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        packets_chunk: List[Dict[str, str]] = []
        is_first_write_for_pcap = True 

        # 2. 【关键】将【原始字节流】(process.stdout) 直接喂给 iterparse
        context = ET.iterparse(process.stdout, events=('end',))
        
        for _, elem in tqdm(context, desc=f"  -> Streaming {os.path.basename(pcap_path)}"):
            if elem.tag == 'packet':
                packet_fields = extract_fields_from_packet(elem)
                if packet_fields:
                    packets_chunk.append(packet_fields)
                
                # 3. 累积到批次大小后，处理并写入
                if len(packets_chunk) >= chunk_size:
                    df_chunk = pd.DataFrame(packets_chunk)
                    df_chunk['label'] = label
                    
                    if header_list is None: # 这一定是这个Label的第一批数据
                        header_list = df_chunk.columns.tolist() # 从数据中动态确定表头
                        # 确保元数据列在前面
                        meta_cols = [c for c in ['frame.number', 'label'] if c in header_list]
                        feature_cols = [c for c in header_list if c not in meta_cols]
                        header_list = meta_cols + sorted(feature_cols)
                        
                        df_chunk = df_chunk[header_list]
                        df_chunk.to_csv(output_csv_file, mode='w', header=True, index=False)
                    else:
                        # 使用已有的表头进行对齐
                        df_chunk = df_chunk.reindex(columns=header_list, fill_value='0')
                        df_chunk.to_csv(output_csv_file, mode='a', header=False, index=False)
                    
                    packets_chunk = [] # 清空批次列表
                
                elem.clear() 
        
        # 4. 处理最后一个不满批次的“尾巴”数据
        if packets_chunk:
            df_chunk = pd.DataFrame(packets_chunk)
            df_chunk['label'] = label
            if header_list is None:
                header_list = df_chunk.columns.tolist()
                meta_cols = [c for c in ['frame.number', 'label'] if c in header_list]
                feature_cols = [c for c in header_list if c not in meta_cols]
                header_list = meta_cols + sorted(feature_cols)
                
                df_chunk = df_chunk[header_list]
                df_chunk.to_csv(output_csv_file, mode='w', header=True, index=False)
            else:
                df_chunk = df_chunk.reindex(columns=header_list, fill_value='0')
                df_chunk.to_csv(output_csv_file, mode='a', header=False, index=False)
        
        # 5. 检查tshark进程是否在后台报错
        stdout, stderr = process.communicate() 
        if process.returncode != 0:
            # 【关键】安全地解码 stderr，使用 'latin-1' 作为“万能”解码器
            error_message = stderr.decode('latin-1', 'ignore')
            print(f"  -> 警告: tshark在处理 {os.path.basename(pcap_path)} 时出错。Stderr:\n{error_message}")
            return False, header_list

    except FileNotFoundError:
        print("\n" + "="*60)
        print("!!! 致命错误: 'tshark' 命令未找到 !!!")
        print("请确保您已经安装了 Wireshark，并且 tshark 的路径已经添加到了系统的环境变量(PATH)中。")
        print("您可以在命令行中运行 'tshark -v' 来进行验证。")
        print("="*60)
        return False, header_list
    except ET.ParseError as e:
        print(f"  -> XML解析错误 (文件可能已损坏或tshark输出中断): {e}")
        return False, header_list
    except Exception as e:
        try:
            print(f"  -> 处理流时发生未知错误: {e}")
        except UnicodeEncodeError:
            safe_error = str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            print(f"  -> 处理流时发生未知错误 (已净化): {safe_error}")
        return False, header_list
    finally:
        # 【关键】确保tshark子进程被彻底关闭
        if process:
            try: process.kill()
            except Exception: pass
                
    return True, header_list

# ==============================================================================
# 3. 【核心修改点】主流程“编排”函数
# ==============================================================================

def convert_pcap_to_raw_csv(pcap_dir: str, output_dir: str, debug: bool = False):
    """
    【最终内存/磁盘优化版 Step 1】
    主函数，执行“智能发现、按标签分组、实时流式写入”的完整流程。
    """
    print("="*60)
    print("###   规范化的 PCAP to Labeled CSV 转换器 (实时流版)   ###")
    print("="*60)

    # --- 步骤一：智能文件发现与分组 (无变化) ---
    print(f"\n[1/3] 正在从 {pcap_dir} 及其子目录中发现并分组pcap文件...")
    pcap_groups = defaultdict(list)
    for root, dirs, files in os.walk(pcap_dir):
        for filename in files:
            if filename.lower().endswith(('.pcap', '.pcapng')):
                pcap_path = os.path.join(root, filename)
                parent_dir_name = os.path.basename(root)
                label = parent_dir_name if root != pcap_dir else os.path.splitext(filename)[0].split('_')[0]
                pcap_groups[label].append(pcap_path)
    
    if not pcap_groups:
        print(f"错误: 在 {pcap_dir} 中未找到任何pcap(ng)文件。")
        return False
    print(f" -> 发现完成，共找到 {len(pcap_groups)} 个标签组。")

    # --- 步骤二：按标签进行处理与合并 (实现实时流式写入) ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[2/3] 正在按标签组，逐个处理pcap并【实时流式写入】CSV...")

    for label, pcap_paths in tqdm(pcap_groups.items(), desc="Processing Labels"):
        print(f"\n--- 正在处理 Label: '{label}' ({len(pcap_paths)}个pcap文件) ---")
        
        output_csv_path = os.path.join(output_dir, f"{label}.csv")
        header_list = None # 每个新label的表头都是未知的
        
        # 【关键】如果文件已存在，则重置
        if os.path.exists(output_csv_path):
             if debug:
                 print(f" -> [DEBUG] 文件 {label}.csv 已存在，将被覆盖。")
                 # 强制覆盖
             else:
                 print(f" -> 文件 {label}.csv 已存在，将跳过。") 
                 continue # 跳过这个label
        
        try:
            # 【关键】我们在循环外部打开文件句柄
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as output_csv_file:
                for pcap_path in pcap_paths:
                    
                    # 【关键】调用新的实时流式处理函数
                    success, new_header_list = stream_pcap_to_csv_chunked(
                        pcap_path, 
                        output_csv_file, 
                        label, 
                        header_list # 传入当前的表头
                    )
                    
                    if success and header_list is None:
                        # 在第一次成功写入后，锁定这个label的表头
                        header_list = new_header_list
                        
                    gc.collect() 
                        
        except Exception as e:
            print(f"  -> 写入文件 {output_csv_path} 时发生严重错误: {e}")
            continue
            
        if header_list is None: # 意味着没有写入任何数据
            print(f" -> 警告: 未能为标签 '{label}' 提取出任何有效的数据包。")
            if os.path.exists(output_csv_path):
                os.remove(output_csv_path) # 删除创建的空文件
        else:
            print(f" -> 成功为标签 '{label}' 生成CSV文件: {output_csv_path}")
            
    print("\n[3/3] 所有pcap文件已处理完毕！")
    return True
# ==============================================================================
# 4. 命令行接口
# ==============================================================================