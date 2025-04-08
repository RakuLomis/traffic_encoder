import pyshark 
from binascii import hexlify
from scapy.all import rdpcap
import yaml 


path_tls_pcap = 'E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\Data\\Test\\tls_test_01.pcapng' 
pcap_test = rdpcap(path_tls_pcap) 
packet_0 = pcap_test[0] 
print(packet_0)
 
# pcap = pyshark.FileCapture(path_tls_pcap,use_json=True, include_raw=True) # raw data must keep use_json and include_raw be True
# pcap = pyshark.FileCapture(path_tls_pcap)
# packet = pcap[0] 
# packet2 = pcap[142] # quic 
# info = packet2.frame_info
# print(packet.transport_layer, type(packet.transport_layer)) 
# print(info.get_field('number'), type(int(info.get_field('number'))))
# raw_packet = packet.get_raw_packet() 

# hex_str = binascii.hexlify(raw_packet, sep=' ').decode()
# print(hex_str) 
# pcap.close()

def load_from_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['protocols'] 

def extract_bits(raw_data, bit_offset, bit_length):
    # 将字节流转为完整的二进制字符串
    full_binary = ''.join(format(byte, '08b') for byte in raw_data)
    
    # 检查是否越界
    if bit_offset + bit_length > len(full_binary):
        return None
    
    # 提取指定位范围
    return full_binary[bit_offset:bit_offset + bit_length]

def parse_packet(packet, protocols):
    raw_data = bytes(packet)  # 获取数据包的原始字节
    result = {}
    current_offset = 0
    current_proto = 'ETH'  # 从以太网层开始

    while current_proto and current_offset < len(raw_data):
        if current_proto not in protocols:
            break

        proto_def = protocols[current_proto]
        result[current_proto] = {}
        fields = proto_def['fields']

        # 解析当前层字段
        for field in fields:
            offset = current_offset + field['offset']
            length = field['length']
            field_name = field['name']
            field_type = field['type']

            # 处理动态长度
            if length == 'dynamic':
                length = len(raw_data) - offset
            else:
                length = int(length)

            # 提取字段数据
            if offset + length <= len(raw_data):
                field_data = raw_data[offset:offset + length]

                # 根据类型转换
                if field_type == 'hex': # 即使是十六进制也要掩码操作得到真实值
                    if 'bitmask' in field: 
                        value = hex(int.from_bytes(field_data, 'big'))[2:]
                        mask = field['bitmask'] 
                        if isinstance(mask, str): 
                            mask = int(field['bitmask'], 16) 
                        value = hex(int(value, 16) & mask)[2:] 
                    else: 
                        value = hexlify(field_data).decode('utf-8')
                elif field_type == 'binary':
                    value = bin(int.from_bytes(field_data, 'big'))[2:].zfill(length * 8) 
                    # [2:]: 删去二进制标志位0b, 只保留数据部分
                    # zfill: 二进制直接去除会忽略左侧的0, 所以要填充
                    if 'bitmask' in field: 
                        mask = field['bitmask'] # 十六进制直接是整数, 不用转换
                        if isinstance(mask, str): 
                            mask = int(field['bitmask'], 16)
                        value = bin(int(value, 2) & mask)[2:].zfill(length * 8)
                        if 'shift' in field:
                            value = bin(int(value, 2) >> field['shift'])[2:] 
                            value = value.zfill((int(len(value) / 4) + 1) * 4) # 填充位数至4的整数倍
                else:
                    value = field_data

                result[current_proto][field_name] = value

        # 计算当前层长度并更新偏移量
        layer_length = max(field['offset'] + (int(field['length']) if field['length'] != 'dynamic' else 0) 
                          for field in fields)
        current_offset += layer_length

        # 确定下一层协议
        if 'next_layer_map' in proto_def:
            proto_value = result[current_proto].get('Protocol')  # 示例：IP的Protocol字段
            current_proto = proto_def['next_layer_map'].get(proto_value)
        elif 'next_layer' in proto_def:
            current_proto = proto_def['next_layer']
        else:
            current_proto = None

    return result 

def get_field_value(info: dict, proto: str, field: str): 
    """
    Get the specific field value in the handled dictonary not raw data. 
    """
    value = info[proto][field]
    if isinstance(value, str): 
        value = int(value)
    return value 

def parse_packet_bit(packet, protocols): 
    raw_data = bytes(packet)  # get original byte info of packet
    result = {}
    current_bit_offset = 0
    current_proto = 'ETH'  # 从以太网层开始

    while current_proto and current_bit_offset < len(raw_data) * 8:
        if current_proto not in protocols:
            break

        proto_def = protocols[current_proto]
        result[current_proto] = {}
        fields = proto_def['fields']

        # 解析当前层字段
        for field in fields:
            bit_offset = current_bit_offset + field['bit_offset']
            bit_length = field['bit_length']
            field_name = field['name'] 

            if 'details' in field: 
                pass 

            # 处理动态长度
            if bit_length == 'dynamic': 
                if current_proto == 'IP': 
                    bit_length = get_field_value(result, current_proto, 'Total_Length') - bit_offset 
                elif current_proto == 'TCP': 
                    bit_length = get_field_value(result, current_proto, 'Data_Offset') - bit_offset 
                else: 
                    bit_length = len(raw_data) * 8 - bit_offset 
            else:
                bit_length = int(bit_length)

            # 提取字段二进制值
            value = extract_bits(raw_data, bit_offset, bit_length)
            if value is not None:
                result[current_proto][field_name] = value

        # 更新位偏移量
        # proto_bit_length = proto_def['bit_length']
        proto_bit_length = max(field['bit_offset'] + (int(field['bit_length']) if field['bit_length'] != 'dynamic' else 0) 
                          for field in fields) 
        current_bit_offset += proto_bit_length

        # 确定下一层协议
        if 'next_layer_map' in proto_def:
            proto_value = result[current_proto].get('Protocol')  # 示例：IP的Protocol字段
            current_proto = proto_def['next_layer_map'].get(proto_value)
        elif 'next_layer' in proto_def:
            current_proto = proto_def['next_layer']
        else:
            current_proto = None

    return result

# protocol_rules =  load_from_yaml('E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\utils\\fields.yaml') 
protocol_rules_bit = load_from_yaml('E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\utils\\fields_bits.yaml') 
parsed_data = parse_packet_bit(packet_0, protocol_rules_bit) 
print(parsed_data)