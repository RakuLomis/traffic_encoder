import pyshark 
import binascii 
from scapy.all import *

path_tls_pcap = 'E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\Data\\Test\\tls_test_01.pcapng' 
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

pcap_test = rdpcap(path_tls_pcap) 
print('end.')