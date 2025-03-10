import pyshark 

path_tls_pcap = 'E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\Data\\Test\\tls_test_01.pcapng' 
pcap = pyshark.FileCapture(path_tls_pcap) 
packet = pcap[0] 
hex_packet = packet.frame_raw.value 
print(hex_packet) 
pcap.close()