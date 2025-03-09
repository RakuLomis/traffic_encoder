import pyshark 

def read_pcap_file(pcap_path: str): 
    pcap = pyshark.FileCapture(pcap_path) 
    return pcap 

path_tls_pcap = 'E:\\Program\\VSCode\\MyGit\\TrafficEncoder\\Data\\Test\\tls_test_01.pcapng' 
pcap = read_pcap_file(path_tls_pcap) 
print(pcap[250])
print(pcap[251])