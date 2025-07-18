import os 
from utils.pcap_tools import pcap_to_csv 
from utils.pcap_tools import pcap_to_csv_v2 
from utils.pcap_tools import pcap_to_csv_v3 
from tqdm import tqdm 

pcap_directory_01 = os.path.join('..', 'TrafficData', 'dataset_29_d1')
output_directory_csv = os.path.join(pcap_directory_01 + '_csv') 
test_pcap_path = os.path.join('.', 'Data', 'Test')

# # 检查目录是否存在
# if os.path.exists(pcap_directory_01):
#     print(f"Directory is existed: {pcap_directory_01}")
    
#     # 获取目录下的所有内容
#     items = os.listdir(pcap_directory_01)
    
#     # 筛选文件夹名称
#     folders = [item for item in items if os.path.isdir(os.path.join(pcap_directory_01, item))]
    
#     print("目录下的文件夹名称:")
#     for folder in tqdm(folders):
#         pcap_folder = os.path.join(pcap_directory_01, folder) 
#         output_folder = os.path.join(output_directory_csv, folder)
#         os.makedirs(output_folder, exist_ok=True) 
#         pcap_to_csv_v2(pcap_folder, output_folder) 
# else:
#     print(f"目录不存在: {pcap_directory_01}") 

output_test_path = os.path.join('.', 'Data', 'Test', 'optimized_tls_test_01.csv') 

pcap_to_csv_v3(test_pcap_path, test_pcap_path)