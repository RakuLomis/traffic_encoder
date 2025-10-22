import argparse
from utils.dataframe_tools import generate_summary_tables
import os

DATASET_NAME = 'ISCX-VPN'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析Field Blocks并生成标签分布和特征存在矩阵。")
    parser.add_argument(
        '-d', '--directory', 
        # required=True, 
        default=os.path.join('..', 'TrafficData', 'datasets_csv_add1', 'datasets_fbt', 'merger', DATASET_NAME),
        help="包含所有Block CSV文件的目录路径。"
    )
    parser.add_argument(
        '-lo', '--label_output', 
        default=os.path.join('..', 'res_analysis', DATASET_NAME + '_label_distribution_chiefs.csv'), 
        help="输出标签分布矩阵的CSV文件路径 (默认: ./label_distribution.csv)。"
    )
    parser.add_argument(
        '-fo', '--feature_output', 
        default=os.path.join('..', 'res_analysis', DATASET_NAME + '_feature_presence_chiefs.csv'), 
        help="输出特征存在矩阵的CSV文件路径 (默认: ./feature_presence.csv)。"
    )

    args = parser.parse_args()

    generate_summary_tables(args.directory, args.label_output, args.feature_output)