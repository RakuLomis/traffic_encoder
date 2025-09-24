import argparse
from utils.dataframe_tools import generate_summary_tables
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析Field Blocks并生成标签分布和特征存在矩阵。")
    parser.add_argument(
        '-d', '--directory', 
        # required=True, 
        default=os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'reborn_blocks_merge_ps'),
        help="包含所有Block CSV文件的目录路径。"
    )
    parser.add_argument(
        '-lo', '--label_output', 
        default=os.path.join('.', 'label_distribution_ps.csv'), 
        help="输出标签分布矩阵的CSV文件路径 (默认: ./label_distribution.csv)。"
    )
    parser.add_argument(
        '-fo', '--feature_output', 
        default=os.path.join('.', 'feature_presence_ps.csv'), 
        help="输出特征存在矩阵的CSV文件路径 (默认: ./feature_presence.csv)。"
    )

    args = parser.parse_args()

    generate_summary_tables(args.directory, args.label_output, args.feature_output)