import os
import argparse
import shutil
from typing import Dict 
from utils.pcap_tools import convert_pcap_to_raw_csv_v2 
from utils.dataframe_tools import consolidate_raw_csvs_memory_optimized
from utils.dataframe_tools import global_stratified_split_memory_optimized
from utils.dataframe_tools import truncate_to_block_by_schema, augment_main_block_v2
from utils.pruning_and_merge import merge_field_blocks_tree_similarity 

def run_full_pipeline(raw_data_root: str, output_root: str, force_overwrite: bool = False):
    """
    执行完整的数据预处理流水线，从pcap到最终的增强版训练集。
    """
    print("="*80)
    print("###   启动全自动数据预处理流水线   ###")
    print("="*80)

    # 1. 自动发现在 raw_data_root 下的所有数据集
    try:
        dataset_names = [d for d in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, d))]
    except FileNotFoundError:
        print(f"错误: 原始数据集根目录未找到 -> {raw_data_root}")
        return
        
    if not dataset_names:
        print(f"警告: 在 {raw_data_root} 中未找到任何数据集子目录。")
        return
        
    print(f"发现 {len(dataset_names)} 个待处理的数据集: {dataset_names}\n")

    # 2. 为每个数据集执行完整的处理流程
    for dataset_name in dataset_names:
        print("\n" + "#"*80)
        print(f"###   开始处理数据集: {dataset_name}   ###")
        print("#"*80)

        # a) 生成该数据集的所有相关路径
        paths = {
            "pcap_dir": os.path.join(raw_data_root, dataset_name), # 源pcap文件夹
            "raw_csv_dir": os.path.join(output_root, 'datasets_csv', dataset_name), # 按label转换成csv
            "consolidated_csv": os.path.join(output_root, 'datasets_consolidate', f"{dataset_name}.csv"), # labelcsv合并成总的csv
            "split_dir": os.path.join(output_root, 'datasets_split', dataset_name), # train, val, test数据集划分
            "truncated_blocks_dir": os.path.join(output_root, 'datasets_fbt', 'truncation', dataset_name), # truncate成小block
            "merged_blocks_dir": os.path.join(output_root, 'datasets_fbt', 'merger', dataset_name), # 第一次合并
            "augmented_train_set": os.path.join(output_root, 'datasets_final', f"{dataset_name}_chief_block_augmented.csv") # chief block
        }

        # b) 依次执行流水线的每一步
        try:
            # --- Step 1: PCAP to CSV ---
            print("\n>>> Step 1/3: PCAP -> Raw CSVs")
            # 检查输出目录是否已存在且非空
            if not force_overwrite and os.path.exists(paths['raw_csv_dir']) and os.listdir(paths['raw_csv_dir']):
                print(f" -> 输出目录 {paths['raw_csv_dir']} 已存在且非空，跳过此步骤。")
            else:
                if force_overwrite and os.path.exists(paths['raw_csv_dir']):
                    print(f" -> [FORCE] 正在删除旧目录: {paths['raw_csv_dir']}")
                    shutil.rmtree(paths['raw_csv_dir'])
                convert_pcap_to_raw_csv_v2(paths['pcap_dir'], paths['raw_csv_dir'])
            print(" -> Step 1 完成。")

            # --- Step 2: Consolidate CSVs ---
            print("\n>>> Step 2/3: Raw CSVs -> Consolidated CSV")
            if not force_overwrite and os.path.exists(paths['consolidated_csv']):
                print(f" -> 输出文件 {paths['consolidated_csv']} 已存在，跳过此步骤。")
            else:
                consolidate_raw_csvs_memory_optimized(paths['raw_csv_dir'], paths['consolidated_csv'])
            print(" -> Step 2 完成。")

            # --- Step 3: Split, Truncate, Merge & Augment ---
            print("\n>>> Step 3/3: Final Processing (Split, Truncate, Merge, Augment)")
            if not force_overwrite and os.path.exists(paths['augmented_train_set']):
                print(f" -> 最终训练集 {paths['augmented_train_set']} 已存在，跳过此步骤。")
            else:
                # 3.1: 全局分割
                print("\n  -> Sub-step 3.1: Global Stratified Split...")
                global_stratified_split_memory_optimized(paths['consolidated_csv'], paths['split_dir'])
                
                # 3.2: FBT切块
                print("\n  -> Sub-step 3.2: Field Block Truncation...")
                train_set_path = os.path.join(paths['split_dir'], 'train_set.csv')
                truncate_to_block_by_schema(train_set_path, paths['truncated_blocks_dir'])
                
                # 3.3: 结构化合并
                print("\n  -> Sub-step 3.3: Structurally-aware Merging...")
                merge_field_blocks_tree_similarity(paths['truncated_blocks_dir'], paths['merged_blocks_dir'], similarity_threshold=0.8)

                # 3.4: 主干增强
                print("\n  -> Sub-step 3.4: Chief Block Augmentation...")
                os.makedirs(os.path.dirname(paths['augmented_train_set']), exist_ok=True)
                augment_main_block_v2(paths['merged_blocks_dir'], paths['augmented_train_set'])
            print(" -> Step 3 完成。")
            
            print(f"\n### 数据集 '{dataset_name}' 已成功处理完毕！###")
            print(f"最终可用于训练的文件位于: {paths['augmented_train_set']}")

        except Exception as e:
            print(f"\n!!!!!! 在处理数据集 '{dataset_name}' 时发生严重错误: {e} !!!!!!")
            print("!!!!!! 流水线已在该数据集处中断，请检查错误并重试。 !!!!!!")
            continue # 继续处理下一个数据集

    print("\n" + "="*80)
    print("###   所有数据集已处理完毕。   ###")
    print("="*80)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="全自动数据预处理流水线，从pcap到最终增强版训练集。")
    # parser.add_argument(
    #     '-i', '--input_root', 
    #     required=True, 
    #     help="包含所有待处理数据集子目录的根目录 (例如 '../TrafficData/datasets_raw')。"
    # )
    # parser.add_argument(
    #     '-o', '--output_root', 
    #     required=True, 
    #     help="用于存放所有预处理阶段输出的根目录 (例如 '../TrafficData')。"
    # )
    # parser.add_argument(
    #     '--force',
    #     action='store_true',
    #     help="如果指定，将强制覆盖并重新生成所有已存在的文件和目录。"
    # )
    
    # args = parser.parse_args()
    # run_full_pipeline(args.input_root, args.output_root, args.force)

    input_root = os.path.join('..', 'TrafficData', 'datasets_raw_add1') 
    output_root = os.path.join('..', 'TrafficData', 'datasets_csv_add1')

    run_full_pipeline(input_root, output_root)