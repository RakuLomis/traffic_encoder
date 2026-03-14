import pandas as pd
import os
from tqdm import tqdm
import argparse
from typing import List

def consolidate_raw_csvs(input_dir: str, output_path: str):
    """
    整合一个目录中的所有原始CSV文件，执行字段筛选、标签合并和对齐。

    :param input_dir: 包含所有原始(raw)CSV文件的目录。
    :param output_path: 保存最终整合后的总数据集的CSV文件路径。
    """
    print("="*60)
    print("###   开始整合与清洗原始CSV文件   ###")
    print("="*60)

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        return
        
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {input_dir} 中未找到任何CSV文件。")
        return

    # 定义我们想要保留的协议层前缀
    PREFIXES_TO_KEEP = ('eth.', 'ip.', 'tcp.', 'tls.')
    
    all_dfs: List[pd.DataFrame] = []

    print(f"\n[步骤 1/3] 正在加载、筛选和清洗 {len(all_files)} 个CSV文件...")
    for filename in tqdm(all_files, desc="Processing raw CSVs"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            # --- 步骤 3: 标签合并 ---
            # 假设标签来源于文件名，例如 'baidu_1.csv' -> 'baidu'
            base_label = os.path.splitext(filename)[0].split('_')[0]
            df['label'] = base_label
            
            # --- 步骤 1: 字段筛选 ---
            original_cols = df.columns.tolist()
            # 我们总是保留 'label' 列，以及所有符合前缀要求的特征列
            cols_to_keep = ['label']
            for col in original_cols:
                if col.startswith(PREFIXES_TO_KEEP):
                    cols_to_keep.append(col)
            
            # 应用筛选
            df_filtered = df[cols_to_keep]
            
            all_dfs.append(df_filtered)

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")
            
    if not all_dfs:
        print("未能成功处理任何CSV文件。")
        return

    # --- 步骤 4: 合并对齐一个总的CSV数据集 ---
    print("\n[步骤 2/3] 正在将所有数据合并为一个总数据集...")
    # pd.concat 会自动取所有列的并集，并在不存在值的地方填充NaN (步骤2)
    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    print(f" -> 合并完成，总数据集包含 {len(consolidated_df)} 条记录和 {len(consolidated_df.columns)} 个字段。")

    # --- 步骤 5: 创建新的frame_num作为索引 ---
    print("\n[步骤 3/3] 正在创建新的唯一索引 'frame_num'...")
    # 先删除可能存在的旧的、不连续的 'frame_num' 列
    if 'frame_num' in consolidated_df.columns:
        consolidated_df = consolidated_df.drop(columns=['frame_num'])
        
    # 创建一个从1开始的新索引
    consolidated_df.insert(0, 'frame_num', range(1, len(consolidated_df) + 1))
    
    # 调整列顺序，将label也放在前面
    if 'label' in consolidated_df.columns:
        cols = consolidated_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('label')))
        consolidated_df = consolidated_df[cols]

    # --- 保存最终结果 ---
    try:
        consolidated_df.to_csv(output_path, index=False)
        print(f"\n总数据集已成功保存到: {output_path}")
    except Exception as e:
        print(f"\n保存总数据集时失败: {e}")

def consolidate_raw_csvs_memory_optimized(input_dir: str, output_path: str):
    """
    【内存优化版】整合一个目录中的所有原始CSV文件。
    采用“两遍扫描，流式写入”的策略，以极低的内存开销处理大量数据。
    """
    print("="*60)
    print("###   开始【内存优化版】的整合与清洗流程   ###")
    print("="*60)

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        return
        
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"错误: 在目录 {input_dir} 中未找到任何CSV文件。")
        return

    PREFIXES_TO_KEEP = ('eth.', 'ip.', 'tcp.', 'tls.')
    
    # --- 第一遍扫描：确定全局Schema ---
    print(f"\n[Pass 1/2] 正在扫描 {len(all_files)} 个文件的表头以确定全局Schema...")
    universal_schema = {'label'} # 始终包含label列
    for filename in tqdm(all_files, desc="Scanning Schemas"):
        file_path = os.path.join(input_dir, filename)
        try:
            # 只读第一行（表头）来获取列名，速度极快，内存占用极小
            df_header = pd.read_csv(file_path, dtype=str, nrows=0)
            
            # 筛选符合前缀的字段
            filtered_cols = {col for col in df_header.columns if col.startswith(PREFIXES_TO_KEEP)}
            universal_schema.update(filtered_cols)
        except Exception as e:
            print(f"\n扫描文件 {filename} 表头时出错: {e}")

    # 确定最终的、有序的列名列表
    final_columns = ['frame_num', 'label'] + sorted(list(universal_schema - {'label'}))
    print(f" -> 全局Schema确定，共 {len(final_columns)} 个最终字段。")

    # --- 第二遍扫描：逐块处理并流式写入 ---
    print(f"\n[Pass 2/2] 正在逐个处理文件并写入到 {output_path}...")
    
    # a) 首先，创建输出文件并写入表头
    pd.DataFrame(columns=final_columns).to_csv(output_path, index=False)
    
    global_row_counter = 0
    for filename in tqdm(all_files, desc="Processing and Appending"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            # 1. 标签合并
            base_label = os.path.splitext(filename)[0].split('_')[0]
            df['label'] = base_label
            
            # 2. 特征空间对齐
            #    - reindex 会自动添加缺失列（值为NaN），并删除多余列
            df_aligned = df.reindex(columns=final_columns)
            
            # 3. 创建新的frame_num (因为是追加模式，需要手动计算)
            num_rows = len(df)
            df_aligned['frame_num'] = range(global_row_counter + 1, global_row_counter + 1 + num_rows)
            global_row_counter += num_rows
            
            # 4. 【核心】将处理好的数据【追加】到输出文件中，不写表头
            df_aligned.to_csv(output_path, mode='a', header=False, index=False)

        except Exception as e:
            print(f"\n处理并写入文件 {filename} 时发生错误: {e}")
    
    print(f"\n总数据集已成功保存到: {output_path}")
    print(f"总计处理了 {global_row_counter} 条记录。")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="将多个原始CSV文件，整合成一个统一的、经过筛选和清洗的总数据集。")
    # parser.add_argument(
    #     '-i', '--input_dir', 
    #     required=True, 
    #     help="包含所有待处理的原始CSV文件的目录路径。"
    # )
    # parser.add_argument(
    #     '-o', '--output_path', 
    #     required=True, 
    #     help="保存最终整合后的总数据集的CSV文件路径。"
    # )
    # args = parser.parse_args()
    input_dir = os.path.join('..', 'TrafficData', 'datasets_csv','dataset_20_d2') 
    output_path = os.path.join('..','TrafficData', 'datasets_consolidate', 'dataset_20_d2.csv') 
    consolidate_raw_csvs_memory_optimized(input_dir, output_path)
    # consolidate_raw_csvs_memory_optimized(input_dir, output_path)