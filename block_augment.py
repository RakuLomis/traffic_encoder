import pandas as pd
import os
from tqdm import tqdm
import argparse
from typing import Dict, List, Set

def augment_main_block(
    block_dir: str, 
    main_block_name: str, 
    output_path: str, 
    min_samples_threshold: int = 3000
):
    """
    实现“靶向数据补充”策略。
    此版本不考虑结构相似性，旨在最大化数据覆盖度。

    :param block_dir: 包含所有【已合并】的Block CSV文件的目录。
    :param main_block_name: 作为基础的Main Block的名称 (例如 '24')。
    :param output_path: 保存增强后的数据集的CSV文件路径。
    :param min_samples_threshold: 定义稀有类别的样本数阈值。
    """
    print("="*50)
    print("开始执行“最大化覆盖”的数据补充策略...")
    print("="*50)
    main_block_path = os.path.join(block_dir, f"{main_block_name}.csv")
    
    # 1. 加载主Block，并确定其目标Schema
    print(f"\n[步骤 1/4] 加载 Main Block '{main_block_name}'...")
    if not os.path.exists(main_block_path):
        print(f"错误: Main Block文件未找到 -> {main_block_path}")
        return
        
    main_df = pd.read_csv(main_block_path, dtype=str)
    # 存储主Block的列顺序，以备后用
    main_df_columns = main_df.columns.tolist()
    
    # 2. 扫描所有Block，建立一个关于类别分布的“情报数据库”
    print("\n[步骤 2/4] 扫描所有Block，建立类别分布情报库...")
    block_info = {}
    all_files = [f for f in os.listdir(block_dir) if f.lower().endswith('.csv')]
    for filename in tqdm(all_files, desc="Scanning Blocks"):
        block_name = os.path.splitext(filename)[0]
        block_path = os.path.join(block_dir, filename)
        try:
            df_label = pd.read_csv(block_path, dtype=str, usecols=['label'])
            if not df_label.empty:
                block_info[block_name] = df_label['label'].value_counts().to_dict()
        except Exception as e:
            print(f"\n扫描 {filename} 时出错: {e}")

    # 3. 找出Main Block中需要补充的“靶向类别”
    main_label_counts = main_df['label'].value_counts()
    target_classes = main_label_counts[main_label_counts < min_samples_threshold].index.tolist()
    
    all_labels_in_db = set(l for stats in block_info.values() for l in stats)
    missing_classes = list(all_labels_in_db - set(main_label_counts.index))
    target_classes.extend(missing_classes)
    
    print(f"\n[步骤 3/4] 在 Main Block 中找到 {len(target_classes)} 个需要补充的类别:")
    print(sorted(target_classes))

    # 4. 遍历靶向类别，寻找【所有】捐献者并合并数据
    print(f"\n[步骤 4/4] 开始从所有其他Block中寻找并合并补充数据...")
    dfs_to_concat = [main_df]
    
    for target_class in tqdm(target_classes, desc="Augmenting Classes"):
        # 遍历所有其他Block，寻找所有合格的捐献者
        for donor_name, label_counts in block_info.items():
            if donor_name == main_block_name or target_class not in label_counts:
                continue

            # --- 核心修改点：移除了相似度检查 ---
            # 只要这个Block有我们需要的类别，就直接征用
            
            donor_path = os.path.join(block_dir, f"{donor_name}.csv")
            donor_df = pd.read_csv(donor_path, dtype=str)
            supplement_df = donor_df[donor_df['label'] == target_class]
            
            # 特征空间对齐
            aligned_df = pd.DataFrame()
            for col in main_df_columns:
                if col in supplement_df.columns:
                    aligned_df[col] = supplement_df[col]
                else:
                    # 填充缺失的特征列
                    aligned_df[col] = '0'
            
            dfs_to_concat.append(aligned_df)

    # 5. 将所有数据合并成最终的增强版DataFrame
    print("\n正在合并所有数据...")
    if len(dfs_to_concat) > 1:
        augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        augmented_df = main_df
    
    # 6. 保存结果
    augmented_df.to_csv(output_path, index=False)
    print(f"\n数据补充成功！")
    print(f" - 原始 Main Block 样本数: {len(main_df)}")
    print(f" - 补充后总样本数: {len(augmented_df)}")
    print(f" - 最终数据集已保存到: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="通过从其他Block补充稀有类别，来增强Main Block。")
    parser.add_argument('-d', '--block_dir', required=True, help="包含所有Block CSV文件的目录。")
    parser.add_argument('-m', '--main_block', required=True, help="作为基础的Main Block的名称 (例如 '24')。")
    parser.add_argument('-o', '--output_path', required=True, help="保存增强后的数据集的CSV文件路径。")
    parser.add_argument('-t', '--threshold', type=int, default=2000, help="定义稀有类别的样本数阈值。")
    args = parser.parse_args()

    augment_main_block(args.block_dir, args.main_block, args.output_path, args.threshold)