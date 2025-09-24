import pandas as pd
import os
import json
from tqdm import tqdm
import argparse
from typing import Set, Dict, List, Tuple, Any
from utils.dataframe_tools import generate_protocol_tree_and_nodes

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """计算两个集合的Jaccard相似度"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def merge_field_blocks(
    source_dir: str, 
    output_dir: str, 
    min_samples_threshold: int = 1000, 
    jaccard_threshold: float = 0.5
):
    """
    根据特征集相似度，智能地合并和剪枝Field Blocks。

    :param source_dir: 包含所有原始Field Block CSV文件的目录。
    :param output_dir: 保存合并后的新Field Block CSV文件的目录。
    :param min_samples_threshold: 定义“核心专家”所需的最小样本数。
    :param jaccard_threshold: 用于合并的最小Jaccard相似度阈值。
    """
    print("开始执行Field Block合并与剪枝流程...")
    os.makedirs(output_dir, exist_ok=True)

    # --- 步骤一：统计与识别 ---
    print(f"\n[步骤 1/5] 正在扫描并分析所有 {len(os.listdir(source_dir))} 个原始Block...")
    
    block_metadata: List[Dict[str, Any]] = []
    for filename in tqdm(os.listdir(source_dir), desc="Scanning blocks"):
        if not filename.endswith('.csv'):
            continue
        
        block_path = os.path.join(source_dir, filename)
        try:
            df = pd.read_csv(block_path, dtype=str)
            # 确保'index'和'label'列存在
            if 'index' not in df.columns or 'label' not in df.columns:
                print(f"警告: Block {filename} 缺少 'index' 或 'label' 列，已跳过。")
                continue

            feature_columns = set(df.columns) - {'index', 'label', 'label_id'}
            
            block_metadata.append({
                "name": os.path.splitext(filename)[0],
                "path": block_path,
                "samples": len(df),
                "schema": feature_columns,
                "data": df
            })
        except Exception as e:
            print(f"读取或分析 {filename} 时出错: {e}")

    # --- 步骤二：确定“核心专家”和“待合并专家” ---
    core_blocks = [b for b in block_metadata if b['samples'] >= min_samples_threshold]
    small_blocks = [b for b in block_metadata if b['samples'] < min_samples_threshold]
    
    print(f"\n[步骤 2/5] 识别完成:")
    print(f" - {len(core_blocks)} 个核心专家 (样本数 >= {min_samples_threshold})")
    print(f" - {len(small_blocks)} 个待合并的小型专家")

    if not core_blocks:
        print("错误：找不到任何核心专家。请尝试降低min_samples_threshold。")
        return

    # --- 步骤三：执行合并逻辑 ---
    print("\n[步骤 3/5] 正在为小型专家寻找最佳合并目标...")
    
    # 使用字典来存储最终的、合并后的DataFrame
    merged_data: Dict[str, pd.DataFrame] = {core['name']: core['data'] for core in core_blocks}
    merge_log: Dict[str, str] = {core['name']: 'Core expert' for core in core_blocks}
    orphan_blocks: List[Dict[str, Any]] = []

    for s_block in tqdm(small_blocks, desc="Merging small blocks"):
        best_target_name = None
        
        # --- 优先规则1：寻找“超集”关系 ---
        superset_candidates = []
        for c_block in core_blocks:
            if s_block['schema'].issubset(c_block['schema']):
                # 计算超集的大小差异，我们想要最“贴身”的那个
                schema_diff = len(c_block['schema'] - s_block['schema'])
                superset_candidates.append((c_block['name'], schema_diff))
        
        if superset_candidates:
            # 按差异从小到大排序，选择差异最小的那个
            superset_candidates.sort(key=lambda x: x[1])
            best_target_name = superset_candidates[0][0]
        else:
            # --- 优先规则2：寻找Jaccard相似度最高的伙伴 ---
            jaccard_scores = []
            for c_block in core_blocks:
                similarity = jaccard_similarity(s_block['schema'], c_block['schema'])
                if similarity >= jaccard_threshold:
                    jaccard_scores.append((c_block['name'], similarity))
            
            if jaccard_scores:
                # 按相似度从高到低排序，选择最高的那个
                jaccard_scores.sort(key=lambda x: x[1], reverse=True)
                best_target_name = jaccard_scores[0][0]

        # --- 执行数据合并 ---
        if best_target_name:
            target_df = merged_data[best_target_name]
            small_df = s_block['data']
            
            # 找出需要填充的缺失列
            missing_columns = set(target_df.columns) - set(small_df.columns)
            
            # 填充缺失列 (Padding)
            for col in missing_columns:
                # 用一个在数据中不可能出现的值（如-1）或0来填充
                # 因为我们是dtype=str，所以填充字符串'0'
                small_df[col] = '0'
            
            # 确保列顺序一致，然后拼接
            small_df = small_df[target_df.columns]
            merged_data[best_target_name] = pd.concat([target_df, small_df], ignore_index=True)
            merge_log[s_block['name']] = f"Merged into {best_target_name}"
        else:
            # 如果找不到任何合适的合并目标，则标记为“孤儿”
            orphan_blocks.append(s_block)
            merge_log[s_block['name']] = "Orphaned"

    # --- 步骤四：处理“孤儿”，创建“默认专家” ---
    print(f"\n[步骤 4/5] 正在处理 {len(orphan_blocks)} 个“孤儿”Block...")
    if orphan_blocks:
        default_expert_name = "default_expert"
        
        # 将所有孤儿的数据拼接在一起，pandas会自动处理列的并集并用NaN填充
        orphan_dfs = [b['data'] for b in orphan_blocks]
        default_df = pd.concat(orphan_dfs, ignore_index=True, sort=False)
        
        # 用'0'填充所有新产生的NaN值
        default_df.fillna('0', inplace=True)
        
        merged_data[default_expert_name] = default_df
        merge_log[default_expert_name] = f"Created from {len(orphan_blocks)} orphan blocks"

    # --- 步骤五：保存所有最终的Block ---
    print(f"\n[步骤 5/5] 正在保存 {len(merged_data)} 个最终的Block...")
    for final_name, final_df in tqdm(merged_data.items(), desc="Saving final blocks"):
        output_path = os.path.join(output_dir, f"{final_name}.csv")
        final_df.to_csv(output_path, index=False)
        
    # 保存合并日志，以便追溯
    log_path = os.path.join(output_dir, "merge_log.json")
    with open(log_path, 'w') as f:
        json.dump(merge_log, f, indent=4)
        
    print(f"\n合并与剪枝成功！最终生成了 {len(merged_data)} 个专家Block。")
    print(f"合并日志已保存到: {log_path}")

def extract_nodes_and_edges(protocol_tree: Dict[str, List[str]]) -> (Set[str], Set[Tuple[str, str]]): # type: ignore
    """
    从protocol_tree字典中，提取出所有的节点集合和边集合。
    """
    node_set = set()
    edge_set = set()
    
    for parent, children in protocol_tree.items():
        # 父节点本身也是一个节点
        node_set.add(parent)
        for child in children:
            # 所有的子节点也是节点
            node_set.add(child)
            # 添加一条从父到子的边 (元组形式)
            edge_set.add((parent, child))
            
    return node_set, edge_set

def calculate_tree_similarity(tree1: Dict, tree2: Dict, alpha: float = 0.5) -> float:
    """
    计算两棵协议树的“结构化树相似度”。
    
    :param tree1: 第一个protocol_tree字典。
    :param tree2: 第二个protocol_tree字典。
    :param alpha: 节点相似度的权重。
    :return: 一个0到1之间的最终相似度分数。
    """
    # 1. 分别提取两棵树的节点集和边集
    nodes1, edges1 = extract_nodes_and_edges(tree1)
    nodes2, edges2 = extract_nodes_and_edges(tree2)
    
    # 2. 计算节点相似度
    node_sim = jaccard_similarity(nodes1, nodes2)
    
    # 3. 计算结构（边）相似度
    edge_sim = jaccard_similarity(edges1, edges2)
    
    # 4. 加权求和
    final_similarity = alpha * node_sim + (1 - alpha) * edge_sim
    
    return final_similarity, node_sim, edge_sim

def merge_field_blocks_tree_similarity(
    source_dir: str, 
    output_dir: str, 
    min_samples_threshold: int = 1000, 
    similarity_threshold: float = 0.7 # 使用一个更合理的结构相似度阈值
):
    """
    根据【结构化树相似度】，智能地合并和剪枝Field Blocks。
    """
    print("开始执行基于【结构化树相似度】的Field Block合并与剪枝流程...")
    os.makedirs(output_dir, exist_ok=True)

    # --- 步骤一：扫描所有Block，提取元数据（包括protocol_tree） ---
    print(f"\n[步骤 1/5] 正在扫描并分析所有原始Block...")
    
    block_metadata: List[Dict[str, Any]] = []
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.csv')]
    for filename in tqdm(all_files, desc="Scanning blocks"):
        block_path = os.path.join(source_dir, filename)
        try:
            # 只读表头以获取schema，提高扫描速度
            df_header = pd.read_csv(block_path, dtype=str, nrows=0)
            num_samples = sum(1 for row in open(block_path)) - 1 # 快速计算行数
            
            if 'label' not in df_header.columns: continue

            feature_columns = [col for col in df_header.columns if col not in ['index', 'label', 'label_id']]
            
            # 为每个Block生成其协议树
            protocol_tree, _ = generate_protocol_tree_and_nodes(feature_columns)
            
            block_metadata.append({
                "name": os.path.splitext(filename)[0],
                "path": block_path,
                "samples": num_samples,
                "schema": set(feature_columns),
                "ptree": protocol_tree # <-- 存储每个Block的协议树结构
            })
        except Exception as e:
            print(f"读取或分析 {filename} 时出错: {e}")

    # --- 步骤二：确定“核心专家”和“待合并专家” ---
    core_blocks = [b for b in block_metadata if b['samples'] >= min_samples_threshold]
    small_blocks = [b for b in block_metadata if b['samples'] < min_samples_threshold]
    
    print(f"\n[步骤 2/5] 识别完成:")
    print(f" - {len(core_blocks)} 个核心专家 (样本数 >= {min_samples_threshold})")
    print(f" - {len(small_blocks)} 个待合并的小型专家")

    if not core_blocks:
        print("错误：找不到任何核心专家。请尝试降低min_samples_threshold。")
        return

    # --- 步骤三：执行合并逻辑 ---
    print("\n[步骤 3/5] 正在为小型专家寻找最佳合并目标...")
    
    # 读取所有核心Block的数据到内存
    merged_data: Dict[str, pd.DataFrame] = {
        core['name']: pd.read_csv(core['path'], dtype=str) for core in core_blocks
    }
    merge_log: Dict[str, str] = {core['name']: 'Core expert' for core in core_blocks}
    orphan_blocks: List[Dict[str, Any]] = []

    for s_block in tqdm(small_blocks, desc="Merging small blocks"):
        best_target_name = None
        highest_similarity = -1
        
        # --- 使用新的“结构化树相似度”来寻找最佳伙伴 ---
        scores = []
        for c_block in core_blocks:
            similarity, _, _ = calculate_tree_similarity(s_block['ptree'], c_block['ptree'])
            scores.append((c_block['name'], similarity))
        
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            if scores[0][1] >= similarity_threshold:
                best_target_name = scores[0][0]

        # --- 执行数据合并 ---
        if best_target_name:
            target_df = merged_data[best_target_name]
            small_df = pd.read_csv(s_block['path'], dtype=str)
            
            missing_columns = set(target_df.columns) - set(small_df.columns)
            
            for col in missing_columns:
                if col not in ['index', 'label', 'label_id']:
                    small_df[col] = '0'
            
            # 删除多余的列，并确保列顺序一致
            extra_columns = set(small_df.columns) - set(target_df.columns)
            small_df.drop(columns=list(extra_columns), inplace=True, errors='ignore')
            small_df = small_df[target_df.columns]
            
            merged_data[best_target_name] = pd.concat([target_df, small_df], ignore_index=True)
            merge_log[s_block['name']] = f"Merged into {best_target_name} (Similarity: {scores[0][1]:.4f})"
        else:
            orphan_blocks.append(s_block)
            merge_log[s_block['name']] = "Orphaned (Low similarity to all cores)"

    # --- 步骤四：处理“孤儿”，创建“默认专家” ---
    print(f"\n[步骤 4/5] 正在处理 {len(orphan_blocks)} 个“孤儿”Block...")
    if orphan_blocks:
        default_expert_name = "default_expert"
        orphan_dfs = [pd.read_csv(b['path'], dtype=str) for b in orphan_blocks]
        default_df = pd.concat(orphan_dfs, ignore_index=True, sort=False)
        default_df.fillna('0', inplace=True)
        merged_data[default_expert_name] = default_df
        merge_log[default_expert_name] = f"Created from {len(orphan_blocks)} orphan blocks"

    # --- 步骤五：保存所有最终的Block ---
    print(f"\n[步骤 5/5] 正在保存 {len(merged_data)} 个最终的Block...")
    for final_name, final_df in tqdm(merged_data.items(), desc="Saving final blocks"):
        output_path = os.path.join(output_dir, f"{final_name}.csv")
        final_df.to_csv(output_path, index=False)
        
    log_path = os.path.join(output_dir, "merge_log.json")
    with open(log_path, 'w') as f:
        json.dump(merge_log, f, indent=4)
        
    print(f"\n合并与剪枝成功！最终生成了 {len(merged_data)} 个专家Block。")
    print(f"合并日志已保存到: {log_path}")
