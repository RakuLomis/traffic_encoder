from typing import Dict, Tuple
import torch
import torch.optim as optim 
import torch.nn as nn 
from tqdm import tqdm 
from utils.data_loader import TrafficDataset
from torch.utils.data import Dataset
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention 
# from models.PTA_rebuild import ProtocolTreeAttention
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from torch.profiler import profile, record_function, ProfilerActivity
from utils.data_loader import custom_collate_fn
from models.MoEPTA import MoEPTA
# from utils.data_loader_gnn import GNNTrafficDataset, gnn_collate_fn
# from utils.data_loader_ptga import GNNTrafficDataset
from utils.data_loader_ptga_le import GNNTrafficDataset
from torch_geometric.loader import DataLoader
from models.ProtocolTreeGAttention import ProtocolTreeGAttention
from models.ProtocolTreeGAttention_le import HierarchicalMoE
from utils.metrics import calculate_metrics
from utils.model_utils import diagnose_gate_weights_for_class
import sys
from transformers import get_linear_schedule_with_warmup
from utils.loss_functions import FocalLoss
import numpy as np
import random 
from torch.optim import RAdam
import copy
import gc

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

TORCH_LONG_MAX = torch.iinfo(torch.long).max
def robust_hex_to_int(x):
    """将十六进制字符串安全转换为整数，处理NaN和错误。"""
    if not pd.notna(x): return 0
    try:
        # 确保是字符串, 去掉可能的'.0' (如果被误读为float)
        val_str = str(x).split('.')[0]
        # 假定'0x'前缀可能存在也可能不存在
        val_str = val_str.lower().replace('0x','')
        if not val_str: return 0 # 空字符串
        return min(int(val_str, 16), TORCH_LONG_MAX)
    except ValueError:
        return 0 # 无法转换 (例如, "eth.dst.lg")
    
def robust_timestamp_to_tsval(x):
    """
    从 'tcp.options.timestamp' 的完整十六进制字符串中
    提取 TSval，并将其转换为十进制整数。
    格式: 080a[TSval:8_chars][TSecr:8_chars]
    """
    if not pd.notna(x): return 0
    try:
        # 1. 清理字符串 (移除 '0x'，转小写)
        s = str(x).lower().replace('0x', '')
        
        # 2. 验证格式 (Kind=08, Len=0a)
        #    总长度必须是 20 个十六进制字符
        if len(s) != 20 or not s.startswith('080a'):
            return 0 # 这不是一个标准的时间戳选项
            
        # 3. 提取 TSval (第4到第12个字符, 即索引4到11)
        tsval_hex = s[4:12]
        
        # 4. 转换为十进制整数
        return int(tsval_hex, 16)
    except (ValueError, TypeError):
        return 0 # 转换失败

def set_seed(seed_value: int):
    """
    为了可复现性，设置一个全局种子。
    """
    print(f"Setting global seed to {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # 适用于多GPU
        
        # 【注意】下面这两行是实现完全可复现的关键，但可能会牺牲一些性能
        # 1. 禁用cudnn的benchmark，它会为不同的输入大小选择不同的（可能非确定的）算法
        torch.backends.cudnn.benchmark = False
        # 2. 强制cudnn使用确定的算法
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """
    为DataLoader的子进程设置种子，以确保数据加载的可复现性。
    """
    # 获取主进程中设置的torch种子的一个“偏移量”
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    num_classes: int, 
    # dynamic_weights: torch.Tensor,  # <-- 动态权重
    loss_fn: nn.Module, # for focal_loss
    alpha: float = 1e-4               # <-- 掩码的正则化系数
) -> (Dict, torch.Tensor): # type: ignore 
    """
    【终极架构版】
    为“分层语义MoE”模型（HierarchicalMoE）定制的训练函数。
    它处理“图字典”的批处理，并计算所有专家掩码的总正则化损失。
    """
    model.train()
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    # 【关键】在函数内部创建最基础的损失函数
    # reduction='none' 意味着它会为batch中的每个样本都返回一个损失值
    # base_loss_fn = nn.CrossEntropyLoss(reduction='none')

    for i, batch_dict in enumerate(tqdm(dataloader, desc="Training")):
        # # 1. 【新】将整个“批处理字典”移动到GPU
        # batch_dict = batch_dict.to(device)

        # ==================== 核心修改点：分别移动值 ====================
        #
        # 1. 【新】遍历字典，将【每一个值】（图和张量）分别移动到GPU
        #
        try:
            for key, value in batch_dict.items():
                if hasattr(value, 'to'): # 检查这个值是否有.to方法
                    batch_dict[key] = value.to(device)
        except Exception as e:
             # 这是一个备用安全网
             print(f"警告: 无法将批处理项 {key} 移动到 device. 错误: {e}")
        
        # 2. 【新】从一个“基础”图（例如'eth'）中获取标签
        #    (因为所有子图的'y'都是一样的)
        labels = batch_dict['eth'].y 

        # ==================== 【!! 核心修改：屏蔽SNI !!】 ====================
        # 定义你想要屏蔽的“作弊”特征
        FIELD_TO_IGNORE = 'tls.handshake.extensions_server_name'
        
        # 检查 'tls_handshake' 专家是否在批次中，并且该批次是否具有该属性
        if 'tls_handshake' in batch_dict and hasattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE):
            try:
                # 从 PyG 的 Batch 对象上删除该属性
                delattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE)
            except AttributeError:
                pass # 以防万一，虽然 hasattr 已经检查过了
        # =====================================================================
        
        # 3. 【新】模型现在接收字典，并返回 logits 和 门控字典
        #    outputs = logits
        #    gates_dict = {'eth': gate_tensor, 'ip': gate_tensor, ...}
        outputs, gates_dict = model(batch_dict)
        
        # 4. 计算【基础】分类损失 (形状: [B])
        # classification_loss_per_sample = base_loss_fn(outputs, labels)

        # # 5. 【核心】应用动态权重
        # #    根据每个样本的真实标签，从 dynamic_weights 中获取对应的权重
        # sample_weights = dynamic_weights[labels]
        # #    计算加权后的批次平均损失
        # classification_loss = (classification_loss_per_sample * sample_weights).mean()
        classification_loss = loss_fn(outputs, labels)
        
        # 6. 【新】计算所有专家掩码的总正则化损失
        total_mask_entropy_loss = 0.0
        num_experts_with_gate = len(gates_dict) # 假设所有专家都有门控
        
        if num_experts_with_gate > 0:
            for name, gate in gates_dict.items():
                # gate 是这个专家的 [num_nodes] 维度的门控权重
                total_mask_entropy_loss += -(gate * torch.log(gate + 1e-8) + 
                                             (1 - gate) * torch.log(1 - gate + 1e-8)).mean()
            
            # 对所有专家的损失取平均
            total_mask_entropy_loss = total_mask_entropy_loss / num_experts_with_gate
        
        # 7. 计算总损失
        total_loss = classification_loss + alpha * total_mask_entropy_loss
        
        # 8. 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 9. 梯度裁剪 (保持)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 10. 优化器步进
        optimizer.step()
        
        # 11. 【注意】调度器(scheduler)不在这里调用。
        #     它将在主训练循环的epoch末尾，
        #     根据 evaluate 的结果被调用 (例如 scheduler.step(val_f1_macro))

        # 12. 统计指标
        #    【新】使用 labels.size(0) 作为批次大小
        running_loss += classification_loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        
        # 累积结果到混淆矩阵
        labels_cpu = labels.cpu()
        predicted_cpu = predicted.cpu()
        for t, p in zip(labels_cpu.view(-1), predicted_cpu.view(-1)):
            if t < num_classes and p < num_classes:
                confusion_matrix[t.long(), p.long()] += 1

    # 在epoch结束后，进行一次性的性能计算
    total_samples = confusion_matrix.sum().item()
    epoch_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    
    epoch_metrics = calculate_metrics(confusion_matrix)
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics, confusion_matrix


@torch.no_grad() # 这是一个装饰器，它会自动为函数内的所有操作禁用梯度计算
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    num_classes: int, 
    loss_fn: nn.Module
) -> Tuple[Dict, torch.Tensor]:  #, torch.Tensor]:
    """
    【终极架构版】
    为“分层语义MoE”模型（HierarchicalMoE）定制的评估函数。
    它处理“图字典”的批处理，并返回详细指标以及“每类F1分数”。
    """
    model.eval() # 将模型设置为评估模式
    
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    
    # 我们在内部创建一个简单的损失函数，只用于【报告】损失值
    # base_loss_fn = nn.CrossEntropyLoss()

    for batch_dict in tqdm(dataloader, desc="Evaluating"):
        # ==================== 核心修改点：分别移动值 ====================
        #
        # 1. 【新】遍历字典，将【每一个值】（图和张量）分别移动到GPU
        #
        try:
            for key, value in batch_dict.items():
                if hasattr(value, 'to'): # 检查这个值是否有.to方法
                    batch_dict[key] = value.to(device)
        except Exception as e:
             # 这是一个备用安全网
             print(f"警告: 无法将批处理项 {key} 移动到 device. 错误: {e}")
        
        # 2. 【新】从一个“基础”图（例如'eth'）中获取标签
        labels = batch_dict['eth'].y 

        # ==================== 【!! 核心修改：屏蔽SNI !!】 ====================
        # 定义你想要屏蔽的“作弊”特征
        FIELD_TO_IGNORE = 'tls.handshake.extensions_server_name'
        
        # 检查 'tls_handshake' 专家是否在批次中，并且该批次是否具有该属性
        if 'tls_handshake' in batch_dict and hasattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE):
            try:
                # 从 PyG 的 Batch 对象上删除该属性
                delattr(batch_dict['tls_handshake'], FIELD_TO_IGNORE)
            except AttributeError:
                pass # 以防万一，虽然 hasattr 已经检查过了
        # =====================================================================
        
        # 3. 【新】模型返回两个值，评估时我们只关心第一个（logits）
        outputs, _ = model(batch_dict) # 忽略 gates_dict
        
        # # 4. 计算并累积损失（仅用于报告）
        # loss = base_loss_fn(outputs, labels)
        # running_loss += loss.item() * labels.size(0) # 使用 labels.size(0) 作为批次大小
        # 4. 【修改】计算并累积损失
        loss = loss_fn(outputs, labels) # <-- 使用传入的 loss_fn
        running_loss += loss.item() * labels.size(0)
        
        # 5. 计算预测
        _, predicted = torch.max(outputs.data, 1)
        
        # 6. 【高效】在GPU上直接累积混淆矩阵
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            if t < num_classes and p < num_classes:
                confusion_matrix[t, p] += 1

    # --- 在epoch结束后，进行一次性的性能计算 ---
    
    # 1. 计算总损失
    epoch_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    
    # 2. 从混淆矩阵计算P/R/F1等指标
    #    【重要】将混淆矩阵移回CPU进行numpy/sklearn计算
    cm_cpu = confusion_matrix.cpu()
    epoch_metrics = calculate_metrics(cm_cpu) # 假设 calculate_metrics 接收一个 tensor
    epoch_metrics['loss'] = epoch_loss
    
    # # 3. 【关键新增】计算并返回“每类F1分数”张量
    # tp = confusion_matrix.diag()
    # fp = confusion_matrix.sum(dim=0) - tp
    # fn = confusion_matrix.sum(dim=1) - tp
    # epsilon = 1e-8 # 防止除以零
    
    # precision = tp / (tp + fp + epsilon)
    # recall = tp / (tp + fn + epsilon)
    
    # # per_class_f1 是一个在GPU/CPU上的张量，形状为 [num_classes]
    # per_class_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # 确保返回的是在CPU上的张量，以便主循环使用
    return epoch_metrics, cm_cpu, # per_class_f1.cpu()


# =====================================================================
if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_LEARNING_RATE = 1e-3
    DROPOUT_RATE = 0.5
    NUM_WORKERS = 4 
    GNN_INPUT_DIM = 32 
    GNN_HIDDEN_DIM = 128
    PATIENCE = 5
    DIAGNOSE = False
    stop_training = False

    USE_FLOW_FEATURES_THIS_RUN = True

    # FocalLoss的超参数
    FOCAL_GAMMA = 2.0 # 0.0 ~ 5.0, 2.0是一个经典的起始值

    ROLLBACK_PATIENCE = NUM_EPOCHS // 10
    MIN_LR_FOR_TRAINING = 1e-6
    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    # dataset_name = 'ISCX-VPN'
    # dataset_name = 'ISCX-TOR-Acctivity'
    dataset_name = 'ISCX-TOR-Application'
    # dataset_name = 'dataset_29_d1'
    root_path = os.path.join('..', 'TrafficData', 'datasets_csv_add2')
    val_test_dir = os.path.join(root_path, 'datasets_split', dataset_name) 
    train_dir = os.path.join(root_path, 'datasets_final')
    vocab_dir = os.path.join(root_path, 'categorical_vocabs')
    config_path = os.path.join('.', 'Data', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join(vocab_dir, dataset_name + '_vocabs.yaml') 
    res_path = os.path.join('..', 'Res')
    # train_set_name = dataset_name + '_chief_block_augmented'
    # train_set_name = dataset_name + '_chief_block_topk_augmented'
    train_set_name = 'train_set'
    val_set_name = 'validation_set' 
    test_set_name = 'test_set'
    # chief_directory = train_dir
    chief_directory = val_test_dir
    val_test_directory = val_test_dir

    train_df_path = os.path.join(chief_directory, train_set_name + '.csv') 
    val_df_path = os.path.join(val_test_directory, val_set_name + '.csv')
    test_df_path = os.path.join(val_test_directory, test_set_name + '.csv')
    
    # --- 3. 加载并对齐数据集 ---
    print("\n[1/4] Loading datasets...")
    try:
        train_df = pd.read_csv(train_df_path, dtype=str)
        val_df = pd.read_csv(val_df_path, dtype=str)
        test_df = pd.read_csv(test_df_path, dtype=str)
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到，请确保您已完成预处理步骤。 {e}")
        exit()
        
    print(f" - Train set (augmented): {len(train_df)} rows")
    print(f" - Validation set: {len(val_df)} rows")
    print(f" - Test set: {len(test_df)} rows")

    # ==================== 代码优化：高效对齐 ====================
    print("\n[2/4] Aligning feature space for validation and test sets...")
    chief_schema = [col for col in train_df.columns if col not in ['label', 'label_id']]
    
    # 使用 reindex + fillna，一步到位，性能更高
    val_df_aligned = val_df.reindex(columns=chief_schema, fill_value='0')
    val_df_aligned['label'] = val_df['label']
    
    test_df_aligned = test_df.reindex(columns=chief_schema, fill_value='0')
    test_df_aligned['label'] = test_df['label']
    
    print(" - Feature alignment complete.")
    # ==============================================================
    del val_df, test_df

    if USE_FLOW_FEATURES_THIS_RUN:
        print("\n[2.5/4] Performing Feature Engineering 3.0 (Normalized Ratios)...")

        # a) 【!! 核心修改：使用比率特征 !!】
        flow_feature_names = [
            'flow_avg_len', 'flow_std_len', 'flow_pkt_count', # (pkt_count 仍然有用)
            'flow_avg_iat', 'flow_std_iat', 'flow_max_iat',
            'flow_duration_per_pkt',  # <-- 【新】替换 'flow_duration'
            'flow_large_pkt_ratio'    # <-- 【新】替换 'flow_large_pkt_count'
        ]
        
        # (epsilon，防止除以零)
        EPSILON = 1e-6 

        # b) (临时列的创建保持不变)
        print(" -> [Hex Fix] Converting key columns to temporary decimal columns...")
        for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
            # ... (df['ip.len_temp_dec'] = ...)
            # ... (df['tcp.stream_temp_dec'] = ...)
            # ... (df['time_temp_dec'] = ...)
            if df.empty: continue
            print(f"   -> Processing {df_name} set for temp columns...")
            df['ip.len_temp_dec'] = df['ip.len'].apply(robust_hex_to_int).astype(np.float32)
            df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
            if 'tcp.options.timestamp' not in df.columns:
                raise ValueError("错误：特征工程2.0需要 'tcp.options.timestamp' 列，但数据集中未找到。")
            df['time_temp_dec'] = df['tcp.options.timestamp'].apply(robust_timestamp_to_tsval).astype(np.int64) 
        print(" -> Temporary decimal columns created.")


        # c) 【!! 核心修改：定义 *函数* 来计算所有统计数据 !!】
        def calculate_flow_stats(df, is_train_set=False):
            print(f"   -> Calculating stats for DataFrame (size: {len(df)})...")
            
            # --- 1. 计算 IAT ---
            print("     -> Calculating IAT...")
            df = df.sort_values(by=['tcp.stream_temp_dec', 'time_temp_dec'])
            df['iat_temp'] = df.groupby('tcp.stream_temp_dec')['time_temp_dec'].diff().fillna(0)
            
            # --- 2. 计算所有 *基础* 统计数据 ---
            print("     -> Grouping base statistics...")
            grouped = df.groupby('tcp.stream_temp_dec')
            
            maps = {}
            maps['flow_avg_len'] = grouped['ip.len_temp_dec'].mean()
            maps['flow_std_len'] = grouped['ip.len_temp_dec'].std().fillna(0)
            maps['flow_pkt_count'] = grouped['ip.len_temp_dec'].count()
            maps['flow_avg_iat'] = grouped['iat_temp'].mean()
            maps['flow_std_iat'] = grouped['iat_temp'].std().fillna(0)
            maps['flow_max_iat'] = grouped['iat_temp'].max()
            
            flow_min_time = grouped['time_temp_dec'].min()
            flow_max_time = grouped['time_temp_dec'].max()
            # (基础特征：Duration)
            base_duration = (flow_max_time - flow_min_time)
            
            LARGE_PKT_THRESHOLD = 1400 
            large_pkts = df[df['ip.len_temp_dec'] > LARGE_PKT_THRESHOLD]
            # (基础特征：Large Pkt Count)
            base_large_pkt_count = large_pkts.groupby('tcp.stream_temp_dec').size().reindex(maps['flow_pkt_count'].index, fill_value=0)

            # --- 3. 【!! 新：计算比率特征 !!】 ---
            print("     -> Calculating ratio features...")
            maps['flow_duration_per_pkt'] = base_duration / (maps['flow_pkt_count'] + EPSILON)
            maps['flow_large_pkt_ratio'] = base_large_pkt_count / (maps['flow_pkt_count'] + EPSILON)
            
            if is_train_set:
                # --- 4. 计算全局默认值 ---
                print("     -> Calculating global defaults...")
                defaults = {}
                for f_name in flow_feature_names:
                    if 'count' in f_name:
                        defaults[f_name] = int(maps[f_name].mean())
                    else:
                        defaults[f_name] = float(maps[f_name].mean())
                    # 清理 defaults 中的 nan/inf
                    if np.isnan(defaults[f_name]) or np.isinf(defaults[f_name]): defaults[f_name] = 0
                return maps, defaults
            else:
                return maps
        
        # d) (使用新函数)
        print(" -> Learning per-flow statistics ONLY from Train set...")
        train_maps, global_defaults = calculate_flow_stats(train_df, is_train_set=True)
        print(f" -> Learned global defaults (Example 'flow_avg_len'): {global_defaults.get('flow_avg_len', 0.0):.2f}")

        # e) 【!! 核心修改：应用 if/else 逻辑 !!】
        
        # 【!! 在这里设置你的实验 !!】
        # OPEN_WORLD = True
        OPEN_WORLD = False  # <-- 【!! 保持 False (诊断实验) !!】

        if OPEN_WORLD:
            print(" -> [OPEN_WORLD MODE] Applying stats learned from Train set to ALL sets...")
            for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
                # ... (这个逻辑是正确的，但我们知道它会导致 F1=0.89)
                if df.empty: continue
                print(f"   -> Processing {df_name} set...")
                if 'tcp.stream_temp_dec' not in df.columns:
                    df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
                for f_name in flow_feature_names:
                    df[f_name] = df['tcp.stream_temp_dec'].map(train_maps[f_name]).fillna(global_defaults[f_name])
        else:
            # --- 方案B: 我们的“诊断性”实验 (有数据泄露/残缺) ---
            print(" -> [DIAGNOSTIC MODE (OPEN_WORLD=False)] Calculating stats directly from each set...")
            
            # 1. 首先处理 Train Set (使用它自己的map)
            print("   -> Processing Train set (self-mapping)...")
            for f_name in flow_feature_names:
                 train_df[f_name] = train_df['tcp.stream_temp_dec'].map(train_maps[f_name]).fillna(global_defaults[f_name])
                 
            # 2. 迭代并【就地修改】验证集和测试集
            for df_name, df in [('Validation', val_df_aligned), ('Test', test_df_aligned)]:
                if df.empty: continue
                print(f"   -> Calculating and applying stats directly from {df_name} set...")
                
                # a. 计算 *这个* df 的统计数据
                df_maps = calculate_flow_stats(df, is_train_set=False)

                # b. 将【自身】的统计数据map回自身
                for f_name in flow_feature_names:
                    df[f_name] = df['tcp.stream_temp_dec'].map(df_maps[f_name]).fillna(0) # 缺失的流用0填充

        # f) 【!! 关键清理 !!】删除所有临时列 (不变)
        print(" -> Cleaning up temporary decimal columns...")
        # ... (df.drop(...) 的逻辑不变) ...
        for df in [train_df, val_df_aligned, test_df_aligned]:
            if 'ip.len_temp_dec' in df.columns: df.drop(columns=['ip.len_temp_dec'], inplace=True)
            if 'tcp.stream_temp_dec' in df.columns: df.drop(columns=['tcp.stream_temp_dec'], inplace=True)
            if 'time_temp_dec' in df.columns: df.drop(columns=['time_temp_dec'], inplace=True)
            if 'iat_temp' in df.columns: df.drop(columns=['iat_temp'], inplace=True)
        
        # g) 【最终清理】(不变, 使用 np.nan_to_num)
        print(" -> Final cleanup of *new* flow features (nan/inf/neginf -> 0.0)...")
        # ... (for df in ... for col in flow_feature_names: ... df[col] = ... 的逻辑不变) ...
        for df in [train_df, val_df_aligned, test_df_aligned]:
             if df.empty: continue
             for col in flow_feature_names: 
                 if col not in df.columns: raise KeyError(f"列 '{col}' 未在 {df_name} 中找到！")
                 col_data_numeric = pd.to_numeric(df[col], errors='coerce')
                 col_data_np = col_data_numeric.values
                 col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0) 
                 if 'count' in col: df[col] = col_data_cleaned.astype(np.int32)
                 else: df[col] = col_data_cleaned.astype(np.float32)
                     
        print(f" -> Feature Engineering 3.0 complete. {len(flow_feature_names)} flow features created.")

    # if USE_FLOW_FEATURES_THIS_RUN: 
    #     # ==================== 核心修改点：流统计特征工程 ====================
    #     print("\n[2.5/4] Performing Flow-level Statistics Engineering...") 

    #     # 这是一个开关，决定了我们是模拟“现实世界”（True）还是进行“理想实验”（False）
    #     OPEN_WORLD = False 

    #     # a) 定义流特征名称
    #     flow_feature_names = ['flow_avg_len', 'flow_std_len', 'flow_pkt_count', 
    #                           'flow_avg_iat', 'flow_std_iat', 'flow_max_iat',
    #                             'flow_duration',
    #                             'flow_large_pkt_count']
    #     # b) 【!! 核心修改 !!】
    #     print(" -> [Hex Fix] Converting key columns to temporary decimal columns...")
        
    #     # 【!! 修正 !!】我们 *只* 需要在 train_df 上进行转换来 *学习*
    #     #     val/test 只需要在 *应用* 知识时转换 tcp.stream
    #     print(f"   -> Processing Train set for temp columns...")
    #     train_df['ip.len_temp_dec'] = train_df['ip.len'].apply(robust_hex_to_int).astype(np.float32)
    #     train_df['tcp.stream_temp_dec'] = train_df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
    #     if 'tcp.options.timestamp' not in train_df.columns:
    #         raise ValueError("错误：特征工程2.0需要 'tcp.options.timestamp' 列，但数据集中未找到。")
    #     train_df['time_temp_dec'] = train_df['tcp.options.timestamp'].apply(robust_timestamp_to_tsval).astype(np.int64) 
    #     print(" -> Temporary decimal columns created for train_df.")
        
    #     # for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
    #     #     if df.empty: continue
    #     #     print(f"   -> Processing {df_name} set for temp columns...")
            
    #     #     # 1. 转换 ip.len (不变)
    #     #     df['ip.len_temp_dec'] = df['ip.len'].apply(robust_hex_to_int).astype(np.float32)
    #     #     # 2. 转换 tcp.stream (不变)
    #     #     df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
            
    #     #     # 3. 【!! 修改 !!】转换 timestamp
    #     #     if 'tcp.options.timestamp' not in df.columns: # <-- 检查新列
    #     #         raise ValueError("错误：特征工程2.0需要 'tcp.options.timestamp' 列，但数据集中未找到。")
            
    #     #     # 使用 *新* 函数转换 *新* 列，并使用 int64 (因为时间戳会很大)
    #     #     df['time_temp_dec'] = df['tcp.options.timestamp'].apply(robust_timestamp_to_tsval).astype(np.int64) 

    #     # print(" -> Temporary decimal columns created.")

    #     # c) 【!! 核心修改：从 *临时* 列中学习 *新* 知识 !!】
    #     print(" -> Learning per-flow statistics ONLY from Train set (using temp decimal values)...")
        
    #     # --- (旧特征) --- (不变)
    #     train_flow_avg_len = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].mean()
    #     train_flow_std_len = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].std().fillna(0)
    #     train_flow_pkt_count = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].count()
        
    #     # --- (新特征 1: IAT) ---
    #     print("   -> Calculating IAT (this may take a moment)...")
    #     # (我们必须先按流和时间排序，以确保 diff() 是正确的)
    #     # (我们假设 TSval (time_temp_dec) 已经是单调递增的，但排序更安全)
    #     train_df = train_df.sort_values(by=['tcp.stream_temp_dec', 'time_temp_dec'])
        
    #     # 【!! 修改 !!】计算 TSval 的差值
    #     #    (注意：如果TSval回绕(wrap-around)，diff()可能会产生巨大负数，
    #     #     我们稍后用 np.nan_to_num(..., neginf=0) 来清理)
    #     train_df['iat_temp'] = train_df.groupby('tcp.stream_temp_dec')['time_temp_dec'].diff().fillna(0)
        
    #     # (计算 IAT 统计数据)
    #     train_flow_avg_iat = train_df.groupby('tcp.stream_temp_dec')['iat_temp'].mean()
    #     train_flow_std_iat = train_df.groupby('tcp.stream_temp_dec')['iat_temp'].std().fillna(0)
    #     train_flow_max_iat = train_df.groupby('tcp.stream_temp_dec')['iat_temp'].max()

    #     # --- (新特征 2: Duration) ---
    #     # (计算 TSval 的总 "持续时间")
    #     flow_min_time = train_df.groupby('tcp.stream_temp_dec')['time_temp_dec'].min()
    #     flow_max_time = train_df.groupby('tcp.stream_temp_dec')['time_temp_dec'].max()
    #     train_flow_duration = (flow_max_time - flow_min_time)
        
    #     # --- (新特征 3: Bins) --- (不变)
    #     LARGE_PKT_THRESHOLD = 1400 
    #     large_pkts = train_df[train_df['ip.len_temp_dec'] > LARGE_PKT_THRESHOLD]
    #     train_flow_large_pkt_count = large_pkts.groupby('tcp.stream_temp_dec').size()

    #     # d) 【!! 核心修改：计算 *所有* 全局默认值 !!】
    #     print(" -> Learning global defaults for all features...")
    #     g_avg_len = float(train_flow_avg_len.mean())
    #     g_std_len = float(train_flow_std_len.mean())
    #     g_pkt_count = int(train_flow_pkt_count.mean())
    #     g_avg_iat = float(train_flow_avg_iat.mean())
    #     g_std_iat = float(train_flow_std_iat.mean())
    #     g_max_iat = float(train_flow_max_iat.mean())
    #     g_duration = float(train_flow_duration.mean())
    #     g_large_pkt_count = 0 
        
    #     global_defaults = {
    #         'flow_avg_len': np.nan_to_num(g_avg_len, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_std_len': np.nan_to_num(g_std_len, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_pkt_count': np.nan_to_num(g_pkt_count, nan=0),
    #         'flow_avg_iat': np.nan_to_num(g_avg_iat, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_std_iat': np.nan_to_num(g_std_iat, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_max_iat': np.nan_to_num(g_max_iat, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_duration': np.nan_to_num(g_duration, nan=0.0, posinf=0.0, neginf=0.0),
    #         'flow_large_pkt_count': g_large_pkt_count 
    #     }
    #     train_maps = {
    #         'flow_avg_len': train_flow_avg_len,
    #         'flow_std_len': train_flow_std_len,
    #         'flow_pkt_count': train_flow_pkt_count,
    #         'flow_avg_iat': train_flow_avg_iat,
    #         'flow_std_iat': train_flow_std_iat,
    #         'flow_max_iat': train_flow_max_iat,
    #         'flow_duration': train_flow_duration,
    #         'flow_large_pkt_count': train_flow_large_pkt_count
    #     }

    #     if OPEN_WORLD: 
    #         # print(" -> [OPEN_WORLD MODE] Applying all stats to Val/Test sets...")
    #         # for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
    #         #     if df.empty: continue
    #         #     print(f"   -> Processing {df_name} set...")
    #         #     for f_name in flow_feature_names:
    #         #         df[f_name] = df['tcp.stream_temp_dec'].map(train_maps[f_name]).fillna(global_defaults[f_name])
    #         print(" -> [OPEN_WORLD MODE] Applying stats learned from Train set to ALL sets...")
    #         for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
    #             if df.empty: continue
    #             print(f"   -> Processing {df_name} set...")
                
    #             # 为 val/test 创建临时 stream 列 (train_df 已经有了)
    #             if 'tcp.stream_temp_dec' not in df.columns:
    #                 df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)

    #             for f_name in flow_feature_names:
    #                 # 使用 .map() 并使用训练集的【全局默认值】填充在训练集中未见过的“新流”
    #                 df[f_name] = df['tcp.stream_temp_dec'].map(train_maps[f_name]).fillna(global_defaults[f_name])
    #     else:
    #         # --- 方案B: 我们的“诊断性”实验 (有数据泄露) ---
    #         print(" -> [DIAGNOSTIC MODE (OPEN_WORLD=False)] Calculating stats directly from each set...")
            
    #         # 1. 首先处理 Train Set (使用它自己的map)
    #         print("   -> Processing Train set (self-mapping)...")
    #         for f_name in flow_feature_names:
    #              train_df[f_name] = train_df['tcp.stream_temp_dec'].map(train_maps[f_name]).fillna(global_defaults[f_name])
                 
    #         # 2. 迭代并【就地修改】验证集和测试集
    #         for df_name, df in [('Validation', val_df_aligned), ('Test', test_df_aligned)]:
    #             if df.empty: continue
    #             print(f"   -> Calculating and applying stats directly from {df_name} set...")

    #             # a. 确保 val/test 也有临时列
    #             if 'ip.len_temp_dec' not in df.columns:
    #                  df['ip.len_temp_dec'] = df['ip.len'].apply(robust_hex_to_int).astype(np.float32)
    #             if 'tcp.stream_temp_dec' not in df.columns:
    #                  df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
    #             if 'time_temp_dec' not in df.columns:
    #                  df['time_temp_dec'] = df['tcp.options.timestamp'].apply(robust_timestamp_to_tsval).astype(np.int64) 

    #             # b. 【关键】计算此DataFrame【自身】的统计数据
    #             df_avg_len = df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].mean()
    #             df_std_len = df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].std().fillna(0)
    #             df_pkt_count = df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].count()
                
    #             df = df.sort_values(by=['tcp.stream_temp_dec', 'time_temp_dec'])
    #             df['iat_temp'] = df.groupby('tcp.stream_temp_dec')['time_temp_dec'].diff().fillna(0)
    #             df_avg_iat = df.groupby('tcp.stream_temp_dec')['iat_temp'].mean()
    #             df_std_iat = df.groupby('tcp.stream_temp_dec')['iat_temp'].std().fillna(0)
    #             df_max_iat = df.groupby('tcp.stream_temp_dec')['iat_temp'].max()

    #             df_min_time = df.groupby('tcp.stream_temp_dec')['time_temp_dec'].min()
    #             df_max_time = df.groupby('tcp.stream_temp_dec')['time_temp_dec'].max()
    #             df_duration = (df_max_time - df_min_time)
                
    #             df_large_pkts = df[df['ip.len_temp_dec'] > LARGE_PKT_THRESHOLD]
    #             df_large_pkt_count = df_large_pkts.groupby('tcp.stream_temp_dec').size()

    #             # c. 【关键】将【自身】的统计数据map回自身
    #             df['flow_avg_len'] = df['tcp.stream_temp_dec'].map(df_avg_len).fillna(0)
    #             df['flow_std_len'] = df['tcp.stream_temp_dec'].map(df_std_len).fillna(0)
    #             df['flow_pkt_count'] = df['tcp.stream_temp_dec'].map(df_pkt_count).fillna(0)
    #             df['flow_avg_iat'] = df['tcp.stream_temp_dec'].map(df_avg_iat).fillna(0)
    #             df['flow_std_iat'] = df['tcp.stream_temp_dec'].map(df_std_iat).fillna(0)
    #             df['flow_max_iat'] = df['tcp.stream_temp_dec'].map(df_max_iat).fillna(0)
    #             df['flow_duration'] = df['tcp.stream_temp_dec'].map(df_duration).fillna(0)
    #             df['flow_large_pkt_count'] = df['tcp.stream_temp_dec'].map(df_large_pkt_count).fillna(0)
    #     # f) 【!! 关键清理 !!】删除所有临时列
    #     print(" -> Cleaning up temporary decimal columns...")
    #     for df in [train_df, val_df_aligned, test_df_aligned]:
    #         if 'ip.len_temp_dec' in df.columns: df.drop(columns=['ip.len_temp_dec'], inplace=True)
    #         if 'tcp.stream_temp_dec' in df.columns: df.drop(columns=['tcp.stream_temp_dec'], inplace=True)
    #         if 'time_temp_dec' in df.columns: df.drop(columns=['time_temp_dec'], inplace=True)
    #         if 'iat_temp' in df.columns: df.drop(columns=['iat_temp'], inplace=True)
        
    #     # g) 【最终清理】(不变, 使用 np.nan_to_num)
    #     print(" -> Final cleanup of *new* flow features (nan/inf/neginf -> 0.0)...")
    #     for df in [train_df, val_df_aligned, test_df_aligned]:
    #          if df.empty: continue
    #          for col in flow_feature_names: 
    #              col_data_numeric = pd.to_numeric(df[col], errors='coerce')
    #              col_data_np = col_data_numeric.values
    #              # 【!! 修复 !!】确保 neginf (来自 IAT diff) 也被清理
    #              col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0) 
                 
    #              if 'count' in col:
    #                  df[col] = col_data_cleaned.astype(np.int32)
    #              else: # avg, std, iat, duration -> float
    #                  df[col] = col_data_cleaned.astype(np.float32)
                     
    #     print(f" -> Feature Engineering 2.0 complete. {len(flow_feature_names)} flow features created.")

        # # b) 【!! 关键修复：在 groupby 之前转换 Hex (使用临时列) !!】
        # print(" -> [Hex Fix] Converting 'ip.len' and 'tcp.stream' to temporary decimal columns for flow calc...")
        
        # # --- 1. 转换 (仅限 train_df，因为我们只从它学习) ---
        # #     我们创建 *新* 的临时列，而不是覆盖旧的
        # train_df['ip.len_temp_dec'] = train_df['ip.len'].apply(robust_hex_to_int).astype(np.float32)
        # train_df['tcp.stream_temp_dec'] = train_df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)
        # print(" -> Temporary decimal columns created in train_df.")

        # # c) 【关键】从 *临时* 列中学习“知识”
        # print(" -> Learning per-flow statistics ONLY from Train set (using temp decimal values)...")
        # # (现在 groupby 正在对 *真实* 的十进制包长进行操作)
        # train_flow_avg_len = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].mean()
        # train_flow_std_len = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].std().fillna(0)
        # train_flow_pkt_count = train_df.groupby('tcp.stream_temp_dec')['ip.len_temp_dec'].count()

        # # d) 【关键】计算“全局默认值” (从训练集中学到)
        # train_global_avg_len = float(train_flow_avg_len.mean())
        # train_global_std_len = float(train_flow_std_len.mean())
        # train_global_pkt_count = int(train_flow_pkt_count.mean())
        
        # # 【健壮性修复：检查全局值是否为 NaN/Inf】
        # if np.isnan(train_global_avg_len) or np.isinf(train_global_avg_len): train_global_avg_len = 0.0
        # if np.isnan(train_global_std_len) or np.isinf(train_global_std_len): train_global_std_len = 0.0
        # if np.isnan(train_global_pkt_count) or np.isinf(train_global_pkt_count): train_global_pkt_count = 0
        
        # print(f" -> Learned global defaults: AvgLen={train_global_avg_len:.2f}, StdLen={train_global_std_len:.2f}, PktCount={train_global_pkt_count:.2f}")
    

        # # e) 将“知识”应用（广播）回训练集
        # # print(" -> Applying learned stats back to Train set...")
        # # train_df['flow_avg_len'] = train_df['tcp.stream'].map(train_flow_avg_len)
        # # train_df['flow_std_len'] = train_df['tcp.stream'].map(train_flow_std_len)
        # # train_df['flow_pkt_count'] = train_df['tcp.stream'].map(train_flow_pkt_count)


        # if OPEN_WORLD:
        #     for df_name, df in [('Train', train_df), ('Validation', val_df_aligned), ('Test', test_df_aligned)]:
        #         if df.empty: continue
        #         print(f"   -> Processing {df_name} set...")

        #         # 【重要】我们 map 用的 key 必须是十进制的 tcp.stream
        #         # 我们需要为所有 df 创建临时 stream 列
        #         if 'tcp.stream_temp_dec' not in df.columns: # 避免重复转换 train_df
        #             df['tcp.stream_temp_dec'] = df['tcp.stream'].apply(robust_hex_to_int).astype(np.int32)

        #         # 使用 temp stream key 和全局默认值来map
        #         df['flow_avg_len'] = df['tcp.stream_temp_dec'].map(train_flow_avg_len).fillna(train_global_avg_len)
        #         df['flow_std_len'] = df['tcp.stream_temp_dec'].map(train_flow_std_len).fillna(train_global_std_len)
        #         df['flow_pkt_count'] = df['tcp.stream_temp_dec'].map(train_flow_pkt_count).fillna(train_global_pkt_count)

        #     # f) 【!! 关键清理 !!】删除所有临时列
        #     print(" -> Cleaning up temporary decimal columns...")
        #     for df in [train_df, val_df_aligned, test_df_aligned]:
        #         if 'ip.len_temp_dec' in df.columns:
        #             df.drop(columns=['ip.len_temp_dec'], inplace=True)
        #         if 'tcp.stream_temp_dec' in df.columns:
        #             df.drop(columns=['tcp.stream_temp_dec'], inplace=True)

        # # g) 【最终清理】确保所有新添加的 *流特征* 列是干净的 (nan/inf -> 0)
        # print(" -> Final cleanup of *new* flow features (nan/inf -> 0.0)...")
        # for df in [train_df, val_df_aligned, test_df_aligned]:
        #     if df.empty: continue
        #     for col in flow_feature_names: # 只清理我们刚添加的列
        #         col_data_numeric = pd.to_numeric(df[col], errors='coerce')
        #         col_data_np = col_data_numeric.values
        #         col_data_cleaned = np.nan_to_num(col_data_np, nan=0.0, posinf=0.0, neginf=0.0)
        #         if col == 'flow_pkt_count':
        #             df[col] = col_data_cleaned.astype(np.int32)
        #         else:
        #             df[col] = col_data_cleaned.astype(np.float32)

        print(" -> Flow-level features successfully engineered for all datasets.")


    # c) 创建全局标签映射
    #    为了确保所有数据集的标签一致，我们基于训练集来创建映射
    print("\n[3/4] Creating label mapping...")
    labels = train_df['label'].unique() # ?
    label_to_int = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    train_df['label_id'] = train_df['label'].map(label_to_int)
    val_df_aligned['label_id'] = val_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) # .fillna(-1)处理未见过的标签
    test_df_aligned['label_id'] = test_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) 
    

    # --- 4. 创建GNN Dataset和DataLoader ---
    print("\n[4/4] Creating GNN Datasets and DataLoaders...")
    
    # a) 实例化 GNNTrafficDataset
    train_dataset = GNNTrafficDataset(train_df, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN)
    val_dataset = GNNTrafficDataset(val_df_aligned, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN)
    test_dataset = GNNTrafficDataset(test_df_aligned, config_path, vocab_path, use_flow_features=USE_FLOW_FEATURES_THIS_RUN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ================== 【!! 核心修改：定义FocalLoss !!】 ==================
    
    print("Calculating static class weights (alpha) for FocalLoss...")
    # 1. 从训练集(train_df)中，计算每个类别的样本数
    class_counts = train_df['label_id'].value_counts().sort_index().values
    
    # 2. 计算权重：使用“总样本数 / (类别数 * 类别样本数)”
    #    这会给样本少的类别（如Google, Twitter）更高的权重
    class_weights = torch.tensor(class_counts, dtype=torch.float)
    total_samples = class_weights.sum()
    class_weights = total_samples / (num_classes * class_weights)
    
    # 3. 将权重tensor移动到GPU
    alpha_weights = class_weights.to(device)
    
    print(f" -> FocalLoss Alpha (weights): {alpha_weights.cpu().numpy().round(2)}")

    # 4. 实例化 FocalLoss，传入计算好的 alpha 和超参数 gamma
    loss_fn = FocalLoss(alpha=alpha_weights, gamma=FOCAL_GAMMA).to(device)

    # 5. 【删除】我们不再需要旧的 dynamic_weights
    # dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)
    
    # =====================================================================

    del train_df, val_df_aligned, test_df_aligned
    gc.collect()
    
    expert_graph_info = train_dataset.expert_graphs

    # b) 从训练数据集中获取模型需要的节点列表
    # node_fields_for_model = train_dataset.node_fields
    # print(f" - Model will be built for {len(node_fields_for_model)} nodes.")

    g = torch.Generator()
    g.manual_seed(SEED)

    # c) 实例化 PyG 的 DataLoader (使用默认collate，无需自定义)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker)
    
    # --- 5. 初始化模型、损失函数和优化器 ---
    
    
    field_embedder = FieldEmbedding(config_path, vocab_path)
    field_embedder.to(device)

    pta_model = HierarchicalMoE(
        config_path=config_path,
        vocab_path=vocab_path,
        num_classes=num_classes,
        expert_graph_info=expert_graph_info, # <-- 传入专家定义
        use_flow_features=USE_FLOW_FEATURES_THIS_RUN,
        num_flow_features=len(train_dataset.flow_feature_names) if USE_FLOW_FEATURES_THIS_RUN else 0,
        hidden_dim=GNN_HIDDEN_DIM, 
        dropout_rate=DROPOUT_RATE
    ).to(device)

    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add weight_decay

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # 我们的目标是最大化 F1
        factor=0.2,      # 当F1停滞时，将 LR 乘以 0.2 (例如: 1e-3 -> 2e-4 -> 4e-5)
        patience=5,      # 【关键】如果 Val F1 在 5 个 epoch 内没有创下新高...
        verbose=True,     # ... 打印一条消息并降低 LR
        min_lr=1e-6 # (你可以保留你现有的 MIN_LR_FOR_TRAINING 逻辑)
    )

    # 【关键】初始化一个“动态权重”张量，一开始所有类别权重都为1.0
    # dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)

    # --- 4. 训练循环 ---
    if not DIAGNOSE: 
        training_results = []
        best_f1 = 0.0
        best_val_f1_macro = 0.0 
        epochs_since_best = 0 
        best_epoch = -1
        best_model_state = None # 在内存中保存最佳模型
        for epoch in range(NUM_EPOCHS): 
            if stop_training:
                print("Learning rate too low. Stopping training early.")
                break

            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            train_metrics, _ = train_one_epoch(pta_model, train_loader, # loss_fn, 
                                               optimizer, # scheduler, 
                                               device, num_classes, loss_fn)
                                            #    dynamic_weights=dynamic_weights)
            # val_metrics, _, val_per_class_f1 = evaluate(pta_model, val_loader, # loss_fn, 
            #                           device, num_classes)
            val_metrics, _ = evaluate(pta_model, val_loader, # loss_fn, 
                                      device, num_classes, loss_fn)
            
            # beta = 2.0 
            # new_weights = (1.0 - val_per_class_f1)**beta
            # # 归一化，防止权重爆炸
            # new_weights = new_weights / new_weights.mean() 
            # dynamic_weights = new_weights.to(device)

            # beta = 1.0 # <-- 可以使用一个较温和的beta，比如1.0
            # new_weights = (1.0 - val_per_class_f1.cpu())**beta # 确保在CPU上计算
            # new_weights = new_weights / new_weights.mean() # 归一化

            # # --- [!! 核心修复 !!] ---
            # # 不要直接替换，使用EMA（指数移动平均）进行平滑更新
            # # 0.9 是“旧权重”的惯性，0.1 是“新权重”的更新力度
            # momentum = 0.9 
            # dynamic_weights_cpu = dynamic_weights.cpu() # 移动到CPU

            # updated_weights = (momentum * dynamic_weights_cpu) + ((1 - momentum) * new_weights)

            # # 将新权重移回GPU
            # dynamic_weights = updated_weights.to(device)
            # --- [!! 修复结束 !!] ---

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1 (macro): {train_metrics['f1_macro']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1 (macro): {val_metrics['f1_macro']:.4f}")

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Summary (Current LR: {current_lr:.1e}):")
            # print(f"Epoch {epoch+1} Summary (LR: {scheduler.get_last_lr()[0]:.1e}):")

            training_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'], 
                'train_recall_macro': train_metrics['recall_macro'], 
                'train_precision_macro': train_metrics['precision_macro'], 
                'train_f1_macro': train_metrics['f1_macro'], 
                'train_recall_weighted': train_metrics['recall_weighted'], 
                'train_precision_weighted': train_metrics['precision_weighted'], 
                'train_f1_weighted': train_metrics['f1_weighted'], 
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'], 
                'val_recall_macro': val_metrics['recall_macro'], 
                'val_precision_macro': val_metrics['precision_macro'], 
                'val_f1_macro': val_metrics['f1_macro'], 
                'val_recall_weighted': val_metrics['recall_weighted'], 
                'val_precision_weighted': val_metrics['precision_weighted'], 
                'val_f1_weighted': val_metrics['f1_weighted'], 
            })

            current_val_f1_macro = val_metrics['f1_macro']
            scheduler.step(current_val_f1_macro)

            if current_val_f1_macro > best_val_f1_macro:
                # --- 发现新高点 ---
                print(f" -> Validation Macro F1 improved from {best_val_f1_macro:.4f} to {current_val_f1_macro:.4f}. Saving state...")
                best_val_f1_macro = current_val_f1_macro
                best_epoch = epoch + 1
                # 【保存】使用深拷贝将最佳状态保存到内存
                best_model_state = copy.deepcopy(pta_model.state_dict())
                torch.save(pta_model.state_dict(), os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_model.pth'))
                epochs_since_best = 0
            else:
                # --- 未发现新高点 ---
                epochs_since_best += 1
                print(f" -> Validation Macro F1 did not improve for {epochs_since_best} epoch(s). Best was {best_val_f1_macro:.4f} at epoch {best_epoch}.")

                if epochs_since_best >= ROLLBACK_PATIENCE:
                    print(f"\n!!! Performance has not improved for {ROLLBACK_PATIENCE} epochs. Rolling back to best model from epoch {best_epoch}. !!!")

                    if best_model_state:
                        # 1. 【回滚】
                        pta_model.load_state_dict(best_model_state)

                        # # 2. 【手动降LR】
                        # print("   -> Aggressively reducing current learning rate by half...")
                        # new_lr = optimizer.param_groups[0]['lr'] * 0.5
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = new_lr

                        # 3. 【重置计数器】
                        epochs_since_best = 0

                        # # 4. 【增加早停条件】
                        # if new_lr < MIN_LR_FOR_TRAINING:
                        #     print(f"   -> Learning rate ({new_lr:.1e}) has fallen below minimum. Triggering final early stop.")
                        #     stop_training = True # 在下一个epoch开始时停止
                    else:
                        print("   -> Warning: No best model state found. Stopping training.")
                        break # 如果从未保存过最佳状态就触发回滚，直接停止
                # 5. 【修改】早停逻辑现在只检查LR
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < MIN_LR_FOR_TRAINING:
                    print(f"Learning rate ({current_lr:.1e}) has fallen below minimum. Triggering final early stop.")
                    stop_training = True # 在下一个epoch开始时停止

        print("\nTraining complete!")

        # ==================== 分析学到的特征重要性 ====================
        print("\n" + "="*50)
        print("###   Learned Feature Importance Report   ###")

        importance_reports_dict = pta_model.get_feature_importance()

        # # to_string()可以打印所有行，确保能看到完整的报告
        # print(importance_report.to_string())

        # # 保存报告为CSV，以便后续分析
        # importance_report.to_csv(os.path.join(res_path,train_set_name + '_feature_importance_report.csv'), index=False)
        # print("\nFeature importance report saved to 'feature_importance_report.csv'")
        # print("="*50)

        all_reports_list = []
        for expert_name, expert_df in importance_reports_dict.items():
            print(f"\n--- Importance for Expert: '{expert_name}' ---")
            # to_string()可以打印所有行
            print(expert_df.to_string())
        
            # 为合并做准备
            expert_df_with_name = expert_df.copy()
            expert_df_with_name['expert_name'] = expert_name
            all_reports_list.append(expert_df_with_name)           
        # 【修复】将所有报告合并为一个大的DataFrame
        combined_report_df = pd.concat(all_reports_list).reset_index(drop=True)

        # 保存报告为CSV
        report_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_feature_importance_report.csv')
        combined_report_df.to_csv(report_output_path, index=False)

        print(f"\nCombined feature importance report saved to: {report_output_path}")
        print("="*50)

        # --- 5. 最终测试 ---
        pta_model.load_state_dict(torch.load(os.path.join(res_path, dataset_name + '_' + train_set_name + '_best_model.pth')))
        pta_model.to(device)
        test_metrics, test_confusion_matrix = evaluate(pta_model, test_loader, device, num_classes, loss_fn)
        print(f"\nFinal Test Performance:")
        print(f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

        # --- 7. 保存混淆矩阵到CSV ---
        print("\nSaving confusion matrix...")

        # a) 创建从整数索引回字符串标签的映射
        #    我们需要之前创建的 label_to_int 字典
        int_to_label = {i: label for label, i in label_to_int.items()}
        class_names = [int_to_label[i] for i in range(num_classes)]

        # b) 将PyTorch Tensor转换为带标签的Pandas DataFrame
        confusion_matrix_df = pd.DataFrame(
            test_confusion_matrix.cpu().numpy(), # 必须先移回CPU
            index=class_names,
            columns=class_names
        )

        # c) 保存为CSV文件
        cm_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_final_test_confusion_matrix.csv')
        confusion_matrix_df.to_csv(cm_output_path)

        print(f"Confusion matrix saved to: {cm_output_path}")

        training_results.append({
            'epoch': 'final_test',
            'train_loss': None,
            'train_acc': None, 
            'train_recall_macro': None, 
            'train_precision_macro': None, 
            'train_f1_macro': None, 
            'train_recall_weighted': None, 
            'train_precision_weighted': None, 
            'train_f1_weighted': None, 
            'val_loss': test_metrics['loss'],
            'val_acc': test_metrics['accuracy'], 
            'val_recall_macro': test_metrics['recall_macro'], 
            'val_precision_macro': test_metrics['precision_macro'], 
            'val_f1_macro': test_metrics['f1_macro'], 
            'val_recall_weighted': test_metrics['recall_weighted'], 
            'val_precision_weighted': test_metrics['precision_weighted'], 
            'val_f1_weighted': test_metrics['f1_weighted']
        })

        results_df = pd.DataFrame(training_results)
        results_df.to_csv(os.path.join(res_path,dataset_name + '_' + train_set_name + '_training_log.csv'), index=False)
        print(f"\nTraining log saved to {train_set_name}_training_log.csv")

    elif DIAGNOSE: 
        d_res_dir = ''
        best_model_path = os.path.join(res_path, d_res_dir,dataset_name + '_' + train_set_name + '_best_model.pth') 
        if not os.path.exists(best_model_path):
            print(f"错误: 找不到已保存的模型文件: {best_model_path}")
            print("请确保 'train_set_name' 变量与你训练时的设置一致。")
            exit()
        pta_model.load_state_dict(torch.load(best_model_path, map_location=device))
        # pta_model.to(device)
        pta_model.eval()
        test_metrics, test_confusion_matrix = evaluate(pta_model, test_loader, 
                                                           device, num_classes, loss_fn)
        print(f"\nFinal Test Performance:")
        print(f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

        # --- 7. 保存混淆矩阵到CSV ---
        print("\nSaving confusion matrix...")

        # a) 创建从整数索引回字符串标签的映射
        #    我们需要之前创建的 label_to_int 字典
        int_to_label = {i: label for label, i in label_to_int.items()}
        class_names = [int_to_label[i] for i in range(num_classes)]

        # b) 将PyTorch Tensor转换为带标签的Pandas DataFrame
        confusion_matrix_df = pd.DataFrame(
            test_confusion_matrix.cpu().numpy(), # 必须先移回CPU
            index=class_names,
            columns=class_names
        )

        # c) 保存为CSV文件
        cm_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_final_test_confusion_matrix.csv')
        confusion_matrix_df.to_csv(cm_output_path)

        print(f"Confusion matrix saved to: {cm_output_path}")

        # ==================== 分析学到的特征重要性 ====================
        print("\n" + "="*50)
        print("###   Learned Feature Importance Report   ###")

        importance_reports_dict = pta_model.get_feature_importance()
        all_reports_list = []
        for expert_name, expert_df in importance_reports_dict.items():
            print(f"\n--- Importance for Expert: '{expert_name}' ---")
            # to_string()可以打印所有行
            print(expert_df.to_string())
        
            # 为合并做准备
            expert_df_with_name = expert_df.copy()
            expert_df_with_name['expert_name'] = expert_name
            all_reports_list.append(expert_df_with_name)           
        # 【修复】将所有报告合并为一个大的DataFrame
        combined_report_df = pd.concat(all_reports_list).reset_index(drop=True)

        # 保存报告为CSV
        report_output_path = os.path.join(res_path, dataset_name + '_' + train_set_name + '_feature_importance_report.csv')
        combined_report_df.to_csv(report_output_path, index=False)

        print(f"\nCombined feature importance report saved to: {report_output_path}")
        print("="*50)
        print("Diagnose completed. ")