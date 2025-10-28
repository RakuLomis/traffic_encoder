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
    dynamic_weights: torch.Tensor,  # <-- 动态权重
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
    base_loss_fn = nn.CrossEntropyLoss(reduction='none')

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
        
        # 3. 【新】模型现在接收字典，并返回 logits 和 门控字典
        #    outputs = logits
        #    gates_dict = {'eth': gate_tensor, 'ip': gate_tensor, ...}
        outputs, gates_dict = model(batch_dict)
        
        # 4. 计算【基础】分类损失 (形状: [B])
        classification_loss_per_sample = base_loss_fn(outputs, labels)

        # 5. 【核心】应用动态权重
        #    根据每个样本的真实标签，从 dynamic_weights 中获取对应的权重
        sample_weights = dynamic_weights[labels]
        #    计算加权后的批次平均损失
        classification_loss = (classification_loss_per_sample * sample_weights).mean()
        
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
    num_classes: int
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """
    【终极架构版】
    为“分层语义MoE”模型（HierarchicalMoE）定制的评估函数。
    它处理“图字典”的批处理，并返回详细指标以及“每类F1分数”。
    """
    model.eval() # 将模型设置为评估模式
    
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    
    # 我们在内部创建一个简单的损失函数，只用于【报告】损失值
    base_loss_fn = nn.CrossEntropyLoss()

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
        
        # 3. 【新】模型返回两个值，评估时我们只关心第一个（logits）
        outputs, _ = model(batch_dict) # 忽略 gates_dict
        
        # 4. 计算并累积损失（仅用于报告）
        loss = base_loss_fn(outputs, labels)
        running_loss += loss.item() * labels.size(0) # 使用 labels.size(0) 作为批次大小
        
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
    
    # 3. 【关键新增】计算并返回“每类F1分数”张量
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    epsilon = 1e-8 # 防止除以零
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # per_class_f1 是一个在GPU/CPU上的张量，形状为 [num_classes]
    per_class_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # 确保返回的是在CPU上的张量，以便主循环使用
    return epoch_metrics, cm_cpu, per_class_f1.cpu()


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
    DROPOUT_RATE = 0.45
    NUM_WORKERS = 4 
    GNN_INPUT_DIM = 32 
    GNN_HIDDEN_DIM = 128
    PATIENCE = 5

    USE_FLOW_FEATURES_THIS_RUN = False

    # FocalLoss的超参数
    FOCAL_GAMMA = 2.0 # 0.0 ~ 5.0, 2.0是一个经典的起始值

    ROLLBACK_PATIENCE = 10
    MIN_LR_FOR_TRAINING = 1e-6
    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    # dataset_name = 'ISCX-VPN'
    # dataset_name = 'ISCX-TOR-Acctivity'
    dataset_name = 'ISCX-TOR-Application'
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
        # ==================== 核心修改点：流统计特征工程 ====================
        print("\n[2.5/4] Performing Flow-level Statistics Engineering...")

        # 这是一个开关，决定了我们是模拟“现实世界”（True）还是进行“理想实验”（False）
        OPEN_WORLD = False 

        # a) 定义流特征名称
        flow_feature_names = ['flow_avg_len', 'flow_std_len', 'flow_pkt_count']

        # b) 确保训练集的统计列是数值型的
        print(" -> Converting stats columns in Train set to numeric...")
        train_df['ip.len'] = pd.to_numeric(train_df['ip.len'], errors='coerce').fillna(0)
        train_df['tcp.stream'] = pd.to_numeric(train_df['tcp.stream'], errors='coerce').fillna(0)

        # c) 【关键】从训练集中学习“知识”
        print(" -> Learning per-flow statistics from Train set...")
        train_flow_avg_len = train_df.groupby('tcp.stream')['ip.len'].mean()
        train_flow_std_len = train_df.groupby('tcp.stream')['ip.len'].std().fillna(0)
        train_flow_pkt_count = train_df.groupby('tcp.stream')['ip.len'].count()

        # d) 【关键】计算用于填充“新流”的“全局默认值” (从训练集中学到)
        train_global_avg_len = train_flow_avg_len.mean()
        train_global_std_len = train_flow_std_len.mean()
        train_global_pkt_count = train_flow_pkt_count.mean()

        print(f" -> Learned global defaults: AvgLen={train_global_avg_len:.2f}, StdLen={train_global_std_len:.2f}, PktCount={train_global_pkt_count:.2f}")

        # e) 将“知识”应用（广播）回训练集
        print(" -> Applying learned stats back to Train set...")
        train_df['flow_avg_len'] = train_df['tcp.stream'].map(train_flow_avg_len)
        train_df['flow_std_len'] = train_df['tcp.stream'].map(train_flow_std_len)
        train_df['flow_pkt_count'] = train_df['tcp.stream'].map(train_flow_pkt_count)
        # 填充训练集自身可能出现的（例如只有一个包的流）的NaN
        # train_df['flow_std_len'].fillna(train_global_std_len, inplace=True)
        train_df['flow_std_len'] = train_df['flow_std_len'].fillna(train_global_std_len)


        if OPEN_WORLD:
            # --- 方案A: 模拟现实世界 (无数据泄露) ---
            print(" -> [OPEN_WORLD MODE] Applying stats learned from Train set to Val/Test sets...")

            for df in [val_df_aligned, test_df_aligned]:
                df['ip.len'] = pd.to_numeric(df['ip.len'], errors='coerce').fillna(0)
                df['tcp.stream'] = pd.to_numeric(df['tcp.stream'], errors='coerce').fillna(0)

            # 应用 .map() 并使用 .fillna() 填充“新流”
            val_df_aligned['flow_avg_len'] = val_df_aligned['tcp.stream'].map(train_flow_avg_len).fillna(train_global_avg_len)
            val_df_aligned['flow_std_len'] = val_df_aligned['tcp.stream'].map(train_flow_std_len).fillna(train_global_std_len)
            val_df_aligned['flow_pkt_count'] = val_df_aligned['tcp.stream'].map(train_flow_pkt_count).fillna(train_global_pkt_count)

            test_df_aligned['flow_avg_len'] = test_df_aligned['tcp.stream'].map(train_flow_avg_len).fillna(train_global_avg_len)
            test_df_aligned['flow_std_len'] = test_df_aligned['tcp.stream'].map(train_flow_std_len).fillna(train_global_std_len)
            test_df_aligned['flow_pkt_count'] = test_df_aligned['tcp.stream'].map(train_flow_pkt_count).fillna(train_global_pkt_count)

        else:
            # --- 方案B: 您的“理想情况”实验 (有数据泄露) ---
            print(" -> [CLOSED_WORLD MODE] Calculating and applying stats directly from Val/Test sets...")

            # 迭代并【就地修改】验证集和测试集
            for df_name, df in [('Validation', val_df_aligned), ('Test', test_df_aligned)]:
                print(f" -> Calculating stats directly from {df_name} set...")
                if df.empty:
                    continue

                # 1. 转换数值
                df['ip.len'] = pd.to_numeric(df['ip.len'], errors='coerce').fillna(0)
                df['tcp.stream'] = pd.to_numeric(df['tcp.stream'], errors='coerce').fillna(0) 

                # 2. 【关键】计算此DataFrame【自身】的统计数据
                df_flow_avg_len = df.groupby('tcp.stream')['ip.len'].mean()
                df_flow_std_len = df.groupby('tcp.stream')['ip.len'].std().fillna(0)
                df_flow_pkt_count = df.groupby('tcp.stream')['ip.len'].count()

                # 3. 计算此DataFrame【自身】的全局平均值
                df_global_avg_len = df_flow_avg_len.mean()
                df_global_std_len = df_flow_std_len.mean()
                df_global_pkt_count = df_flow_pkt_count.mean() 

                # 4. 【关键】将【自身】的统计数据map回自身
                df['flow_avg_len'] = df['tcp.stream'].map(df_flow_avg_len)
                df['flow_std_len'] = df['tcp.stream'].map(df_flow_std_len)
                df['flow_pkt_count'] = df['tcp.stream'].map(df_flow_pkt_count)

                # 5. 填充可能产生的NaN (例如只有一个包的流，其std为NaN)
                # df['flow_std_len'].fillna(df_global_std_len, inplace=True)
                df['flow_std_len'] = df['flow_std_len'].fillna(df_global_std_len)

        print(" -> Flow-level features successfully engineered for all datasets.")


    # c) 创建全局标签映射
    #    为了确保所有数据集的标签一致，我们基于训练集来创建映射
    print("\n[3/4] Creating label mapping...")
    labels = train_df[F'label'].unique()
    label_to_int = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    train_df['label_id'] = train_df['label'].map(label_to_int)
    val_df_aligned['label_id'] = val_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) # .fillna(-1)处理未见过的标签
    test_df_aligned['label_id'] = test_df_aligned['label'].map(label_to_int).fillna(-1).astype(int) 
    

    # --- 4. 创建GNN Dataset和DataLoader ---
    print("\n[4/4] Creating GNN Datasets and DataLoaders...")
    
    # a) 实例化 GNNTrafficDataset
    train_dataset = GNNTrafficDataset(train_df, config_path, vocab_path)
    val_dataset = GNNTrafficDataset(val_df_aligned, config_path, vocab_path)
    test_dataset = GNNTrafficDataset(test_df_aligned, config_path, vocab_path)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
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

    # 【关键】初始化一个“动态权重”张量，一开始所有类别权重都为1.0
    dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)

    DIAGNOSE = True
    stop_training = False
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
                                               device, num_classes, 
                                               dynamic_weights=dynamic_weights)
            val_metrics, _, val_per_class_f1 = evaluate(pta_model, val_loader, # loss_fn, 
                                      device, num_classes)
            
            # beta = 2.0 
            # new_weights = (1.0 - val_per_class_f1)**beta
            # # 归一化，防止权重爆炸
            # new_weights = new_weights / new_weights.mean() 
            # dynamic_weights = new_weights.to(device)

            beta = 1.0 # <-- 可以使用一个较温和的beta，比如1.0
            new_weights = (1.0 - val_per_class_f1.cpu())**beta # 确保在CPU上计算
            new_weights = new_weights / new_weights.mean() # 归一化

            # --- [!! 核心修复 !!] ---
            # 不要直接替换，使用EMA（指数移动平均）进行平滑更新
            # 0.9 是“旧权重”的惯性，0.1 是“新权重”的更新力度
            momentum = 0.9 
            dynamic_weights_cpu = dynamic_weights.cpu() # 移动到CPU

            updated_weights = (momentum * dynamic_weights_cpu) + ((1 - momentum) * new_weights)

            # 将新权重移回GPU
            dynamic_weights = updated_weights.to(device)
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
            if current_val_f1_macro > best_val_f1_macro:
                # --- 发现新高点 ---
                print(f" -> Validation Macro F1 improved from {best_val_f1_macro:.4f} to {current_val_f1_macro:.4f}. Saving state...")
                best_val_f1_macro = current_val_f1_macro
                best_epoch = epoch + 1
                # 【保存】使用深拷贝将最佳状态保存到内存
                best_model_state = copy.deepcopy(pta_model.state_dict())
                torch.save(pta_model.state_dict(), os.path.join(res_path, train_set_name + '_best_model.pth'))
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

                        # 2. 【手动降LR】
                        print("   -> Aggressively reducing current learning rate by half...")
                        new_lr = optimizer.param_groups[0]['lr'] * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr

                        # 3. 【重置计数器】
                        epochs_since_best = 0

                        # 4. 【增加早停条件】
                        if new_lr < MIN_LR_FOR_TRAINING:
                            print(f"   -> Learning rate ({new_lr:.1e}) has fallen below minimum. Triggering final early stop.")
                            stop_training = True # 在下一个epoch开始时停止
                    else:
                        print("   -> Warning: No best model state found. Stopping training.")
                        break # 如果从未保存过最佳状态就触发回滚，直接停止
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
        report_output_path = os.path.join(res_path, train_set_name + '_feature_importance_report.csv')
        combined_report_df.to_csv(report_output_path, index=False)

        print(f"\nCombined feature importance report saved to: {report_output_path}")
        print("="*50)

        # --- 5. 最终测试 ---
        pta_model.load_state_dict(torch.load(os.path.join(res_path, train_set_name + '_best_model.pth')))
        pta_model.to(device)
        test_metrics, test_confusion_matrix, _ = evaluate(pta_model, test_loader, 
                                                    #    loss_fn, 
                                                       device, num_classes)
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
        cm_output_path = os.path.join(res_path, train_set_name + '_final_test_confusion_matrix.csv')
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
        results_df.to_csv(os.path.join(res_path,train_set_name + '_training_log.csv'), index=False)
        print(f"\nTraining log saved to {train_set_name}_training_log.csv")

    elif DIAGNOSE: 
        best_model_path = os.path.join(res_path, train_set_name + '_best_model.pth') 
        if not os.path.exists(best_model_path):
            print(f"错误: 找不到已保存的模型文件: {best_model_path}")
            print("请确保 'train_set_name' 变量与你训练时的设置一致。")
            exit()
        pta_model.load_state_dict(torch.load(best_model_path, map_location=device))
        # pta_model.to(device)
        pta_model.eval()
        test_metrics, test_confusion_matrix, _ = evaluate(pta_model, test_loader, 
                                                    #    loss_fn, 
                                                       device, num_classes)
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
        cm_output_path = os.path.join(res_path, train_set_name + '_final_test_confusion_matrix.csv')
        confusion_matrix_df.to_csv(cm_output_path)

        print(f"Confusion matrix saved to: {cm_output_path}")