import torch
import torch.optim as optim 
import torch.nn as nn 
from tqdm import tqdm 
from utils.data_loader import TrafficDataset
from torch.utils.data import Dataset #, DataLoader
from torch_geometric.loader import DataLoader
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
from utils.data_loader_ptga import GNNTrafficDataset
from torch_geometric.loader import DataLoader
from models.ProtocolTreeGAttention import ProtocolTreeGAttention
from utils.metrics import calculate_metrics
from utils.model_utils import diagnose_gate_weights_for_class
import sys
from transformers import get_linear_schedule_with_warmup
from utils.loss_functions import FocalLoss
import numpy as np
import random 
from torch.optim import RAdam
import copy

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

def train_one_epoch(model, dataloader, # loss_fn, 
                    optimizer, device, num_classes, 
                    dynamic_weights: torch.Tensor, 
                    alpha=1e-4):
    """
    一个完整的、带有负熵正则化的训练函数。

    :param alpha: 负熵正则化损失的权重系数。
    """
    model.train()
    running_loss = 0.0
    
    # 初始化混淆矩阵，用于计算详细指标
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # 【关键】在函数内部创建最基础的损失函数
    # reduction='none' 意味着它会为batch中的每个样本都返回一个损失值
    base_loss_fn = nn.CrossEntropyLoss(reduction='none')

    for i, batched_graph in enumerate(tqdm(dataloader, desc="Training")):
        # 1. 将数据移动到GPU
        batched_graph.to(device)
        labels = batched_graph.y
        
        outputs, gate = model(batched_graph)

        # 1. 计算【基础】分类损失 (形状: [B])
        classification_loss_per_sample = base_loss_fn(outputs, labels)

        # 2. 【核心】应用动态权重
        #    根据每个样本的真实标签，从 dynamic_weights 中获取对应的权重
        sample_weights = dynamic_weights[labels]
        #    计算加权后的批次平均损失
        classification_loss = (classification_loss_per_sample * sample_weights).mean()
        

        # c) 计算负熵正则化损失，鼓励gate权重保持在0.5附近，避免过早饱和
        #    我们希望最大化熵，即最小化负熵
        mask_entropy_loss = -(gate * torch.log(gate + 1e-8) + 
                              (1 - gate) * torch.log(1 - gate + 1e-8)).mean()

        # d) 计算加权总损失
        # ==================== 核心修改点：暂时禁用负熵正则化 ====================
        # alpha = 0.0 
        # alpha = 1e-4
        total_loss = classification_loss + alpha * mask_entropy_loss
        
        # ======================================================================
        # if i % 50 == 0: # 每50个batch打印一次
        #     print(f"\n  Batch {i}: CE_Loss={classification_loss.item():.4f}, Ent_Loss={mask_entropy_loss.item():.4f}, Total_Loss={total_loss.item():.4f}")
        
        # 3. 使用【总损失】进行反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 4. （可选但推荐）梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # ==================== 核心修改点：按Step更新 ====================
        # OneCycleLR需要在每个batch(step)之后被调用
        # scheduler.step()
        # =============================================================

        # 5. 统计指标时，我们只关心【主任务的损失】
        running_loss += classification_loss.item() * batched_graph.num_graphs
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


def evaluate(model, dataloader, # loss_fn, 
             device, num_classes):
    """
    一个完整的、适配新模型输出的评估函数。
    """
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # 损失函数现在只在评估时用于报告，不在内部计算
    eval_loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for batched_graph in tqdm(dataloader, desc="Evaluating"):
            batched_graph.to(device)
            labels = batched_graph.y
            
            # ==================== 核心修改点：正确处理模型输出 ====================
            #
            # 模型依然返回两个值，但在评估时我们只关心第一个（logits）
            #
            outputs, _ = model(batched_graph)
            #
            # =================================================================

            # loss = loss_fn(outputs, labels)
            loss = eval_loss_fn(outputs, labels) # <-- 只用于报告

            running_loss += loss.item() * batched_graph.num_graphs
            _, predicted = torch.max(outputs.data, 1)

            labels_cpu = labels.cpu()
            predicted_cpu = predicted.cpu()
            for t, p in zip(labels_cpu.view(-1), predicted_cpu.view(-1)):
                if t < num_classes and p < num_classes:
                    confusion_matrix[t.long(), p.long()] += 1
                
    total_samples = confusion_matrix.sum().item()
    epoch_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    
    epoch_metrics = calculate_metrics(confusion_matrix)
    epoch_metrics['loss'] = epoch_loss

    # 2. 【关键】单独计算并返回“每类F1分数”
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    per_class_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return epoch_metrics, confusion_matrix, per_class_f1

# =====================================================================
if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    # LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    # WEIGHT_DECAY = 5e-4
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
    train_set_name = dataset_name + '_chief_block_topk_augmented'
    val_set_name = 'validation_set' 
    test_set_name = 'test_set'
    chief_directory = train_dir
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

    

    # # a) 确定“主干Schema”，即模型期望的输入特征
    # chief_schema = [col for col in train_df.columns if col not in ['label', 'label_id']]
    
    # # b) 【关键】对验证集和测试集进行特征空间对齐
    # print("\n[2/4] Aligning feature space for validation and test sets...")
    
    # # 对齐验证集
    # val_df_aligned = pd.DataFrame(columns=chief_schema)
    # for col in chief_schema:
    #     if col in val_df.columns:
    #         val_df_aligned[col] = val_df[col]
    #     else:
    #         val_df_aligned[col] = '0' # 用'0'填充缺失的特征
    # val_df_aligned['label'] = val_df['label'] # 补回标签列
    
    # # 对齐测试集
    # test_df_aligned = pd.DataFrame(columns=chief_schema)
    # for col in chief_schema:
    #     if col in test_df.columns:
    #         test_df_aligned[col] = test_df[col]
    #     else:
    #         test_df_aligned[col] = '0'
    # test_df_aligned['label'] = test_df['label']

    # print(" - Feature alignment complete.")
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
    
    # b) 从训练数据集中获取模型需要的节点列表
    node_fields_for_model = train_dataset.node_fields
    print(f" - Model will be built for {len(node_fields_for_model)} nodes.")

    g = torch.Generator()
    g.manual_seed(SEED)

    # c) 实例化 PyG 的 DataLoader (使用默认collate，无需自定义)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    
    # --- 5. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    field_embedder = FieldEmbedding(config_path, vocab_path)
    field_embedder.to(device)

    if USE_FLOW_FEATURES_THIS_RUN:
        print("\n[5/5] 初始化模型 (模式: PTGA + 流特征)")
        # 确保您的数据预处理步骤 [2.5/4] 已经运行
        pta_model = ProtocolTreeGAttention(
            field_embedder=field_embedder,
            num_classes=num_classes,
            node_fields_list=node_fields_for_model,
            
            use_flow_features=True, # <-- 开启
            num_flow_features=3,    # <-- 指定流特征数量
            
            hidden_dim=GNN_HIDDEN_DIM, 
            dropout_rate=DROPOUT_RATE
        ).to(device)
    else:
        print("\n[5/5] 初始化模型 (模式: 原始PTGA，不含流特征)")
        # 确保您【跳过】了数据预处理的步骤 [2.5/4]
        # (或者GNNTrafficDataset中的[c] [e]步骤被跳过)
        pta_model = ProtocolTreeGAttention(
            field_embedder=field_embedder,
            num_classes=num_classes,
            node_fields_list=node_fields_for_model,
            hidden_dim=GNN_HIDDEN_DIM, 
            dropout_rate=DROPOUT_RATE
        ).to(device)

    # pta_model = ProtocolTreeGAttention(
    #     field_embedder=field_embedder,
    #     num_classes=num_classes,
    #     node_fields_list=node_fields_for_model,
    #     num_flow_features=3, 
    #     hidden_dim=GNN_HIDDEN_DIM, 
    #     dropout_rate=DROPOUT_RATE # change to 0.5 to against overfit
    # ).to(device)

    # ADD: Class Weighting
    # # 1. 计算每个类别的权重 (样本数越少，权重越高)
    # class_counts = train_df['label_id'].value_counts().sort_index().values
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # class_weights = class_weights / class_weights.sum() * num_classes # 归一化
    # class_weights = class_weights.to(device)

    # # 2. 将权重传入损失函数
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add weight_decay
    # optimizer = RAdam(pta_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # # ==============================================================

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-4) # 监控f1_macro

    # 【关键】初始化一个“动态权重”张量，一开始所有类别权重都为1.0
    dynamic_weights = torch.ones(num_classes, dtype=torch.float).to(device)

    DIAGNOSE = False
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
            # scheduler.step()
            # scheduler.step(val_metrics['loss']) 
            # scheduler.step(val_metrics['f1_macro']) 

            # 3. 【关键】动态更新损失权重
            #    这是一个简单的启发式规则：权重 = (1 - F1)^2
            #    F1越低，权重越高
            beta = 2.0 
            new_weights = (1.0 - val_per_class_f1)**beta
            # 归一化，防止权重爆炸
            new_weights = new_weights / new_weights.mean() 
            dynamic_weights = new_weights.to(device)
            
            # scheduler.step(val_metrics['f1_macro']) 


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

            # 这里可以添加保存最佳模型的逻辑
            # if val_metrics['f1_macro'] > best_f1:
            #     torch.save(pta_model.state_dict(), os.path.join(res_path, train_set_name + '_best_model.pth'))
            #     print("The best epoch parameters has been saved. ")
            #     best_f1 = val_metrics['f1_macro']

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

                # # --- 触发回滚与早停 ---
                # if epochs_since_best >= PATIENCE:
                #     print(f"\n!!! Performance has not improved for {PATIENCE} epochs. Rolling back to best model from epoch {best_epoch}. !!!")
                #     rollback_times += 1
                #     # 【回滚】
                #     if best_model_state:
                #         pta_model.load_state_dict(best_model_state)
                #     if rollback_times >= 3: 
                #         break 

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

        importance_report = pta_model.get_feature_importance()

        # to_string()可以打印所有行，确保能看到完整的报告
        print(importance_report.to_string())

        # 保存报告为CSV，以便后续分析
        importance_report.to_csv(os.path.join(res_path,train_set_name + '_feature_importance_report.csv'), index=False)
        print("\nFeature importance report saved to 'feature_importance_report.csv'")
        print("="*50)

        # --- 5. 最终测试 ---
        pta_model.load_state_dict(torch.load(os.path.join(res_path, train_set_name + '_best_model.pth')))
        pta_model.to(device)
        test_metrics, test_confusion_matrix, _ = evaluate(pta_model, test_loader, 
                                                    #    loss_fn, 
                                                       device, num_classes)
        print(f"\nFinal Test Performance:")
        print(f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")

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
         # --- 诊断模式 ---
        print("\n" + "="*40)
        print("###   运行模式：诊断 (DIAGNOSTIC MODE)   ###")
        print("="*40)
        
        model_path = os.path.join(res_path,train_set_name + '_best_model.pth')
        if not os.path.exists(model_path):
            print(f"错误: 模型文件未找到 -> {model_path}")
            print("请确保将训练好的 'best_model.pth' 文件放在当前目录下，或通过 --model_path 指定路径。")
            sys.exit(1) # 退出程序

        # a) 加载已训练好的模型参数
        print(f"正在从 {model_path} 加载模型参数...")
        pta_model.load_state_dict(torch.load(model_path))
        pta_model.eval()

        # ==================== 核心修改点：调用新的诊断函数 ====================
        
        # 诊断 'google', 'twitter', 和一个表现良好的 'tudou' 作为对比
        diagnose_gate_weights_for_class(pta_model, val_dataset, 'google', label_to_int, device)
        diagnose_gate_weights_for_class(pta_model, val_dataset, 'twitter', label_to_int, device)
        diagnose_gate_weights_for_class(pta_model, val_dataset, 'tudou', label_to_int, device)
        
        # =======================================================================
        print("="*80 + "\n")
        print("诊断完成。")