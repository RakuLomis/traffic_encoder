import warnings
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
import numpy as np 
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
import copy

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler,device, num_classes, alpha=1e-3):
    """
    一个完整的、带有负熵正则化的训练函数。

    :param alpha: 负熵正则化损失的权重系数。
    """
    model.train()
    running_loss = 0.0
    
    # 初始化混淆矩阵，用于计算详细指标
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # ==================== CORE FIX: Initialize the counter ====================
    grad_clip_counter = 0 # Initialize counter to 0 before the loop
    # =======================================================================
    for i, batched_graph in enumerate(tqdm(dataloader, desc="Training")):
        # 1. 将数据移动到GPU
        batched_graph.to(device)
        labels = batched_graph.y
        
        # ==================== 核心修改点 1：接收gate并计算总损失 ====================
        
        # a) 模型现在返回两个输出：预测logits和特征门控权重
        outputs, gate = model(batched_graph)
        
        # b) 计算主任务的交叉熵损失
        classification_loss = loss_fn(outputs, labels)

        # c) 计算负熵正则化损失，鼓励gate权重保持在0.5附近，避免过早饱和
        #    我们希望最大化熵，即最小化负熵
        mask_entropy_loss = -(gate * torch.log(gate + 1e-8) + 
                              (1 - gate) * torch.log(1 - gate + 1e-8)).mean()

        # d) 计算加权总损失
        # ==================== 核心修改点：暂时禁用负熵正则化 ====================
        # alpha = 0.0 
        alpha = 1e-4
        total_loss = classification_loss + alpha * mask_entropy_loss
        
        # ======================================================================
        # if i % 50 == 0: # 每50个batch打印一次
        #     print(f"\n  Batch {i}: CE_Loss={classification_loss.item():.4f}, Ent_Loss={mask_entropy_loss.item():.4f}, Total_Loss={total_loss.item():.4f}")
        
        # 3. 使用【总损失】进行反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 4. （可选但推荐）梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

        # # ==================== 自适应梯度裁剪 ====================
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        # # 记录裁剪情况，但暂时不在这里手动调整LR，让SequentialLR主导
        # if total_norm.item() > 0.5:
        #      grad_clip_counter += 1
        # # =======================================================

        # --- 自适应梯度裁剪 + 即时降LR ---
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if total_norm.item() > 0.5:
             grad_clip_counter += 1
             # 当场把 LR 降 20%，让优化器走小步
            #  print(f"\n   -> Grad norm {total_norm.item():.2f} > 0.5, reducing LR immediately by 20%...")
             current_lr = 0.0
             for param_group in optimizer.param_groups:
                 param_group['lr'] *= 0.8
                 current_lr = param_group['lr'] # 获取更新后的LR
             # 同样重置 scheduler 内部状态，让它从新的LR开始
             # 注意：访问内部属性可能在未来PyTorch版本中失效，但目前是必需的
             if hasattr(scheduler, '_schedulers'): # 适用于 SequentialLR
                 # 重置内部所有调度器的状态可能比较复杂，
                 # 一个简化但有效的方法是强制更新 _last_lr
                 scheduler._last_lr = [current_lr] * len(optimizer.param_groups)
             else: # 备用方案，如果不是SequentialLR
                  scheduler._last_lr = [current_lr] * len(optimizer.param_groups)
        
        optimizer.step()

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


def evaluate(model, dataloader, loss_fn, device, num_classes):
    """
    一个完整的、适配新模型输出的评估函数。
    """
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

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

            loss = loss_fn(outputs, labels)

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
    
    return epoch_metrics, confusion_matrix

def evaluate_with_rejection(model, dataloader, loss_fn, device, num_classes, confidence_threshold=0.9):
    """
    一个带有“拒绝选项”的评估函数。
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    
    # 1. 首先，收集所有真实标签和模型预测
    with torch.no_grad():
        for batched_graph in tqdm(dataloader, desc="Evaluating with Rejection"):
            batched_graph.to(device)
            labels = batched_graph.y
            
            # a) 获取模型的原始输出 (logits)
            outputs, _ = model(batched_graph) # 假设模型返回 (logits, gate)
            
            # b) 将logits转换为概率
            probabilities = F.softmax(outputs, dim=1)
            
            # c) 找出每个样本的最高概率及其对应的预测类别
            max_probs, predicted_classes = torch.max(probabilities, dim=1)
            
            # d) 【核心决策逻辑】找出那些置信度不足的预测
            #    创建一个布尔掩码，其中True代表“需要拒绝”
            rejection_mask = max_probs < confidence_threshold
            
            # e) 将被拒绝的预测的类别索引，设置为一个新的“Unknown”索引
            #    我们用 num_classes 作为'Unknown'的索引 (0..N-1是真实类别, N是Unknown)
            predicted_classes[rejection_mask] = num_classes
            
            # f) 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())

    # 2. 构建新的、更宽的混淆矩阵
    #    行数是真实类别数，列数是真实类别数 + 1 (为'Unknown'列)
    confusion_matrix = torch.zeros(num_classes, num_classes + 1, dtype=torch.long)
    for t, p in zip(all_labels, all_predictions):
        if t < num_classes: # 确保真实标签在范围内
            confusion_matrix[t, p] += 1
            
    # 3. 计算性能指标
    #    a) 计算在【未被拒绝】的样本上的性能
    accepted_mask = np.array(all_predictions) < num_classes
    accepted_labels = np.array(all_labels)[accepted_mask]
    accepted_preds = np.array(all_predictions)[accepted_mask]
    
    if len(accepted_labels) > 0:
        accepted_metrics = calculate_metrics(
            torch.tensor(
                pd.crosstab(
                    pd.Series(accepted_labels, name='Actual'), 
                    pd.Series(accepted_preds, name='Predicted')
                ).values
            )
        )
    else: # 如果所有样本都被拒绝了
        accepted_metrics = {k: 0.0 for k in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'precision_macro', 'recall_macro', 'f1_macro']}

    #    b) 计算拒绝率
    rejection_rate = (len(all_labels) - len(accepted_labels)) / len(all_labels) if len(all_labels) > 0 else 0
    
    final_metrics = {f"accepted_{k}": v for k, v in accepted_metrics.items()}
    final_metrics['rejection_rate'] = rejection_rate
    
    return final_metrics, confusion_matrix

class WarmupReduceLROnPlateau(_LRScheduler):
    """
    一个自定义的学习率调度器，结合了线性预热和ReduceLROnPlateau。
    """
    def __init__(self, optimizer, warmup_steps, target_lr, reduce_factor=0.5, reduce_patience=5, min_lr=1e-7, mode='max'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.reduce_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode=mode, 
            factor=reduce_factor, 
            patience=reduce_patience, 
            min_lr=min_lr
        )
        self.last_epoch = -1 # PyTorch scheduler convention
        super().__init__(optimizer) # 调用父类初始化

    def get_lr(self):
        # 这个方法在 PyTorch 1.x 中是必需的，但在较新版本中不是核心逻辑
        # 我们主要通过 step 方法来更新优化器
        # 返回当前的LR，用于日志记录或旧版兼容
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def step(self, metric=None, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch < self.warmup_steps:
            # --- 预热阶段 ---
            # 线性增加学习率
            lr_scale = float(epoch + 1) / float(self.warmup_steps)
            new_lr = self.target_lr * lr_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # --- 高原衰减阶段 ---
            # 将监控指标传递给内部的ReduceLROnPlateau
            if metric is None:
                 warnings.warn("WarmupReduceLROnPlateau requires a metric during the ReduceLROnPlateau phase.", UserWarning)
            else:
                 self.reduce_scheduler.step(metric)

# =====================================================================
if __name__ == '__main__':
    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    # LEARNING_RATE = 3e-4
    # LEARNING_RATE = 1e-4
    # WEIGHT_DECAY = 1e-4
    WEIGHT_DECAY = 5e-4
    # DROPOUT_RATE = 0.5
    # DROPOUT_RATE = 0.4
    NUM_WORKERS = 4 
    GNN_INPUT_DIM = 32 
    GNN_HIDDEN_DIM = 128

    # NUM_WARMUP_EPOCHS = 5 # Warm up over the first 5 epochs

    # a) 设定目标学习率 (Cosine退火的峰值)
    LEARNING_RATE = 3e-4 
    
    # b) 【关键】初始使用极小的权重衰减
    INITIAL_WEIGHT_DECAY = 1e-5
    LATER_WEIGHT_DECAY = 5e-4 # 后期增加的WD
    
    # c) 保持我们认为有效的Dropout
    DROPOUT_RATE = 0.45
    
    # d) Warmup & Cosine 设置
    NUM_WARMUP_EPOCHS = 5
    
    # e) 回滚/早停参数
    ROLLBACK_PATIENCE = 10 # 连续10个epoch F1下降则回滚

    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    dataset_name = 'ISCX-VPN'
    root_path = os.path.join('..', 'TrafficData', 'datasets_csv_add1')
    val_test_dir = os.path.join(root_path, 'datasets_split', dataset_name) 
    train_dir = os.path.join(root_path, 'datasets_final')
    vocab_dir = os.path.join(root_path, 'categorical_vocabs')
    config_path = os.path.join('.', 'Data', 'fields_embedding_configs_v1.yaml')
    # vocab_path = os.path.join('.', 'Data', 'completed_categorical_vocabs.yaml') 
    # vocab_path = os.path.join('.', 'Data', 'Test','completed_categorical_vocabs.yaml') 
    vocab_path = os.path.join(vocab_dir, dataset_name + '_vocabs.yaml') 
    res_path = os.path.join('..', 'Res')
    train_set_name = dataset_name + '_chief_block_augmented'
    # train_set_name = 'chief_block_dataset_20_d2' 
    # train_set_name = 'chief_block_v2' # 特征合并
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


    # c) 创建全局标签映射
    #    为了确保所有数据集的标签一致，我们基于训练集来创建映射
    print("\n[3/4] Creating label mapping...")
    labels = train_df['label'].unique()
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

    # c) 实例化 PyG 的 DataLoader (使用默认collate，无需自定义)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,)
    
    # --- 5. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    field_embedder = FieldEmbedding(config_path, vocab_path)
    field_embedder.to(device)

    pta_model = ProtocolTreeGAttention(
        field_embedder=field_embedder,
        num_classes=num_classes,
        node_fields_list=node_fields_for_model,
        hidden_dim=GNN_HIDDEN_DIM, 
        dropout_rate=DROPOUT_RATE # change to 0.5 to against overfit
    ).to(device)

    # ==================== 核心修改点 3：恢复类权重 ====================
    print("Calculating class weights...")
    class_counts = train_df['label_id'].value_counts().sort_index().values
    # 避免除以零，给极小计数加1
    class_counts = np.maximum(class_counts, 1) 
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes 
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # =================================================================
    
    # loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add weight_decay
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7) 

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     # mode='min',      # 监控验证集损失
    #     mode='max', # 改成val_macrof1
    #     factor=0.8,      # 每次衰减一半
    #     patience=5,      # 容忍5个epoch没有改进
    #     min_lr=1e-4      # 设置一个最小学习率
    # )

    # scheduler = WarmupReduceLROnPlateau(
    #     optimizer,
    #     warmup_steps=NUM_WARMUP_EPOCHS,
    #     target_lr=LEARNING_RATE, # 告知调度器预热的目标LR
    #     mode='max',
    #     reduce_patience=5,
    #     reduce_factor=0.5,
    #     min_lr=1e-5
    # )

    # f) 【关键】创建 Warmup + Cosine Annealing 调度器
    warm_up_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=NUM_WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - NUM_WARMUP_EPOCHS, eta_min=1e-6)
    
    scheduler = SequentialLR(optimizer, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[NUM_WARMUP_EPOCHS])

    # --- 训练循环 ---

    DIAGNOSE = False

    # --- 4. 训练循环 ---
    if not DIAGNOSE: 
        # training_results = []
        best_val_f1_macro = 0.0 # 监控Macro F1
        best_epoch = -1
        best_model_state = None # 用于保存最佳状态
        epochs_since_best = 0 # 回滚计数器
        wd_increased = False # 标记是否已增加WD
        training_results = []
        # best_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            train_metrics, _ = train_one_epoch(pta_model, train_loader, loss_fn, optimizer, scheduler, device, num_classes)
            val_metrics, _ = evaluate(pta_model, val_loader, loss_fn, device, num_classes)
            scheduler.step()
            # scheduler.step(val_metrics['loss']) 
            # scheduler.step(val_metrics['f1_macro']) 
            # scheduler.step(metric=val_metrics['f1_macro'], epoch=epoch)
            current_val_f1_macro = val_metrics['f1_macro']
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1 (Weighted): {train_metrics['f1_weighted']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Summary (LR: {current_lr:.1e}):")

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

        # # ==================== 回滚 + 两段式WD 逻辑 ====================
        # if current_val_f1_macro > best_val_f1_macro:
        #     print(f" -> Validation Macro F1 improved from {best_val_f1_macro:.4f} to {current_val_f1_macro:.4f}. Saving state...")
        #     best_val_f1_macro = current_val_f1_macro
        #     best_epoch = epoch + 1
        #     # 使用深拷贝将当前最佳模型状态保存到内存
        #     best_model_state = copy.deepcopy(pta_model.state_dict())
        #     epochs_since_best = 0
        #     # (可选) 也可以在这里同时保存到文件，作为双保险
        #     torch.save(best_model_state, os.path.join(res_path, train_set_name + '_best_model.pth'))
        # else:
        #     epochs_since_best += 1
        #     print(f" -> Validation Macro F1 did not improve for {epochs_since_best} epoch(s). Best was {best_val_f1_macro:.4f} at epoch {best_epoch}.")
            
        #     # --- 两段式WD ---
        #     if not wd_increased and epochs_since_best >= 3: # 连续3个epoch没提升，增加WD
        #         print("   -> Increasing weight decay to LATER_WEIGHT_DECAY...")
        #         for param_group in optimizer.param_groups:
        #             param_group['weight_decay'] = LATER_WEIGHT_DECAY
        #         wd_increased = True # 确保只增加一次

        #     # --- 回滚逻辑 ---
        #     if epochs_since_best >= ROLLBACK_PATIENCE:
        #         print(f"\n!!! Performance degraded for {ROLLBACK_PATIENCE} epochs. Rolling back to the best model state from epoch {best_epoch} !!!")
        #         if best_model_state:
        #             pta_model.load_state_dict(best_model_state)
        #             # 强制将学习率再减半 (在当前Cosine值的基础上)
        #             print("   -> Aggressively reducing current learning rate by half...")
        #             for param_group in optimizer.param_groups:
        #                 param_group['lr'] *= 0.5
        #             epochs_since_best = 0 # 重置计数器，给模型在新LR下再次尝试的机会
        #             # (可选) 也可以在这里重置WD回初始值
        #             # for param_group in optimizer.param_groups:
        #             #     param_group['weight_decay'] = INITIAL_WEIGHT_DECAY
        #             # wd_increased = False
        #         else:
        #             print("   -> Warning: No best model state found to roll back to. Continuing...")
        #             # 如果没有保存过最佳状态（例如训练一开始就下降），则不回滚
        #             # 也可以选择在这里直接 break 终止训练
        #             # print("   -> Stopping training early.")
        #             # break 
        # # ==============================================================
            # --- 回滚 + 两段式WD 逻辑 (现在基于Macro F1) ---
            if current_val_f1_macro > best_val_f1_macro:
                print(f" -> Validation Macro F1 improved from {best_val_f1_macro:.4f} to {current_val_f1_macro:.4f}. Saving state...")
                best_val_f1_macro = current_val_f1_macro
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(pta_model.state_dict()) 
                epochs_since_best = 0
                # (可选保存到文件)
                # torch.save(best_model_state, os.path.join(res_path, train_set_name + '_best_model.pth'))
            else:
                epochs_since_best += 1
                print(f" -> Validation Macro F1 did not improve for {epochs_since_best} epoch(s). Best was {best_val_f1_macro:.4f} at epoch {best_epoch}.")

                # --- 两段式WD ---
                if not wd_increased and epochs_since_best >= 3: 
                    print("   -> Increasing weight decay to LATER_WEIGHT_DECAY...")
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] = LATER_WEIGHT_DECAY
                    wd_increased = True 

                # --- 回滚逻辑 ---
                if epochs_since_best >= ROLLBACK_PATIENCE:
                    print(f"\n!!! Rolling back to the best model state from epoch {best_epoch} !!!")
                    if best_model_state:
                        pta_model.load_state_dict(best_model_state)
                        # ==================== 核心修改点 1 (应用) ====================
                        print("   -> Aggressively reducing current learning rate by half...")
                        new_lr = optimizer.param_groups[0]['lr'] * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        # 关键：重置 scheduler 内部计数器/状态，让它接受新LR
                        # 这依赖于具体实现，对于SequentialLR可能需要更复杂的处理
                        # 尝试更新_last_lr，并祈祷它有效
                        scheduler._last_lr = [new_lr] * len(optimizer.param_groups)
                        # 如果上面无效，可能需要重新创建cosine_scheduler部分
                        # ==========================================================
                        epochs_since_best = 0 
                    else:
                        print("   -> Warning: No best model state found to roll back to. Stopping training.")
                        break # 如果从未保存过最佳状态就触发回滚，直接停止

            # 这里可以添加保存最佳模型的逻辑
            # if val_metrics['f1_macro'] > best_f1:
            #     torch.save(pta_model.state_dict(), os.path.join(res_path, train_set_name + '_best_model.pth'))
            #     print("The best epoch parameters has been saved. ")
            #     best_f1 = val_metrics['f1_macro']
        print("\nTraining complete!")

        # --- 7. 分析特征重要性 ---
        # 【关键】使用训练结束时的模型状态来分析，或者加载最佳状态
        if best_model_state:
            print("\nLoading best model state for feature importance analysis...")
            pta_model.load_state_dict(best_model_state)
        else:
             print("\nWarning: No best model state saved. Analyzing final model state.")

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

        # # --- 5. 最终测试 ---
        print("\nLoading best model state for final testing...")
        if best_model_state:
            # pta_model.load_state_dict(torch.load(os.path.join(res_path,train_set_name + '_best_model.pth')))
            pta_model.load_state_dict(best_model_state)
            pta_model.to(device)
            pta_model.eval()
            # test_metrics, test_confusion_matrix = evaluate(pta_model, test_loader, loss_fn, device, num_classes)
            # print(f"\nFinal Test Performance:")
            # print(f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")

            # ==================== 核心修改点：调用新的评估函数 ====================

            # a) 定义您的置信度阈值 (这是一个可以调整的超参数)
            CONFIDENCE_THRESHOLD = 0.4

            # b) 调用带有拒绝选项的评估函数
            test_metrics, test_confusion_matrix = evaluate_with_rejection(
                pta_model, test_loader, loss_fn, device, num_classes, 
                confidence_threshold=CONFIDENCE_THRESHOLD
            )

            print(f"\nFinal Test Performance (with Rejection Threshold = {CONFIDENCE_THRESHOLD}):")
            print(f"  - Rejection Rate: {test_metrics['rejection_rate']:.4f}")
            print(f"  - Accuracy (on Accepted): {test_metrics['accepted_accuracy']:.4f}")
            print(f"  - F1-Score (Weighted, on Accepted): {test_metrics['accepted_f1_weighted']:.4f}")
            print(f"  - F1-Score (Macro, on Accepted): {test_metrics['accepted_f1_macro']:.4f}")
            # --- 7. 保存混淆矩阵到CSV ---
            print("\nSaving confusion matrix...")

            # # a) 创建从整数索引回字符串标签的映射
            # #    我们需要之前创建的 label_to_int 字典
            # int_to_label = {i: label for label, i in label_to_int.items()}
            # class_names = [int_to_label[i] for i in range(num_classes)]

            # # b) 将PyTorch Tensor转换为带标签的Pandas DataFrame
            # confusion_matrix_df = pd.DataFrame(
            #     test_confusion_matrix.cpu().numpy(), # 必须先移回CPU
            #     index=class_names,
            #     columns=class_names
            # )

            # --- 7. 保存新的、更宽的混淆矩阵 ---
            print("\nSaving confusion matrix with 'Unknown' column...")

            int_to_label = {i: label for label, i in label_to_int.items()}
            class_names = [int_to_label[i] for i in range(num_classes)]

            # 列名现在需要增加一个'Predicted_Unknown'
            column_names = class_names + ['Predicted_Unknown']

            confusion_matrix_df = pd.DataFrame(
                test_confusion_matrix.cpu().numpy(),
                index=class_names,      # 行名是真实类别
                columns=column_names    # 列名是预测类别 + Unknown
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
                # 'val_loss': test_metrics['loss'], 
                'rejection_rate': test_metrics.get('rejection_rate'),
                'val_acc': test_metrics['accepted_accuracy'], 
                'val_recall_macro': test_metrics['accepted_recall_macro'], 
                'val_precision_macro': test_metrics['accepted_precision_macro'], 
                'val_f1_macro': test_metrics['accepted_f1_macro'], 
                'val_recall_weighted': test_metrics['accepted_recall_weighted'], 
                'val_precision_weighted': test_metrics['accepted_precision_weighted'], 
                'val_f1_weighted': test_metrics['accepted_f1_weighted']
            })


            results_df = pd.DataFrame(training_results)
            results_df.to_csv(os.path.join(res_path,train_set_name + '_training_log.csv'), index=False)
            print(f"\nTraining log saved to {train_set_name}_training_log.csv")
        else: 
            print("No best model saved. Skipping final test.")

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