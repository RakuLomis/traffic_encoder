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

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, num_classes): # <-- 新增 num_classes 参数
    model.train()
    running_loss = 0.0
    
    # --- 核心修改点 1：初始化混淆矩阵 ---
    # 我们在CPU上创建，以避免频繁的GPU-CPU数据传输
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batched_graph in tqdm(dataloader, desc="Training"):
        batched_graph.to(device)
        labels = batched_graph.y
        outputs = model(batched_graph)
        loss = loss_fn(outputs, labels)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batched_graph.num_graphs
        _, predicted = torch.max(outputs.data, 1)
        
        # --- 核心修改点 2：累积结果到混淆矩阵 ---
        # 将张量移回CPU进行累积
        labels_cpu = labels.cpu()
        predicted_cpu = predicted.cpu()
        for t, p in zip(labels_cpu.view(-1), predicted_cpu.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    # --- 核心修改点 3：在epoch结束后，进行一次性的性能计算 ---
    total_samples = confusion_matrix.sum().item()
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    
    # 调用我们独立的指标计算函数
    epoch_metrics = calculate_metrics(confusion_matrix)
    epoch_metrics['loss'] = epoch_loss # 将损失也加入字典
    
    return epoch_metrics, confusion_matrix 

def evaluate(model, dataloader, loss_fn, device, num_classes):
    model.eval() # 将模型设置为评估模式
    running_loss = 0.0
    
    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # 在评估时，我们完全不需要计算梯度
    with torch.no_grad(): 
        # 将tqdm的描述符修正为"Evaluating"
        for batched_graph in tqdm(dataloader, desc="Evaluating"):
            batched_graph.to(device)
            labels = batched_graph.y
            outputs = model(batched_graph)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * batched_graph.num_graphs
            _, predicted = torch.max(outputs.data, 1)

            # 累积结果到混淆矩阵
            labels_cpu = labels.cpu()
            predicted_cpu = predicted.cpu()
            for t, p in zip(labels_cpu.view(-1), predicted_cpu.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    # 在epoch结束后，进行一次性的性能计算
    total_samples = confusion_matrix.sum().item()
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    
    # 调用我们独立的指标计算函数
    epoch_metrics = calculate_metrics(confusion_matrix)
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics, confusion_matrix

# =====================================================================
if __name__ == '__main__':
    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 4 
    GNN_INPUT_DIM = 32 
    GNN_HIDDEN_DIM = 128

    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    
    config_path = os.path.join('.', 'Data', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join('.', 'Data', 'completed_categorical_vocabs.yaml') 
    train_set_name = 'chief_block' 
    val_set_name = 'validation_set' 
    test_set_name = 'test_set'
    chief_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'train_test', 'blocks')
    val_test_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'train_test')

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

    # a) 确定“主干Schema”，即模型期望的输入特征
    chief_schema = [col for col in train_df.columns if col not in ['label', 'label_id']]
    
    # b) 【关键】对验证集和测试集进行特征空间对齐
    print("\n[2/4] Aligning feature space for validation and test sets...")
    
    # 对齐验证集
    val_df_aligned = pd.DataFrame(columns=chief_schema)
    for col in chief_schema:
        if col in val_df.columns:
            val_df_aligned[col] = val_df[col]
        else:
            val_df_aligned[col] = '0' # 用'0'填充缺失的特征
    val_df_aligned['label'] = val_df['label'] # 补回标签列
    
    # 对齐测试集
    test_df_aligned = pd.DataFrame(columns=chief_schema)
    for col in chief_schema:
        if col in test_df.columns:
            test_df_aligned[col] = test_df[col]
        else:
            test_df_aligned[col] = '0'
    test_df_aligned['label'] = test_df['label']

    print(" - Feature alignment complete.")
    
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
        hidden_dim=GNN_HIDDEN_DIM
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    # --- 4. 训练循环 ---
    training_results = []
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_metrics, _ = train_one_epoch(pta_model, train_loader, loss_fn, optimizer, device, num_classes)
        val_metrics, _ = evaluate(pta_model, val_loader, loss_fn, device, num_classes)
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1 (Weighted): {train_metrics['f1_weighted']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"Epoch {epoch+1} Summary (LR: {scheduler.get_last_lr()[0]:.1e}):")

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
        if val_metrics['f1_weighted'] > best_f1:
            torch.save(pta_model.state_dict(), 'best_model.pth')
            print("The best epoch parameters has been saved. ")
            best_f1 = val_metrics['f1_weighted']
    print("\nTraining complete!")

    # --- 5. 最终测试 ---
    # test_dataset = GNNTrafficDataset(test_df, config_path, vocab_path)
    # test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,)# collate_fn=gnn_collate_fn)
    test_metrics, test_confusion_matrix = evaluate(pta_model, test_loader, loss_fn, device, num_classes)
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
    cm_output_path = 'final_test_confusion_matrix.csv'
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
    results_df.to_csv('moe_pta_training_log.csv', index=False)
    print("\nTraining log saved to 'moe_pta_training_log.csv'")