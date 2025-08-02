import torch
import torch.optim as optim 
import torch.nn as nn 
from tqdm import tqdm 
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
import os

# --- 导入所有我们自定义的、重构后的模块 ---
from utils.data_loader import TrafficDataset, custom_collate_fn
from utils.collator import PTACollator # 假设您已将PTACollator保存到此文件
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention # 导入重构后的PTA模型

# train_one_epoch 和 evaluate 函数与您之前的版本完全相同，无需修改
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for features_batch, labels in tqdm(dataloader, desc="Training"):
        features_batch = {k: v.to(device, non_blocking=True) for k, v in features_batch.items() if isinstance(v, torch.Tensor)}
        labels = labels.to(device, non_blocking=True)
        outputs = model(features_batch)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for features_batch, labels in tqdm(dataloader, desc="Evaluating"):
            features_batch = {k: v.to(device, non_blocking=True) for k, v in features_batch.items() if isinstance(v, torch.Tensor)}
            labels = labels.to(device, non_blocking=True)
            outputs = model(features_batch)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 10
    BATCH_SIZE = 256 # 既然我们优化了性能，可以尝试更大的批次大小
    LEARNING_RATE = 1e-4
    
    # --- 2. 准备数据路径 (无变化) ---
    config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml') 
    raw_df_path = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label', 'discrete', '0.csv')

    # --- 3. 数据分割 (无变化) ---
    print("正在分割数据集...")
    block_0_df = pd.read_csv(raw_df_path, dtype=str) 
    labels = block_0_df['label'].unique()
    label_to_int = {label: i for i, label in enumerate(labels)}
    block_0_df['label_id'] = block_0_df['label'].map(label_to_int)
    train_df, temp_df = train_test_split(block_0_df, test_size=0.3, random_state=42, stratify=block_0_df['label_id'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id'])
    print("数据集分割完成。")

    # ==================== 核心修改点：严格按照依赖顺序进行实例化 ====================

    # --- 4. 初始化所有组件 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # a) 首先实例化 FieldEmbedding
    field_embedder = FieldEmbedding(config_path, vocab_path)
    
    # b) 然后生成协议树
    ptree = protocol_tree(train_df.columns.tolist())
    num_classes = len(label_to_int)

    # c) 实例化我们全新的 PTACollator，它依赖于 field_embedder 和 ptree
    pta_collator = PTACollator(field_embedder, ptree)
    
    # d) 实例化 Dataset (无变化)
    train_dataset = TrafficDataset(train_df, config_path, vocab_path)
    val_dataset = TrafficDataset(val_df, config_path, vocab_path)
    
    # e) 实例化 DataLoader，并使用 pta_collator 作为 collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        # num_workers=8,
        pin_memory=True,
        collate_fn=pta_collator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        # num_workers=8,
        pin_memory=True,
        collate_fn=pta_collator
    )
    
    # f) 实例化我们重构后的 PTA 模型，它依赖于 pta_collator 提供的维度信息
    pta_model = ProtocolTreeAttention(
        parent_fields_list=pta_collator.parent_fields,
        simple_fields_by_layer=pta_collator.simple_fields_by_layer, # 确保Collator中有这个属性
        num_classes=num_classes
    ).to(device)
    
    # g) 初始化损失函数和优化器 (无变化)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE)

    # ==================== 修改结束 ====================

    # --- 5. 训练循环 (无变化) ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(pta_model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(pta_model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

    # --- 6. 最终测试 (同样需要使用新架构) ---
    print("\nTraining complete! Starting final test...")
    test_dataset = TrafficDataset(test_df, config_path, vocab_path)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        # num_workers=8,
        pin_memory=True,
        collate_fn=pta_collator
    )
    
    # 假设我们加载训练好的模型权重进行测试
    # pta_model.load_state_dict(torch.load('best_model.pth')) 
    test_loss, test_acc = evaluate(pta_model, test_loader, loss_fn, device)
    print(f"\nFinal Test Performance:")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")