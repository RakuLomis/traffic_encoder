import torch
import torch.optim as optim 
import torch.nn as nn 
from tqdm import tqdm 
from utils.data_loader import TrafficDataset 
from torch.utils.data import Dataset, DataLoader
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree 
from models.ProtocolTreeAttention import ProtocolTreeAttention 
from utils.dataframe_tools import get_file_path 
from utils.dataframe_tools import output_csv_in_fold 
from utils.dataframe_tools import padding_or_truncating
import pandas as pd 
from sklearn.model_selection import train_test_split
import os

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() # 将模型设置为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        # 将数据移动到指定设备 (GPU or CPU)
        features = {k: v.to(device) for k, v in features.items() if hasattr(v, 'to')}
        labels = labels.to(device)

        # 1. 前向传播
        outputs = model(features)
        
        # 2. 计算损失
        loss = loss_fn(outputs, labels)
        
        # 3. 反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward()      # 计算梯度
        optimizer.step()     # 更新权重

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval() # 将模型设置为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # 在评估时，不计算梯度
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = {k: v.to(device) for k, v in features.items() if hasattr(v, 'to')}
            labels = labels.to(device)

            outputs = model(features)
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
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    


    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    
    config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml') 
    csv_name = '0' 
    raw_df_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness') 
    block_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label', 'discrete') 
    # raw_df_path = os.path.join(raw_df_directory, csv_name + '.csv') 
    raw_df_path = os.path.join(block_directory, csv_name + '.csv') 

    # --- 数据分割 ---
    print("正在分割数据集...")
    # 假设 block_0_df 是您从 '0.csv' 加载的完整DataFrame
    block_0_df = pd.read_csv(raw_df_path, low_memory=False) 

    # 首先，创建一个从字符串标签到整数的映射，这对模型至关重要
    # {'aimchat': 0, 'amazon': 1, ...}
    labels = block_0_df['label'].unique()
    label_to_int = {label: i for i, label in enumerate(labels)}
    # 将字符串标签列转换为整数标签列
    block_0_df['label_id'] = block_0_df['label'].map(label_to_int)


    # 第一次分割：从总数据中分出训练集和临时集（包含验证+测试）
    train_df, temp_df = train_test_split(
        block_0_df,
        test_size=0.3,       # 30%的数据用于验证和测试
        random_state=42,     # 保证每次分割结果都一样
        stratify=block_0_df['label_id'] # 保证训练集和测试集中各标签比例相似
    )

    # 第二次分割：从临时集中分出验证集和测试集
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,       # 将临时集对半分
        random_state=42,
        stratify=temp_df['label_id']
    )

    print(f"数据集分割完成:")
    print(f" - 训练集: {len(train_df)} 条")
    print(f" - 验证集: {len(val_df)} 条")
    print(f" - 测试集: {len(test_df)} 条")

    train_dataset = TrafficDataset(train_df, config_path, vocab_path)
    val_dataset = TrafficDataset(val_df, config_path, vocab_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 3. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 假设protocol_tree和label_to_int已经准备好
    ptree = protocol_tree(train_df.columns.tolist())
    num_classes = len(label_to_int)

    field_embedder = FieldEmbedding(config_path, vocab_path)
    pta_model = ProtocolTreeAttention(field_embedder, ptree, num_classes=num_classes).to(device)
    
    loss_fn = nn.CrossEntropyLoss() # 适用于多分类的标准损失函数
    optimizer = optim.AdamW(pta_model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练循环 ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(pta_model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(pta_model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        
        # 这里可以添加保存最佳模型的逻辑
        # if val_acc > best_val_acc:
        #     torch.save(pta_model.state_dict(), 'best_model.pth')
        #     best_val_acc = val_acc

    print("\nTraining complete!")
    # --- 5. 最终测试 ---
    test_dataset = TrafficDataset(test_df, config_path, vocab_path)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    # pta_model.load_state_dict(torch.load('best_model.pth')) # 加载最佳模型
    final_model = ProtocolTreeAttention(field_embedder, ptree, num_classes=num_classes).to(device)
    test_loss, test_acc = evaluate(final_model, test_loader, loss_fn, device)
    print(f"\nFinal Test Performance:")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")