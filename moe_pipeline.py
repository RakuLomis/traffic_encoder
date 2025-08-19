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
from torch.profiler import profile, record_function, ProfilerActivity
from utils.data_loader import custom_collate_fn
from models.MoEPTA import MoEPTA

# def train_one_epoch(model, train_loaders_dict, loss_fn, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     # 遍历每一个Block的DataLoader进行训练
#     for block_name, loader in tqdm(train_loaders_dict.items(), desc="Training Epoch"):
#         for features, labels in loader:
#             batch_for_model = {block_name: features}
            
#             # 将数据移动到GPU
#             batch_for_model = {k: {fname: fval.to(device, non_blocking=True) for fname, fval in fdict.items()} for k, fdict in batch_for_model.items()}
#             labels = labels.to(device, non_blocking=True)
            
#             # 执行前向和反向传播
#             outputs = model(batch_for_model)
#             loss = loss_fn(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 统计损失和准确率
#             total_loss += loss.item() * len(labels)
#             _, predicted = torch.max(outputs.data, 1)
#             total_samples += len(labels)
#             total_correct += (predicted == labels).sum().item()

#     epoch_loss = total_loss / total_samples
#     epoch_acc = total_correct / total_samples
#     return epoch_loss, epoch_acc

# def evaluate_one_epoch(model, data_loaders_dict, loss_fn, device):
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     with torch.no_grad():
#         for block_name, loader in tqdm(data_loaders_dict.items(), desc="Evaluating Epoch"):
#             for features, labels in loader:
#                 batch_for_model = {block_name: features}
                
#                 batch_for_model = {k: {fname: fval.to(device, non_blocking=True) for fname, fval in fdict.items()} for k, fdict in batch_for_model.items()}
#                 labels = labels.to(device, non_blocking=True)
                
#                 outputs = model(batch_for_model)
#                 # outputs, attn_weights = model(batch_for_model) 
#                 # print(attn_weights[0, 0, 0, 1:])
#                 loss = loss_fn(outputs, labels)

#                 total_loss += loss.item() * len(labels)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_samples += len(labels)
#                 total_correct += (predicted == labels).sum().item()

#     epoch_loss = total_loss / total_samples
#     epoch_acc = total_correct / total_samples
#     return epoch_loss, epoch_acc

def train_one_epoch(model, train_loaders_dict, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # --- 核心修改点：创建并行的迭代器 ---
    # a) 为每个 a loader 创建一个迭代器
    iterators = {name: iter(loader) for name, loader in train_loaders_dict.items()}
    
    # b) 找出批次数量最多的那个 a loader，以确定本轮 epoch 的总步数
    if not train_loaders_dict: return 0.0, 0.0 # 处理没有合格block的情况
    num_steps = max(len(loader) for loader in train_loaders_dict.values())

    # c) 使用总步数来驱动主训练循环
    for _ in tqdm(range(num_steps), desc="Training Epoch"):
        optimizer.zero_grad()
        
        # --- d) 在每一步构建“混合批次” ---
        mega_batch_features = {}
        mega_batch_labels = []
        
        # 从每个 a loader 的迭代器中取出一个批次
        for block_name, it in iterators.items():
            try:
                features, labels = next(it)
                mega_batch_features[block_name] = features
                mega_batch_labels.append(labels)
            except StopIteration:
                # 如果某个 a loader 的数据用完了，就为它重新创建一个迭代器
                iterators[block_name] = iter(train_loaders_dict[block_name])
                features, labels = next(iterators[block_name])
                mega_batch_features[block_name] = features
                mega_batch_labels.append(labels)
        
        # 将混合批次的数据和标签都移动到GPU
        for block_name, features in mega_batch_features.items():
            mega_batch_features[block_name] = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        labels = torch.cat(mega_batch_labels).to(device, non_blocking=True)
        
        # --- e) 用“混合批次”进行一次完整的前向和反向传播 ---
        outputs = model(mega_batch_features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # --- f) 统计损失和准确率 (逻辑不变) ---
        total_loss += loss.item() * len(labels)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += len(labels)
        total_correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

# evaluate_one_epoch 函数也需要进行类似的修改
def evaluate_one_epoch(model, data_loaders_dict, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 同样，并行处理所有评估用的 a loader
    iterators = {name: iter(loader) for name, loader in data_loaders_dict.items()}
    if not data_loaders_dict: return 0.0, 0.0
    num_steps = max(len(loader) for loader in data_loaders_dict.values())

    with torch.no_grad():
        for _ in tqdm(range(num_steps), desc="Evaluating Epoch"):
            mega_batch_features = {}
            mega_batch_labels = []

            for block_name, it in iterators.items():
                try:
                    features, labels = next(it)
                    mega_batch_features[block_name] = features
                    mega_batch_labels.append(labels)
                except StopIteration:
                    # 在评估时，数据用完就用完，不需要重置
                    continue 
            
            if not mega_batch_features: continue

            for block_name, features in mega_batch_features.items():
                mega_batch_features[block_name] = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = torch.cat(mega_batch_labels).to(device, non_blocking=True)
            
            outputs = model(mega_batch_features)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += len(labels)
            total_correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

# ==============================================================================
# 2. 主执行脚本
# ==============================================================================

if __name__ == '__main__':
    # --- 1. 设置超参数 ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8

    # --- 2. 准备数据路径 ---
    config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml') 
    csv_name = '0' 
    raw_df_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness') 
    block_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'reborn_blocks_merge') 
    # block_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label', 'test')
    # raw_df_path = os.path.join(raw_df_directory, csv_name + '.csv') 
    raw_df_path = os.path.join(block_directory, csv_name + '.csv') 
    
    # --- 3. 为每个合格的Block创建Dataloader ---
    train_loaders = {} 
    val_loaders = {}
    test_loaders = {}
    block_label_nums = {}
    eligible_blocks = []
    
    print("开始为每个Field Block准备数据...")
    for block_filename in tqdm(os.listdir(block_directory), desc="Building DataLoaders"): 
        if not block_filename.lower().endswith('.csv'): 
            print(f"{block_filename} is not a csv file.")
            continue
        block_name = os.path.splitext(block_filename)[0]
        block_path = os.path.join(block_directory, block_filename)
        block_df = pd.read_csv(block_path, dtype=str)
        # print(f"Handling block {block_name}. ")

        # a) 过滤样本过少的类别
        label_counts = block_df['label'].value_counts()
        min_samples_per_class = 8
        valid_labels = label_counts[label_counts >= min_samples_per_class].index
        # if len(valid_labels) < 2:
        #     print(f"\nBlock {block_name} 有效类别少于2个，跳过。")
        #     continue
        block_df = block_df[block_df['label'].isin(valid_labels)]

        num_unique_labels = block_df['label'].nunique()

        if num_unique_labels < 2:
            # 如果类别少于2个，无法用于分类任务，直接跳过整个Block
            print(f"\nBlock {block_name} 过滤后有效类别少于2个，跳过。")
            continue 

        # b) 创建标签映射并分割数据
        labels = block_df['label'].unique()
        label_to_int = {label: i for i, label in enumerate(labels)}
        block_df['label_id'] = block_df['label'].map(label_to_int)
        
        if num_unique_labels == 1: 
            print(f"{block_filename} only has one class, its all data will be used to train model. ")
            train_df = block_df 
            val_df = pd.DataFrame(columns=block_df.columns)
            test_df = pd.DataFrame(columns=block_df.columns)
        else: 
            train_df, temp_df = train_test_split(block_df, test_size=0.3, random_state=42, stratify=block_df['label_id'])
            if temp_df['label_id'].value_counts().min() < 2:
                print(f"\nBlock {block_name} 无法安全分割成验证/测试集，跳过。")
                continue
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id'])

            # 尝试创建一个临时的Dataset
            temp_train_dataset = TrafficDataset(train_df, config_path, vocab_path)

            # 检查这个Dataset是否包含了任何有效的、可在config中找到的特征
            if not temp_train_dataset.fields: # .fields 列表为空
                print(f"\nBlock {block_name} 虽然样本充足，但其特征均不在config文件中，视为无效Block，跳过。")
                continue

            block_label_nums[block_name] = len(labels)
        
        # c) 创建Dataset和DataLoader
        train_dataset = TrafficDataset(train_df, config_path, vocab_path)
        val_dataset = TrafficDataset(val_df, config_path, vocab_path)
        test_dataset = TrafficDataset(test_df, config_path, vocab_path)
        
        train_loaders[block_name] = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
        val_loaders[block_name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
        test_loaders[block_name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)

        eligible_blocks.append(block_filename)

    print(f"\n数据准备完成，共 {len(eligible_blocks)} 个合格的Block。")

    # --- 4. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 确定全局类别数（所有Block中出现过的最大类别数）
    num_classes = 0
    # 我们需要一个全局的label_to_int映射，以确保所有专家输出的维度一致
    # 重新读取所有合并后的数据来创建这个全局映射
    all_labels_df = pd.concat([pd.read_csv(os.path.join(block_directory, f), dtype=str, usecols=['label']) for f in os.listdir(block_directory) if f.endswith('.csv')])
    global_labels = all_labels_df['label'].unique()
    num_classes = len(global_labels)
    print(f"全局类别数量为: {num_classes}")

    moe_pta_model = MoEPTA(
        block_directory=block_directory,
        config_path=config_path,
        vocab_path=vocab_path,
        eligible_blocks=eligible_blocks, 
        # block_num_classes=block_label_nums,
        num_classes=num_classes
    ).to(device)
    
    optimizer = optim.AdamW(moe_pta_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # --- 5. 训练循环 ---
    training_results = []
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(moe_pta_model, train_loaders, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(moe_pta_model, val_loaders, loss_fn, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        
        # 记录结果
        training_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # 保存性能最佳的模型
        if val_acc > best_val_acc:
            print(f"  Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
            torch.save(moe_pta_model.state_dict(), 'best_moe_model.pth')
            best_val_acc = val_acc

    print("\nTraining complete!")

    # --- 6. 最终测试 ---
    print("\nLoading best model for final testing...")
    moe_pta_model.load_state_dict(torch.load('best_moe_model.pth'))
    
    test_loss, test_acc = evaluate_one_epoch(moe_pta_model, test_loaders, loss_fn, device)
    
    print(f"\nFinal Test Performance:")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # 将最终测试结果也添加到记录中
    training_results.append({
        'epoch': 'final_test',
        'train_loss': None, 'train_acc': None,
        'val_loss': test_loss, 'val_acc': test_acc
    })

    # --- 7. 保存结果到CSV ---
    results_df = pd.DataFrame(training_results)
    results_df.to_csv('moe_pta_training_log.csv', index=False)
    print("\nTraining log saved to 'moe_pta_training_log.csv'")