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


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() # 将模型设置为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        # 将数据移动到指定设备 (GPU or CPU)
        # features = {k: v.to(device) for k, v in features.items() if hasattr(v, 'to')}
        # labels = labels.to(device)

        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        labels = labels.to(device, non_blocking=True)

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
            # features = {k: v.to(device) for k, v in features.items() if hasattr(v, 'to')}
            # labels = labels.to(device)
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = labels.to(device, non_blocking=True)

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
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4

    # --- 2. 准备数据 ---
    # 假设 train_df, val_df, test_df 已经创建好
    
    config_path = os.path.join('.', 'utils', 'fields_embedding_configs_v1.yaml')
    vocab_path = os.path.join('.', 'Data', 'Test', 'completed_categorical_vocabs.yaml') 
    csv_name = '0' 
    raw_df_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness') 
    # block_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label', 'discrete') 
    block_directory = os.path.join('..', 'TrafficData', 'dataset_29_d1_csv_merged', 'completeness', 'dataset_29_completed_label', 'test')
    # raw_df_path = os.path.join(raw_df_directory, csv_name + '.csv') 
    raw_df_path = os.path.join(block_directory, csv_name + '.csv') 

    train_loaders = {} 
    val_loaders = {}
    block_label_nums = {}
    eligible_blocks = []
    # test_flag = 0
    for block_filename in tqdm(os.listdir(block_directory), desc="Building dataloader for each block. "): 
        block_name = os.path.splitext(block_filename)[0] 
        block_path = os.path.join(block_directory, block_filename) 
        block_df = pd.read_csv(block_path, low_memory=False, dtype=str) 

        label_counts = block_df['label'].value_counts() 
        min_samples_per_class = 4
        valid_labels = label_counts[label_counts >= min_samples_per_class].index 

        if len(valid_labels) < 2: # 如果有效类别少于2个，无法进行分类和分割
            print(f"\nBlock {block_name} 过滤后有效类别少于2个，跳过此Block。")
            continue

        original_rows = len(block_df)
        block_df = block_df[block_df['label'].isin(valid_labels)]
        # print(f"过滤掉 {original_rows - len(block_df)} 行，因为它们的label样本数少于 {min_samples_per_class}。")
        # print(f"开始后，数据总行数: {len(block_df)}")


        labels = block_df['label'].unique() 
        label_to_int = {label: i for i, label in enumerate(labels)} 
        block_df['label_id'] = block_df['label'].map(label_to_int) 
        print(f"Block {block_name}'s label has been transformed. ") 
        # 第一次分割：从总数据中分出训练集和临时集（包含验证+测试）
        train_df, temp_df = train_test_split(
            block_df,
            test_size=0.3,       # 30%的数据用于验证和测试
            random_state=42,     # 保证每次分割结果都一样
            stratify=block_df['label_id'] # 保证训练集和测试集中各标签比例相似
        )
        temp_label_counts = temp_df['label_id'].value_counts()
        if temp_label_counts.min() < 2:
            print(f"\nBlock {block_name} 在第一次分割后，部分类别的样本数少于2，无法进行第二次分割，跳过此Block。")
            # 跳过当前循环，处理下一个Block
            continue
        # 第二次分割：从临时集中分出验证集和测试集
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,       # 将临时集对半分
            random_state=42,
            stratify=temp_df['label_id']
        )

        print(f"数据集{block_name}分割完成:")
        print(f" - 训练集: {len(train_df)} 条")
        print(f" - 验证集: {len(val_df)} 条")
        print(f" - 测试集: {len(test_df)} 条")

        block_label_nums[block_name] = len(labels) 
        del block_df, temp_df 

        train_dataset = TrafficDataset(train_df, config_path, vocab_path)
        # val_dataset = TrafficDataset(val_df, config_path, vocab_path)
        train_loaders[block_name] = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4, # 保持多进程
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        eligible_blocks.append(block_filename)

        # test_flag += 1
        # if test_flag >= 3: 
        #     print(f"先跑{test_flag}个看看")
        #     break
    
    # --- 3. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = max(block_label_nums.values())

    moe_pta_model = MoEPTA(
        block_directory=block_directory,
        eligible_blocks=eligible_blocks,
        config_path=config_path,
        vocab_path=vocab_path,
        block_num_classes=block_label_nums, 
        num_classes=num_classes
    )
    moe_pta_model.to(device)
    
    optimizer = optim.AdamW(moe_pta_model.parameters(), lr=LEARNING_RATE) 
    loss_fn = nn.CrossEntropyLoss() 

    for block_name, loader in train_loaders.items(): 
        for features, labels in loader: 
            # 构造一个只包含当前Block数据的批次
                batch_for_model = {block_name: features}
                
                # 将数据移动到GPU
                batch_for_model = {k: {fname: fval.to(device) for fname, fval in fdict.items()} for k, fdict in batch_for_model.items()}
                labels = labels.to(device)
                
                # 执行前向和反向传播
                outputs = moe_pta_model(batch_for_model)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
