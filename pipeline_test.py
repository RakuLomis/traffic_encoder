import pandas as pd 
from sklearn.model_selection import train_test_split
import os

csv_name = '2' 
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