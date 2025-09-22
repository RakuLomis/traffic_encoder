import numpy as np
import torch 

def calculate_metrics(confusion_matrix: torch.Tensor):
    """
    根据输入的混淆矩阵，计算详细的性能指标。
    
    :param confusion_matrix: 一个 CxC 的张量，其中 C 是类别数。
                             矩阵的[i, j]元素代表真实类别为i，预测类别为j的样本数。
    :return: 一个包含所有计算指标的字典。
    """
    num_classes = confusion_matrix.shape[0]
    metrics = {}

    # 1. 计算每个类别的 TP, FP, FN
    # TP (True Positives): 对角线上的元素
    tp = confusion_matrix.diag()
    # FP (False Positives): 每一列的和 - 对角线上的元素
    fp = confusion_matrix.sum(dim=0) - tp
    # FN (False Negatives): 每一行的和 - 对角线上的元素
    fn = confusion_matrix.sum(dim=1) - tp
    
    # 2. 计算每个类别的 Precision, Recall, F1-Score
    # 使用 epsilon (一个极小值) 来避免除以零的错误
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # 3. 计算宏平均 (Macro Average)
    # 简单地对所有类别的指标取平均，平等对待每个类别
    metrics['precision_macro'] = precision.mean().item()
    metrics['recall_macro'] = recall.mean().item()
    metrics['f1_macro'] = f1.mean().item()

    # 4. 计算加权平均 (Weighted Average)
    # 按每个类别的真实样本数（支持度）进行加权平均
    support = confusion_matrix.sum(dim=1)
    total_support = support.sum()
    metrics['precision_weighted'] = (precision * support).sum().item() / total_support
    metrics['recall_weighted'] = (recall * support).sum().item() / total_support
    metrics['f1_weighted'] = (f1 * support).sum().item() / total_support

    # 5. 计算总准确率 (Overall Accuracy)
    total_correct = tp.sum().item()
    total_samples = confusion_matrix.sum().item()
    metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 6. 计算总损失 (将在主循环中计算)
    # metrics['loss'] = ...
    
    return metrics
