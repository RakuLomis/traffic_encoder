import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    一个健壮的、实现了Focal Loss的类。
    """
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        """
        :param alpha: (Tensor) 形状为 [C] 的类别权重。
        :param gamma: (float) 聚焦参数, gamma > 0。
        :param reduction: (str) 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: (Tensor) 模型的原始输出logits, 形状 [B, C]
        :param targets: (Tensor) 真实标签, 形状 [B]
        """
        # 首先，计算标准的交叉熵损失（但不要取mean）
        # log_softmax(inputs) 提供了数值稳定性
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 将logits转换为概率
        pt = torch.exp(-ce_loss) # pt 是模型对【正确类别】的预测概率
        
        # 计算Focal Loss
        # (1-pt)^gamma 是关键的调制因子
        focal_loss = (1 - pt)**self.gamma * ce_loss

        # 如果提供了alpha（类别权重），则应用它
        if self.alpha is not None:
            # 确保alpha在正确的设备上
            self.alpha = self.alpha.to(inputs.device)
            # 根据真实标签，为每个样本获取其对应的alpha权重
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 应用最终的reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss