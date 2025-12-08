import torch
import torch.nn as nn


class ListMLELoss(nn.Module):
    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, preds, labels):
        """
        :param preds: Tensor of predicted scores, shape (batch_size, list_size)
        :param labels: Tensor of true labels (ranks), shape (batch_size, list_size)
        :return: Computed ListMLE loss
        """
        # 对 labels 进行排序，获得排序后的索引
        sorted_indices = torch.argsort(labels, dim=1, descending=True)

        # 按照排序后的索引对 preds 进行排序
        sorted_preds = torch.gather(preds, 1, sorted_indices)

        # 计算 ListMLE 损失
        loss = 0.0
        for i in range(sorted_preds.size(1)):
            log_sum_exp = torch.logsumexp(sorted_preds[:, i:], dim=1)
            loss += (log_sum_exp - sorted_preds[:, i]).sum()

        return loss / preds.size(0)
    
    
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, preds, targets):
        """
        :param preds: Tensor of predicted values, shape (batch_size, ...)
        :param targets: Tensor of true values, shape (batch_size, ...)
        :param weights: Tensor of weights for each sample, shape (batch_size, ...)
        :return: Computed weighted MSE loss
        """
        # 计算加权的 MSE 损失
        loss = targets * (preds - targets) ** 2
        return loss.mean()