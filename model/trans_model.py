import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class NonNegativeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NonNegativeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features).abs())  # Initialize weights as non-negative
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Apply non-negative constraint
        self.weight.data.clamp_(0)
        return F.linear(x, self.weight, self.bias)

class InverseSigmoid(nn.Module):
    def __init__(self):
        super(InverseSigmoid, self).__init__()

    def forward(self, x):
        # 确保输入x在(0, 1)范围内
        eps = 1e-6  # 防止除以0或对0取对数
        x = torch.clamp(x, eps, 1 - eps)
        
        # 计算sigmoid的逆
        return torch.log(x / (1 - x))


class InverseTanh(nn.Module):
    def __init__(self):
        super(InverseTanh, self).__init__()

    def forward(self, x):
        # 确保输入x在(-1, 1)范围内
        eps = 1e-6  # 防止除以0或对0取对数
        x = torch.clamp(x, -1 + eps, 1 - eps)
        
        # 计算tanh的逆
        return 0.5 * torch.log((1 + x) / (1 - x))


class TransModel_inverse(torch.nn.Module):
    def __init__(self, hidden_size, dropout, a=0.05):
        super().__init__()
        self.W4 = nn.Parameter(torch.randn(1, 1).abs())
        self.a = nn.Parameter(torch.randn(1, 1).abs())
    
    def forward(self, x):
        self.W4.data.clamp_(0)
        x = F.tanh((x @ self.W4) * torch.sigmoid(self.a))
        return x.squeeze(1)


class Use_inverse_model():
    def __init__(self, model):
        self.model = model
        # 逆线性变换
        self.W4_inv = torch.pinverse(model.W4)
        self.a = model.a

        self.W4_inv.data.clamp_(0)
        
    
    def inverse_sigmoid(self, y):
        eps = 1e-6
        y = torch.clamp(y, eps, 1 - eps)
        return torch.log(y / (1 - y))
    
    def inverse_tanh(self, y):
        eps = 1e-6
        y = torch.clamp(y, -1 + eps, 1 - eps)
        return (0.5/torch.sigmoid(self.a))* torch.log((1 + y) / (1 - y))

    # 逆LeakyReLU函数
    def inverse_leaky_relu(self, y, alpha=0.01):
        return torch.where(y >= 0, y, y / alpha)

    
    def cal_inverse(self, Y):
        # 逆向推导
        Y_inv = self.inverse_tanh(Y)
        X_approx = (Y_inv @ self.W4_inv)
        return X_approx.detach()
