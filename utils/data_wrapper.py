from torch.utils.data import DataLoader, Dataset
import torch
import ast
import numpy as np

class Wrap_Dataset(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y) 
    

class Wrap_Dataset2(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, X_usr, duration ,use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.X_usr = torch.LongTensor(X_usr).cuda()
            self.y = torch.Tensor(y).cuda()
            self.duration = torch.Tensor(duration).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.X_usr = torch.LongTensor(X_usr).cpu()
            self.y = torch.Tensor(y).cpu()
            self.duration = torch.Tensor(duration).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index], self.X_usr[index], self.duration[index]

    def __len__(self):
        return len(self.y) 


class Wrap_Dataset3(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, weight ,use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
            self.weight = torch.Tensor(weight).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()
            self.weight = torch.Tensor(weight).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index], self.weight[index]

    def __len__(self):
        return len(self.y)

class Wrap_Dataset4(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, weight, y1, y2, use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
            self.weight = torch.Tensor(weight).cuda()
            self.y1 = torch.Tensor([ast.literal_eval(y1_) for y1_ in y1]).cuda()
            self.y2 = torch.Tensor([ast.literal_eval(y2_) for y2_ in y2]).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()
            self.weight = torch.Tensor(weight).cpu()
            self.y1 = torch.Tensor([ast.literal_eval(y1_) for y1_ in y1]).cpu()
            self.y2 = torch.Tensor([ast.literal_eval(y2_) for y2_ in y2]).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index], self.weight[index], self.y1[index], self.y2[index]

    def __len__(self):
        return len(self.y)