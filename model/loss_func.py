from turtle import forward
import torch
import numpy as np
import time
from torch import nn

class MSE_rel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rel_score, usr_score, wt, duration) :
        rel_score = torch.sigmoid(rel_score)
        wt_pred_raw = usr_score * torch.log((1+rel_score)/(1-rel_score+1e-3))

        wt_pred_raw = wt_pred_raw * (wt_pred_raw>0)
        wt_pred_mask = wt_pred_raw * (wt_pred_raw<duration)
        duration_mask = duration * (wt_pred_raw>=duration)
        wt_pred_clip = wt_pred_mask + duration_mask
        # print(wt_pred_raw)
        return (torch.square(wt_pred_clip - wt)).mean()
        # return (torch.square(wt_pred_raw - wt)).mean()

class MSE_jump(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, rel_score, wt, duration, eps):
        # JUMP
        wt_true_less = torch.log(1+wt)[wt<duration]
        wt_pred_less = rel_score[wt<duration]
        wt_true_over = torch.log(1+wt)[wt>=duration]
        wt_pred_over = rel_score[wt>=duration]
        logsigmoid = nn.LogSigmoid()
        loss_less = (torch.square(wt_pred_less - wt_true_less)/(2*(eps**2))).mean()
        loss_over = -logsigmoid(1.6*(wt_pred_over - wt_true_over)/eps).mean()
        return loss_less + loss_over


class MSE_usr(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, wt_pred, wt, duration, eps):
        wt_true_less = wt[wt<duration]
        wt_pred_less = wt_pred[wt<duration]
        wt_true_over = wt[wt>=duration]
        wt_pred_over = wt_pred[wt>=duration]
        logsigmoid = nn.LogSigmoid()
        loss_less = (torch.square(wt_pred_less - wt_true_less)/(2*(eps**2))).mean()
        loss_over = -logsigmoid(1.6*(wt_pred_over - wt_true_over)/eps).mean()
        return loss_less + loss_over


class MSE_rel3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rel_score, rel_label, wt, duration, eps):
        rel_true_less = rel_label[wt<duration]
        rel_score_less = rel_score[wt<duration]
        rel_true_over = rel_label[wt>=duration]
        rel_score_over = rel_score[wt>=duration]
        logsigmoid = nn.LogSigmoid()
        def inv_sigmoid(y):
            d = 1e-6
            y = torch.clamp(y, d, 1 - d)
            x = torch.log(y / (1 - y))
            return x
        loss_less = (torch.square(rel_score_less- inv_sigmoid(rel_true_less))/(2*(eps**2))).mean()
        loss_over = -logsigmoid((rel_score_over - inv_sigmoid(rel_true_over))/eps).mean()
        return loss_less + loss_over 



class MSE_rel2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rel_score, usr_score, wt, duration, eps):
        rel_true_less = torch.exp(-usr_score/(wt+1))[wt<duration]
        rel_score_less = rel_score[wt<duration]
        rel_true_over = torch.exp(-usr_score/(wt+1))[wt>=duration]
        rel_score_over = rel_score[wt>=duration]
        logsigmoid = nn.LogSigmoid()
        def inv_sigmoid(y):
            d = 1e-6
            y = torch.clamp(y, d, 1 - d)
            x = torch.log(y / (1 - y))
            return x
        loss_less = (torch.square(rel_score_less- inv_sigmoid(rel_true_less))/(2*(eps**2))).mean()
        loss_over = -logsigmoid((rel_score_over - inv_sigmoid(rel_true_over))/eps).mean()

        return loss_less + loss_over
