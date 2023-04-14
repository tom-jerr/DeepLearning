# -*- coding: utf-8 -*-



import torch
import torch.nn as nn

from smooth_l1_loss import SmoothL1Loss


class MultiTaskLoss(nn.Module):

    def __init__(self, lam=1):
        super(MultiTaskLoss, self).__init__()
        #随机值lamda
        self.lam = lam
        # L_cls使用交叉熵损失
        self.cls = nn.CrossEntropyLoss()
        # L_loc自定义
        self.loc = SmoothL1Loss()

    def forward(self, scores, preds, targets):
        """
        计算多任务损失。N表示RoI数目
        :param scores: [N, C]，C表示类别数
        :param preds: [N, 4]，4表示边界框坐标x,y,w,h
        :param targets: [N]，0表示背景类
        :return:
        """
        N = targets.shape[0]
        for i in range(N):
            loss = self.cls(scores[i], targets[i])
        return loss + self.loc(scores[range(N), self.indicator(targets)],
                                                    preds[range(N), self.indicator(preds)])

    def indicator(self, cate):
        return cate != 0
