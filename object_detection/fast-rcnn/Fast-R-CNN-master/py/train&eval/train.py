#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/6 12:49
# @Author  : lzy
# @File    : train.py
from selectivesearch import get_selective_search, config, get_rects

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as Data

import vgg16_roi
from data.custom_finetune_dataset import CustomFinetuneDataset
from multi_task_loss import MultiTaskLoss


def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model: object, optimizer: object, data_loader: object, device: object) -> object:
    model.to(device)
    model.train()
    torch.set_num_threads(1)
    for i,(image,target,rect) in enumerate(data_loader):
        '''
            选择搜索框
        '''
        # img = image[0]
        # img = (img * 0.5 + 0.5) * 255
        # img = img.numpy()
        # img = np.transpose(img, (1, 2, 0))
        # gs = get_selective_search()
        # config(gs, img, strategy='q')
        # rects = get_rects(gs)
        # imag = img.copy()
        # for i, rect in enumerate(rects):
        #     if (i < 100):
        #         x, y, w, h = rect
        #         cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        #     else:
        #         break

        rects = image.to(device)
        target = target.to(device)

        classify,regression = model(rects)

        criterion = MultiTaskLoss(lam=1)
        loss = criterion(classify, regression, target)

        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
        loss_record.append(loss.item())







if __name__ == "__main__":
    ##全局变量
    loss_record = []

    root_dir = '../../py/data/finetune_car/train'
    s = 600
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(s),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_set = CustomFinetuneDataset(root_dir, transform)

    data_loader = Data.DataLoader(data_set,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = vgg16_roi.VGG16_RoI()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    train_one_epoch(model,optimizer,data_loader,device)
    print(loss_record)

