#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：NLPcode 
@File ：torch_metrics.py
@Author ：xiao zhang
@Date ：2021/8/4 18:41 
'''

import torch


def metrics(m, y, pred, sequence_lengths):
    y_mask = torch.ne(y, 0).type(torch.uint8)
    total_y_labels = torch.sum(y_mask)

    pred = torch.tensor(pred)
    pred_mask = torch.ne(pred, 0).type(torch.uint8)
    total_pred_lables = 0
    for i, l in enumerate(sequence_lengths):
        without_pad = pred_mask[i, :l]
        total_pred_lables += torch.sum(without_pad)

    correct_labels = torch.eq(y, pred).type(torch.uint8)
    correct_labels = torch.sum(correct_labels * y_mask)
    if m == 'recall':
        return correct_labels / float(total_y_labels)
    else:
        return correct_labels / float(total_pred_lables)
