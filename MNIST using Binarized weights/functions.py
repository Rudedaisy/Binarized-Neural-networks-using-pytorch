# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:30:49 2018

@author: akash
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from argLibrary import *

class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input > 0] = 1
        if args.bin_zero:
            output[input <= 0] = 0
        else:
            output[input <= 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

"""
def BinZeroRegularizer(model, loss):
    reg_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            #print(m.weight)
            temp_loss = torch.sum(torch.abs(m.weight[m.weight < 0.5])) + torch.sum(torch.abs(m.weight[m.weight >= 0.5] - 1.0))
            reg_loss += 0.1*temp_loss
    loss = loss + 0.01*reg_loss
    return loss
"""

def BinZeroProxUpdate():
    global model
    #W[W > LR*reg] = W[W > LR*reg] - (LR*reg)
    #W[np.abs(W) <= LR*reg] = 0
    #W[W < (LR*reg*-1.0)] = W[W < (LR*reg*-1.0)] + (LR*reg)
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # easy case: push values that exist between [0,1]
            m.weight[m.weight < 0.5 and m.weight >= 0] = 0
            m.weight[m.weight >= 0.5 and m.weight <= 1] = 1
            # now apply standard clipped proximity update
            m.weight[torch.abs(m.weight) <= LR*reg and m.weight != 0 and m.weight != 1] = 0
            m.weight[torch.abs(m.weight - 1.0) <= LR*reg and m.weight != 0 and m.weight != 1] = 1
            m.weight[m.weight < (LR*reg*-1.0) and m.weight != 0 and m.weight != 1] = m.weight[m.weight < (LR*reg*-1.0) and m.weight != 0 and m.weight != 1] + (LR*reg)
            m.weight[m.weight > (LR*reg) and m.weight != 0 and m.weight != 1] = m.weight[m.weight > (LR*reg) and m.weight != 0 and m.weight != 1] - (LR*reg)
            
# aliases
binarize = BinarizeF.apply
