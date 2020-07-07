# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:29:33 2018

@author: akash
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from argLibrary import *

from functions import *

class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        #self.hardtanh = nn.Hardtanh()
        # HACK - override tanh with ReLU
        self.relu = nn.ReLU()
        
    def forward(self, input):
        #output = self.hardtanh(input)
        output = self.relu(input)
        if args.bin_acts:
            output = binarize(output)
        return output
        

class BinaryLinear(nn.Linear):
    def forward(self, input):
        if args.bin_weights:
            weight = binarize(self.weight)
        else:
            weight = self.weight
        if self.bias is None:
            return F.linear(input, weight)
        else:
            return F.linear(input, weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv



class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        if args.bin_weights:
            weight = binarize(self.weight)
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
