# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:37:45 2018

@author: akash
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt
import torch.utils.data as D
import torch.optim as optim
from torch.autograd import Variable
from modules import *
from torchvision import datasets,transforms
from argLibrary import *
import numpy as np
import ctypes # allow binary representation of the FP numbers
from decimal import Decimal # quantize number to fixed-point
import sys

# to export DNN dimensions
sys.path.insert(1, '../../torch-summary/torchsummary')
from torchsummary import summary

# to save model weights
import pandas as pd

saveIFM = False
firstBatch = True

def timeSince(since):
    now = time.time()
    s = now - since
    #m = math.floor(s / 60)
    #s -= m * 60
    return s

# works with only 1d and 2d arrays
def binary(arr):
    temp = []
    #print(type(arr[0]))
    
    if np.ndim(arr) == 2:
        ndim = 2
    else:
        ndim = 1
    
    for row in range(len(arr)):
        if ndim == 2:
            temp.append([])
            for col in range(len(arr[row])):
                val = bin(ctypes.c_uint.from_buffer(ctypes.c_float(arr[row][col])).value)[2:]
                val = val.zfill(32)
                temp[row].append(val)
        else:
            val = bin(ctypes.c_uint.from_buffer(ctypes.c_float(arr[row])).value)[2:]
            val = val.zfill(32)
            temp.append(val)
    return np.array(temp)
        
# export IFM data
def exportIFM(out, saveIFM, firstBatch, fname):
    if saveIFM and firstBatch:
        d = np.reshape(out.cpu().numpy()[0], -1, 'C')
        #print(np.shape(out.cpu().numpy()[0]))
        if args.bin_acts:
            d[d > 1e-6] = 1
            d[d < -1e-6] = 1
            d[d != 1] = 0
            d = d.astype(int)
        else:
            #d = binary(d)
            d[d > 1e-6] = 1
            d[d < -1e-6] = 1
            d[d != 1] = 0
            d = d.astype(int)
        pd.DataFrame(d).to_csv(fname)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #train_loader
train_loader = D.DataLoader(datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=args.batch_size, shuffle=True)
    
    #test_loaer
test_loader = D.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=args.test_batch_size, shuffle=True)


################################################################
#MODEL
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.layer2 = nn.Sequential(
            BinaryConv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.fc = BinaryLinear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

##LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = BinaryConv2d(1, 6, kernel_size=5, padding=2)
        self.layer1_2 = nn.BatchNorm2d(6, momentum=args.momentum, eps=args.eps)
        self.layer1_3 = nn.MaxPool2d(2)
        self.layer1_4 = BinaryTanh()
        self.conv2 = BinaryConv2d(6, 16, kernel_size=5, padding=2)
        self.layer2_2 = nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps)
        self.layer2_3 = nn.MaxPool2d(2)
        self.layer2_4 = BinaryTanh()

        # Linear layers
        self.fc1 = BinaryLinear(7*7*16, 120)
        self.layer3_2 = nn.BatchNorm1d(120, momentum=args.momentum, eps=args.eps)
        self.layer3_3 = BinaryTanh()
        self.fc2 = BinaryLinear(120, 84)
        self.layer4_2 = nn.BatchNorm1d(84, momentum=args.momentum, eps=args.eps)
        self.layer4_3 = BinaryTanh()
        self.fc3 = BinaryLinear(84, 10)
        # CrossEntropyLoss() loss function already implements nn.Softmax(), so no need to include here
        
    def forward(self, x):
        global firstBatch
        if saveIFM and firstBatch:
            #d = binary(np.reshape(x.cpu().numpy()[0], -1, 'C'))
            #d = np.reshape(x.cpu().numpy()[0], -1, 'C')
            #print("Type of input")
            #print(type(d[0]))
            #print(np.shape(x.cpu().numpy()[0]))
            #pd.DataFrame(d).to_csv('data/act0.csv')
            exportIFM(x, saveIFM, firstBatch, 'data/act0.csv')
        out = self.conv1(x)
        exportIFM(out, saveIFM, firstBatch, 'data/act1_1.csv')
        out = self.layer1_2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act1_2.csv')
        out = self.layer1_3(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act1_3.csv')
        out = self.layer1_4(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act1_4.csv')
        out = self.conv2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act2_1.csv')
        out = self.layer2_2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act2_2.csv')
        out = self.layer2_3(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act2_3.csv')
        out = self.layer2_4(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act2_4.csv')

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act3_1.csv')
        out = self.layer3_2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act3_2.csv')
        out = self.layer3_3(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act3_3.csv')
        out = self.fc2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act4_1.csv')
        out = self.layer4_2(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act4_2.csv')
        out = self.layer4_3(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act4_3.csv')
        out = self.fc3(out)
        exportIFM(out, saveIFM, firstBatch, 'data/act5.csv')

        if saveIFM:
            firstBatch = False
        return out

    def BinZeroProxUpdate(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    # easy case: push values that exist between [0,1]
                    m.weight[(m.weight < 0.5) & (m.weight >= 0)] = 0
                    m.weight[(m.weight >= 0.5) & (m.weight <= 1)] = 1
                    # now apply standard clipped proximity update
                    m.weight[(torch.abs(m.weight) <= args.lr*args.reg) & (m.weight != 0) & (m.weight != 1)] = 0
                    m.weight[(torch.abs(m.weight - 1.0) <= args.lr*args.reg) & (m.weight != 0) & (m.weight != 1)] = 1
                    m.weight[(m.weight < (args.lr*args.reg*-1.0)) & (m.weight != 0) & (m.weight != 1)] = m.weight[(m.weight < (args.lr*args.reg*-1.0)) & (m.weight != 0) & (m.weight != 1)] + (args.lr*args.reg)
                    m.weight[(m.weight > (args.lr*args.reg)) & (m.weight != 0) & (m.weight != 1)] = m.weight[(m.weight > (args.lr*args.reg)) & (m.weight != 0) & (m.weight != 1)] - (args.lr*args.reg)
    
#model = Model()
model = LeNet5()
########################################################################
if args.cuda:
    #torch.cuda.set_device(3)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        #if epoch%40==0:
            #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        #optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        #if args.bin_zero:
        # proximal gradient update
        #model.BinZeroProxUpdate()
        
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                if args.bin_zero:
                    p.org.copy_(p.data.clamp_(0,1))
                else:
                    p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data)) # loss.data[0]
accur=[]
def test(exportData = False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data#[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
        a=100.*correct / len(test_loader.dataset)
        accur.append(a)  
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        # export weight data
        if exportData:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    weights = module.weight.detach().cpu().numpy()
                    if args.bin_weights:
                        weights[weights > 0] = 1
                        if args.bin_zero:
                            weights[weights <= 0] = 0
                        else:
                            weights[weights <= 0] = -1
                        weights = weights.astype(int)
                    # FOR EASE OF USE: convert to weights-sparsity matrix
                    weights[weights > 1e-6] = 1
                    weights[weights < -1e16] = 1
                    weights[weights != 1] = 0
                    weights = weights.astype(int)
                    print(np.shape(weights))
                    weights = np.reshape(weights, (np.shape(weights)[0], -1),'C')
                    #print(np.shape(weights))
                    pd.DataFrame(weights).to_csv('data/' + str(name) + '.csv')
                
start = time.time()
time_graph=[]
e=[]
for epoch in range(1, args.epochs + 1):
    e.append(epoch)
    train(epoch)   
    seco=timeSince(start)
    time_graph.append(seco)
    exportData = False
    if epoch == args.epochs:
        exportData = True
        saveIFM = True
    test(exportData)

def listToString(thisList):
    if len(thisList) == 0:
        return "NA"
    string = str(thisList[0])
    if len(thisList) > 1:
        for item in thisList[1:]:
            string += ("," + str(item))
    return string
    
    
# export layerwise data
modelShape = "LayerName\tLayerID\tInputShape\tOutputShape\tKernelShape\n"
summaryobject = summary(model, (1,28,28))
for layer in summaryobject.summary_list:
    if ("Conv" in str(layer)) or ("Linear" in str(layer)):
        modelShape += (str(layer) + "\t" + listToString(layer.input_size) + "\t" + listToString(layer.output_size) + "\t" + listToString(layer.kernel_size) + "\n")
print(modelShape)
modelShapeFile = open("ModelShape.txt",'w')
modelShapeFile.write(modelShape)
modelShapeFile.close()
#print(summaryobject.summary_list.kernel_size)

#print(time_graph)
#plt.title('Training for MNIST with epoch', fontsize=20)
#plt.ylabel('time (s)')
#plt.plot(e,time_graph,'bo')
#plt.show()
#plt.title('Accuracy With epoch', fontsize=20)
#plt.plot(e,accur,'bo')
#plt.show()


    
