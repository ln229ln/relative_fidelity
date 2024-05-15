import copy

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt




#频域
class CNNTriplet(nn.Module):
    def __init__(self):
        super(CNNTriplet, self).__init__()
        self.convs=nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=3)
        ,nn.MaxPool1d(3, stride=2)
        ,nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        , nn.MaxPool1d(3, stride=2)
        ,nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        , nn.MaxPool1d(3, stride=1)
        ,nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )
        self.conv=nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4)
        self.layernorm = nn.LayerNorm(245)
        self.Bathnorm=nn.BatchNorm1d(15)
        self.dropout=nn.Dropout(p = 0.1)
        self.linear = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pen1fea, referfea, pen2fea):#pen1fea, referfea, pen2fea，energy:96*15*1
        x = torch.cat((pen1fea, referfea, pen2fea))
        energy = torch.sum(x ** 2, dim=-1).unsqueeze(-1)
        x= self.layernorm(x)
        energy=self.Bathnorm(energy)
        for fv in range(x.size(1)):#
            every_fv_out=self.convs(x[:,fv,:].unsqueeze(1))#N*channel*emb
            # test=torch.ones((1,1,245))
            # for layer in self.convs:
            #     test=layer(test)
            #     print(test.size())
            every_fv_out=self.dropout(every_fv_out)
            energy_copy = energy[:,fv,:]#N*1
            for channel in range(every_fv_out.size(1) - 1):
                energy_copy = torch.cat([energy_copy, energy[:,fv]], 1)  # N*channel
            energy_copy =energy_copy.unsqueeze(-1)# N*channel*1
            every_fv_out =torch.cat((every_fv_out, energy_copy), dim=-1) # 96*channel*emb->96*channel*(emb+1)
            every_fv_out =self.conv(every_fv_out).transpose(1,2)
            if fv == 0:
                out=every_fv_out
            else:
                out = torch.cat((out, every_fv_out), dim=1)  # N*15*channel*embed

        [y0, y1, y2] = torch.chunk(out, 3)

        y = F.pairwise_distance(y1, y2, p=2) / F.pairwise_distance(y1, y0, p=2)  # pen01的距离-pen02的距离\,32*15
        y = 2.2 * (self.sigmoid(y + 0.2) - 0.55)
        print(y)
        y=self.linear(y).squeeze(1)
        # print('--*--',y)
        return y


class CNNTripletAblation (nn.Module):
    def __init__(self):
        super(CNNTripletAblation , self).__init__()

        self.linear = nn.Linear(15, 1)
        # self.linearEnergy=nn.Linear(1, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pen1fea, referfea, pen2fea):  # pen1fea, referfea, pen2fea，energy:96*15*1
        y = F.pairwise_distance(referfea,  pen2fea, p=2) / F.pairwise_distance(referfea, pen1fea, p=2)  # pen01的距离-pen02的距离\,32*15
        y = 2.2 * (self.sigmoid(y + 0.2) - 0.55)
        # print(y)
        y = self.linear(y).squeeze(1)
        return y



class CNNTripletVisualR(nn.Module):
    def __init__(self):
        super(CNNTripletVisualR, self).__init__()
        self.convs=nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=3)
        ,nn.MaxPool1d(3, stride=2)
        ,nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        , nn.MaxPool1d(3, stride=2)
        ,nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        , nn.MaxPool1d(3, stride=1)
        ,nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )
        self.conv=nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4)
        self.layernorm = nn.LayerNorm(245)
        self.Bathnorm=nn.BatchNorm1d(15)
        self.dropout=nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(15, 1)

    def forward(self, pen1fea, referfea, pen2fea):#pen1fea, referfea, pen2fea，energy:96*15*1
        x = torch.cat((pen1fea, referfea, pen2fea))
        energy = torch.sum(x ** 2, dim=-1).unsqueeze(-1)
        x= self.layernorm(x)
        energy=self.Bathnorm(energy)
        for fv in range(x.size(1)):#
            every_fv_out=self.convs(x[:,fv,:].unsqueeze(1))#N*channel*emb
            every_fv_out=self.dropout(every_fv_out)
            energy_copy = energy[:,fv,:]#N*1
            for channel in range(every_fv_out.size(1) - 1):
                energy_copy = torch.cat([energy_copy, energy[:,fv]], 1)  # N*channel
            energy_copy =energy_copy.unsqueeze(-1)# N*channel*1
            every_fv_out =torch.cat((every_fv_out, energy_copy), dim=-1) # 96*channel*emb->96*channel*(emb+1)
            every_fv_out =self.conv(every_fv_out).transpose(1,2)
            if fv == 0:
                out=every_fv_out
            else:
                out = torch.cat((out, every_fv_out), dim=1)  # N*15*channel*embed
        # out =out*self.linear.weight.reshape(1,15,1)
        [y0, y1, y2] = torch.chunk(out, 3)
        return y0, y1, y2


class CNNTripletVisual(nn.Module):
    def __init__(self):
        super(CNNTripletVisual, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=3)
            , nn.MaxPool1d(3, stride=2)
            , nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
            , nn.MaxPool1d(3, stride=2)
            , nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
            , nn.MaxPool1d(3, stride=1)
            , nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )
        self.conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4)
        self.layernorm = nn.LayerNorm(245)
        self.Bathnorm = nn.BatchNorm1d(15)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(15, 1)

    def forward(self, x):  # pen1fea, referfea, pen2fea，energy:96*15*1
        energy = torch.sum(x ** 2, dim=-1).unsqueeze(-1)
        x = self.layernorm(x)
        energy = self.Bathnorm(energy)
        for fv in range(x.size(1)):  #
            every_fv_out = self.convs(x[:, fv, :].unsqueeze(1))  # N*channel*emb
            every_fv_out = self.dropout(every_fv_out)
            energy_copy = energy[:, fv, :]  # N*1
            for channel in range(every_fv_out.size(1) - 1):
                energy_copy = torch.cat([energy_copy, energy[:, fv]], 1)  # N*channel
            # energy_copy=self.linearEnergy(energy_copy)
            energy_copy = energy_copy.unsqueeze(-1)  # N*channel*1
            every_fv_out = torch.cat((every_fv_out, energy_copy), dim=-1)  # 96*channel*emb->96*channel*(emb+1)
            every_fv_out = self.conv(every_fv_out).transpose(1, 2)
            if fv == 0:
                out = every_fv_out
            else:
                out = torch.cat((out, every_fv_out), dim=1)  # N*15*channel*embed

        return out


