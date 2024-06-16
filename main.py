import math
import os.path
from random import random

import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

from dataload import  featureload, DataLoad
from model import  CNNTriplet

from scipy import signal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from fun import ContrastiveLoss, corrects, computecorrects, computedevia

import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=110
plt.rcParams['figure.figsize']=(10,8)



#Step 1:============================设置CUP及模型参数======================
datatype='F'
penname='surface'#ireader,surface,苹果Pencil,透明笔尖
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
batchsize=32
seed=1
torch.manual_seed(seed)#设置随机数种子
# Step 2:============================加载数据========================
path=r"Data"#/root/autodl-tmp/data/20220722数据采集
train_data=DataLoad(os.path.join(path,"train_{}.json".format(penname)))
test_data=DataLoad(os.path.join(path,"test_{}.json".format(penname)))
train_length=len(train_data)
test_length=len(test_data)
print('train_length:',train_length,'test_length():',test_length)
train_dataloder = DataLoader(train_data, batch_size=batchsize, shuffle=True,num_workers=0,drop_last=False)
test_dataloder = DataLoader(test_data, batch_size=batchsize, shuffle=False)
# Step 3:============================初始化损失函数和优化器等=================
# 创建和实例化一个整个模型类的对象
model=CNNTriplet().to(device)
criterion=nn.MSELoss()
# criterion.to(device)
margin=0.25
epch=250
stopcondition=5
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-2)#,weight_decay=1e-2
# Step 4:============================开始训练网络===================
l_trainloss=[]
l_testloss=[]
l_train_corrects=[]
l_test_corrects=[]
timestr=time.strftime("%d-%H-%M")
print("当前时刻(%d-%H-%M)：",timestr)
writer = SummaryWriter("./logs"+timestr)#
# 为了实时观测效果，我们每一次迭代完数据后都会，用模型在测试数据上跑一次，看看此时迭代中模型的效果。
#将距离转换为选择概率
for e in range(epch):
    # 4.1==========================训练模式==========================
    print('第{}轮训练开始'.format(e + 1))
    start_time = time.time()
    model.train()  # 将模型改为训练模式
    # 每次迭代都是处理一个小批量的数据，batch_size是32
    train_losses = 0
    train_corrects=0
    train_devias = 0
    for batchidx, (feature,tar,pen,pressure,speed) in enumerate(train_dataloder):#feature:list(3),feature[0]:batchsize*96*1
        # 时域feature：batchsize*4096*3转为batch*C*L
        pen1fea, referfea, pen2fea,tar=feature[0].to(device), feature[1].to(device), feature[2].to(device),tar.to(device)
        optimizer.zero_grad()
        predict=model(pen1fea, referfea, pen2fea)
        loss = criterion(predict,tar)#
        loss.backward()
        if(device=='cpu'):
            predict_=predict.detach().numpy()
            tar_=tar.detach().numpy()
        else:
            predict_ = predict.detach().cpu().numpy()
            tar_ = tar.detach().cpu().numpy()
        correct = computecorrects(predict_,tar_,margin)#
        devia=computedevia(predict_,tar_)
        print(predict_, tar_,correct)
        print("loss:", loss.item(),"devia:",devia)
        # print(pen['pen1'],'\n',pen['refer'],'\n',pen['pen2'])
        #loss.requires_grad_(True)
        optimizer.step()
        # 记录误差
        train_losses += loss.item()
        train_corrects+=correct
        train_devias+=devia
    end_time = time.time()
    l_trainloss.append(train_losses / train_length)
    l_train_corrects.append(train_corrects / train_length)
    print('训练第{}轮,训练误差是{}'.format(e + 1, train_losses / train_length*batchsize))#这里与原来不同，整体扩大了3倍
    print('训练第{}轮,训练正确个数是{}'.format(e + 1, train_corrects ))
    print('此轮训练时间：{}'.format(end_time - start_time))
    if(train_corrects / train_length>0.99 and train_devias / train_length*batchsize<0.1):
        stopcondition-=1
        if(stopcondition==0):
            torch.save(model, '{}-{}{}-seed{}.pth'.format(type(model).__name__, penname, e + 1, seed))
            plt.plot(l_train_corrects,label='train_corrects')
            plt.plot(l_test_corrects, label='l_test_corrects')
            plt.plot(l_testloss, label='l_test_loss')
            plt.show()
            exit()
    #4.2==========================每次进行完一个训练迭代，测试一次看此时的效果==========================
    #在测试集上检验效果
    model.eval()  # 将模型改为预测模式
    eval_losses = 0  # 每一轮训练把验证损失清零
    eval_corrects = 0
    eval_devias=0
    #每次迭代都是处理一个小批量的数据，batch_size是32
    with torch.no_grad():
        for batchidx, (feature,tar,pen,pressure,speed) in enumerate(test_dataloder):
            pen1fea, referfea, pen2fea, tar = feature[0].to(device), feature[1].to(device), feature[2].to(device), tar.to(device)
            predict= model(pen1fea, referfea, pen2fea)
            loss = criterion(predict,tar)#
            if (device == 'cpu'):
                predict_ = predict.detach().numpy()
                tar_ = tar.detach().numpy()
            else:
                predict_ = predict.detach().cpu().numpy()
                tar_ = tar.detach().cpu().numpy()
            correct = computecorrects(predict_, tar_, margin)  #
            devia = computedevia(predict_, tar_)
            print(predict_, tar_, correct)
            print("loss:", loss.item(), "devia:", devia)
            eval_losses += loss.item()
            eval_corrects+=correct
            eval_devias+=devia
    writer.add_scalars('loss', {'train':train_losses / train_length * batchsize,'test':eval_losses / test_length*batchsize}, e)
    writer.add_scalars('correct',{'train':train_corrects / train_length,'test':eval_corrects / test_length} , e)
    writer.add_scalars('devias', {'train':train_devias / train_length * batchsize,'test':eval_devias / test_length*batchsize}, e)
    print('整体测试正确个数是{}'.format(eval_corrects))
    l_testloss.append(eval_losses / test_length)
    l_test_corrects.append(eval_corrects / test_length)
torch.save(model, '{}-{}{}.pth'.format(type(model).__name__,penname,e + 1))




#if __name__ == '__main__':
    # datapath = r"D:\数据\20220722数据采集\20220722数据采集"
    # simpath = r"D:\数据\20220722数据采集\SiameseHaptic_F\HapticSimilar.csv"
    # labelpath = r'D:\数据\20220722数据采集\MyVerify\labelnum.xlsx'
    # #model_file= 'Hapticmodels/CNNmodel/hapticTrain1model_T_fv.pth'
    # #print('-------------{}-----------'.format(model_file))
    # singleTrain(train_path, test_path, tar_path, model_file,)
    # #testResult(test_path, tar_path, model_file)
    #singleTrain()
