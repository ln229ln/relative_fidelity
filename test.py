import os
import random
import torch.nn.functional as F
import numpy as np
from numpy import mean
from torch.utils.data import DataLoader
from dataload import featureload, DataLoad
import pandas as pd
import torch
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy import signal
from fun import computedevia, computecorrects
from model import CNNTriplet

plt.rcParams['font.sans-serif']=['SimSun']#Times New Roman
# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=600
plt.rcParams['figure.figsize']=(3,3)#英：2.5,3


def result(penname,modelfile,set='test'):
    data = DataLoad(os.path.join(path, "{}_{}.json".format(set,penname)))
    dataloder = DataLoader(data, batch_size=batchsize, shuffle=False, drop_last=False)
    data_length = len(data)
    model = torch.load(modelfile).to(device)
    d_predict={}
    df_predict = pd.DataFrame(columns=['pen1', 'refer', 'pen2', 'real', 'predict'])#返回值
    print('testpen:',penname, 'test_length:', data_length,modelfile)
    model.eval()
    for batchidx, (feature,tar, pen, pressure, speed) in enumerate(dataloder):
        pen1fea, referfea, pen2fea, tar = feature[0].to(device), feature[1].to(device), feature[2].to(device), tar.to(
            device)
        try:
            predict = model(pen1fea, referfea, pen2fea)
        except:
            print("模型结构错误")
            break
        if (device == 'cpu'):
            predict_ = predict.detach().numpy()
            tar_ = tar.detach().numpy()
        else:
            predict_ = predict.detach().cpu().numpy()
            tar_ = tar.detach().cpu().numpy()
        for i in range(len(tar_)):
            key = pen['pen1'][i] + ',' + pen['refer'][i] + ',' + pen['pen2'][i]+ ',' +str(tar_[i])
            d_predict[key]=d_predict.get(key,[])
            d_predict[key].append(predict_[i])
    for (key,value) in d_predict.items():
        l=key.split(',')
        l.append(mean(value))
        df_predict=df_predict.append(pd.Series(l,index=['pen1', 'refer', 'pen2', 'real', 'predict']),ignore_index=True)
    df_predict['real']=df_predict['real'].astype(float)
    # df_predict.sort_values(by='real', inplace=True)
    return df_predict

#计算容错范围内的正确率以及平均偏差
def corrects(df_predict,margins):
    df_predict['diff']=(df_predict['real']-df_predict['predict']).abs()
    error=df_predict[df_predict['diff'] > 0.25]
    print('预测错误个数：',len(error))
    print(df_predict)

    l_correct=[]
    for margin in margins:
        filtered_df = df_predict[df_predict['diff'] <= margin]
        correct = len(filtered_df)/len(df_predict)
        l_correct.append(correct)
    return pd.Series(l_correct,index=margins)


def plot_meanvar(d_dev):
    total_mean=[]
    total_var=[]
    total_name=[]
    barwidth=0.1
    for key,values in d_dev.items():
        total_name.append(key)
        total_mean.append(np.mean(values))
        total_var.append(np.var(values))
    x=np.arange(len(total_name))
    print(x)
    rect_mean=plt.bar(x=x,height=total_mean,color=['deepskyblue','orange','g'],label=total_name)#
    rect_var =plt.errorbar(x=x
                           ,y=total_mean
                           ,yerr=total_var
                           ,fmt='o'
                           ,color='r'
                           ,elinewidth=1#线宽
                           ,capsize=4)#横线长度
    #给每个柱上标均值
    for i ,y in enumerate(total_mean):
        plt.text(x[i],y+0.001,'%s'%round(y,2),ha='center',fontdict={'size': 12, 'color':  'black'})
    # plt.ylabel('mean_devias')
    plt.xticks(x,total_name)
    plt.ylim(0,0.2)


savepath=r'C:/Users/Administrator/Desktop/IEEE-Trans-Template/picture'#C:\Users\Administrator\Desktop\硕士毕业论文picture
device = "cpu"
batchsize=32
margins=np.linspace(0,0.8,30)
path=r"Data"#/root/autodl-tmp/data/20220722数据采集
set='train'
pennames=['ireader','苹果Pencil','surface','透明笔尖']#'ireader','surface','苹果Pencil','透明笔尖','1','2','3','4','透明笔尖_ireader'
rename={'ireader':'iReader Smart X','苹果Pencil':'Apple Pencil','透明笔尖':'M-Pencil','surface':'Microsoft Surface Pen'}
modelfiles=[r'CNNTriplet-ireader.pth',r'CNNTriplet-苹果Pencil.pth',r'CNNTriplet-surface.pth',r'CNNTriplet-透明笔尖.pth']
total_outputdev={}
pens_corrects=pd.DataFrame(np.zeros(len(margins)),index=margins)
myresult=pd.DataFrame(columns=['pen1','refer','pen2','predict','real'])
worksheet=pd.ExcelWriter('Result/MyResultAverTrain.xlsx')
for i in range(len(pennames)):
    penname=pennames[i]
    modelfile=os.path.join('models/mymodels', modelfiles[i])
    if(penname not in modelfile):
        print('数据和文件未对应！！！')
        break
    df_predict = result(penname, modelfile, set=set)
    correct = corrects(df_predict, margins)  # 返回的是series
    pens_corrects[penname] = correct
    correct.plot(title=set, legend=None, ylabel='Positive Rate', xlabel='Beta')
    fig=plt.figure()
    df_predict.plot.scatter("real", "predict", title='Test{} ({})'.format(i+1,rename[penname]), grid=True, alpha=0.5)
    fig = plt.gcf()  # 获取当前的图表对象
    plt.tight_layout()
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,1,6))
    plt.xlabel("主观评价值")
    plt.ylabel("客观评价值")
    plt.show()
    # fig.set_size_inches(3, 3)
    # fig.savefig(savepath + '/Test{}.png'.format(i+1), bbox_inches='tight', dpi=600)
    df_predict.to_excel(worksheet,sheet_name=rename[penname],index=None)
worksheet.save()
pens_corrects.rename(columns=rename,inplace=True)
pens_corrects.plot(ylabel='Percentage', xlabel=r'$\beta$',grid=True,linewidth=1)#Percentage
# myresult.to_csv('Result/MyResult.csv',index=0)
# plt.figure('2')
# plot_meanvar(total_outputdev)
plt.show()








