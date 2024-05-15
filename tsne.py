import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataload import readtxt
import os
import pandas as pd
from matplotlib.patches import Patch
import numpy as np

from model import  CNNTripletVisualR
from torch.utils.data import DataLoader
from dataload import  DataLoad
plt.rcParams['font.sans-serif']=['SimSun']#Times New Roman
# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=600
plt.rcParams['lines.linewidth']=0.7
plt.rcParams['figure.figsize']=(3.5,2.5)#英：2.5,3


def replaceModel(modelfile,new_model):
    model= torch.load(modelfile)
    model.eval()
    print('--------oldmodel:--------')
    for key, _ in model.state_dict().items():
        print(key)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %d" % (total ))
    print('--------newmodel:---------')
    for key, _ in new_model.state_dict().items():
        print(key)
    print('oldmodel:linear.weight',model.state_dict()['linear.weight'])#,model.state_dict()
    # print('before load param:',new_model.state_dict())
    new_model.load_state_dict(model.state_dict(), strict=False)
    new_model.eval()
    print('after load param:', new_model.state_dict()['convs.2.bias'])
    return new_model


def l_to_df(datas,label):
    d={}
    for i,key in enumerate(label):
        v=datas[i]
        while key in d.keys():
            key+='.'
        d[key] = v
    return pd.DataFrame(d)


def savedis(df_data,filename):
    distance_df = pd.DataFrame(index=df_data.columns, columns=df_data.columns)
    for i in df_data.columns:
        for j in df_data.columns:
            if i != j:
                distance = np.linalg.norm(df_data[i] - df_data[j])
                distance_df.loc[i, j] = distance

    # 将结果保存到Excel文档中
    excel_path = '.\{}.xlsx'.format(filename)
    distance_df.to_excel(excel_path, index=True)

    print(f'距离已保存到 {excel_path}')

class VisualRegression():
    def getModel(self, model):
        self.model = model
    def tripletFeature(self,dataloader1,dataloader2):
        shalollow_features=[]
        deep_features=[]
        labels=[]
        for batchidx, (feature, tar, pen, pressure, speed) in enumerate(dataloader1):
            pen1fea, referfea, pen2fea, tar = feature[0], feature[1], feature[2], tar
            triplet_fea=torch.stack([pen1fea,referfea,pen2fea],dim=1)
            deep_fea1,deep_fear,deep_fea2=self.model(pen1fea, referfea, pen2fea)
            deep_triplet_fea=torch.stack([deep_fea1,deep_fear,deep_fea2],dim=1)
            tar_ = tar.detach().numpy().tolist()
            triplet_fea_=triplet_fea.view(triplet_fea.shape[0],-1).detach().numpy().tolist()
            deep_triplet_fea_=deep_triplet_fea.view(deep_triplet_fea.shape[0],-1).detach().numpy().tolist()
            shalollow_features+=triplet_fea_
            deep_features+=deep_triplet_fea_
            labels+=tar_
            if(len(labels)>=480):
                break
        for batchidx, (feature, tar, pen, pressure, speed) in enumerate(dataloader2):
            pen1fea, referfea, pen2fea, tar = feature[0], feature[1], feature[2], tar
            triplet_fea=torch.stack([pen1fea,referfea,pen2fea],dim=1)
            deep_fea1,deep_fear,deep_fea2=self.model(pen1fea, referfea, pen2fea)
            deep_triplet_fea=torch.stack([deep_fea1,deep_fear,deep_fea2],dim=1)
            tar_ = tar.detach().numpy().tolist()
            triplet_fea_=triplet_fea.view(triplet_fea.shape[0],-1).detach().numpy().tolist()
            deep_triplet_fea_=deep_triplet_fea.view(deep_triplet_fea.shape[0],-1).detach().numpy().tolist()
            shalollow_features+=triplet_fea_
            deep_features+=deep_triplet_fea_
            labels+=tar_
        return shalollow_features,deep_features,labels
    def plotData(self,dataS,dataD,labels,train_length):
        fig = plt.figure()
        ax0 = fig.add_subplot(211)  # , projection='3d'
        ax1 = fig.add_subplot(212)
        ax0.set_title('Raw space')
        ax1.set_title('Deep space')
        marker='o'
        nS=0
        nD = 0
        for i in range(len(labels)):
            if(i>=train_length/2):
                marker='x'
            label=labels[i]
            ax0.scatter(
                dataS[i][0],
                dataS[i][1],
                # label,
                s=4,
                marker=marker,
                color=plt.cm.Paired(self.colormap(label)),
                linewidths=0.6,
                alpha=0.8
            )
            nS+=1
            ax1.scatter(
                dataD[i][0],
                dataD[i][1],
                # label,
                s=4,
                marker=marker,
                color=plt.cm.Paired(self.colormap(label)),
                linewidths=0.6,
                alpha=0.8
            )
            nD+=1
        # plt.xlabel('D1')
        legend_elements = [Patch(facecolor=plt.cm.Paired(v), edgecolor='black', label=k) for k, v in
                           {'P<=0.2':1,'0.2<P<=0.4':2,'0.4<P<0.6':3,'0.6<=P<0.8':4,'P>=0.8':7}.items()]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        # plt.savefig('picture\\ireader2D.png')
        plt.show()
        print('shallow n:',nS,'deep n:',nD)
    def colormap(self,label):
        if label <= 0.2:
            return 1;
        elif label>0.2 and label<=0.4:
            return 2
        elif label>0.4 and label<0.6:
            return 3
        elif label>=0.6 and label<0.8:
            return 4
        else:
            return 7

def norm(X,dim=2):
    X=np.array(X)
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return X_normalized


modelfile=r'models\mymodels\CNNTriplet-ireader.pth'
new_model_object=CNNTripletVisualR()
new_model=replaceModel(modelfile,new_model_object)

#-----metric learning
rdt=readtxt('userData/huawei.txt')
train_pens=rdt.getTrainPaper()
test_pens=rdt.getTestPaper()
all_pens=train_pens+test_pens
repeat = ['R{}-FFT.csv'.format(i + 1) for i in range(0,2)]
colormap={}
for i,pens in enumerate(all_pens):
    colormap[pens]=i
#------regression
pen='ireader'
file_train=r'Data/train_{}.json'.format(pen)
file_test=r'Data/test_{}.json'.format(pen)
data_train = DataLoad(file_train)
data_test = DataLoad(file_test)
print('data_train_length:',len(data_train))
dataloder_train = DataLoader(data_train, batch_size=32, shuffle=False, drop_last=False)
dataloder_test = DataLoader(data_test, batch_size=32, shuffle=False, drop_last=False)
visual=VisualRegression()
visual.getModel(new_model)
datas,deep_features,labels=visual.tripletFeature(dataloder_train,dataloder_test)
D_reduce = TSNE(n_components=2,init='pca',perplexity=30, random_state=12)#tsne
datas_reduced=norm(D_reduce.fit_transform(datas))
features_reduced=norm(D_reduce.fit_transform(deep_features))
visual.plotData(datas_reduced,features_reduced,labels,len(data_train))#

