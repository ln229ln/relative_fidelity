import glob

import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from fun import*
import pandas as pd
import matplotlib as mpl
plt.rcParams['font.sans-serif']=['SimSun']#SimHei Times New Roman
plt.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size'] = 8
plt.rcParams['figure.dpi']=600
plt.rcParams['figure.figsize']=(3.5,3)

def plt_acc(data,savepath,save=True):
    #时域
    fig,axs=plt.subplots(3,1,figsize=(4, 2.6))
    axs[0].plot(np.linspace(0,1,4096),data[1],linewidth=0.8)
    axs[0].set_title('x 轴')#x axis
    axs[1].plot(np.linspace(0,1,4096),data[2],linewidth=0.8)
    axs[1].set_title('y 轴')
    axs[1].set_ylabel(r'Amplitude($m/s^2$)')
    axs[2].plot(np.linspace(0,1,4096),data[3],linewidth=0.8)
    axs[2].set_title('z 轴')
    plt.xlabel('时间 (s)')#Times(s)
    plt.tight_layout()
    if(save):
        plt.savefig(savepath+'/accT.png',bbox_inches='tight',dpi=600)
    # 频域
    fig2 = plt.figure()
    fft=FFT(4096)
    data=fft.fft321(data)
    data=bins(data,head=10,tail=500,reso=2)
    data=data/data[0].max()
    plt.plot(np.linspace(10,500,245),data[0].values,'r')
    plt.xlabel(r'频率 $(Hz)$')#''Frequency
    plt.ylabel('归一化幅值')#Norm Intensity
    plt.xlim([10,500])
    plt.ylim([0,1.1])
    plt.tight_layout()
    fig2.set_size_inches(4, 2)
    if (save):
        fig2.savefig(savepath+'/accF.png',bbox_inches='tight',dpi=600)

def plt_G(savepath=r'C:/Users/Administrator/Desktop/IEEE-Trans-Template/picture'):
    # 画自己设计的激活函数，程序中用不上
    med=torch.linspace(0,10,50)
    sig=torch.nn.Sigmoid()
    sigout=2.2*(sig(med+0.2)-0.55)
    print(sigout)
    plt.plot(med,sigout,c='b',label='2.2*(sig(med+0.2)-0.55)')
    fig = plt.gcf()
    plt.xlim(-100,100)
    plt.ylim(0,1)
    plt.xticks(np.linspace(0,10,11),fontsize=8)
    plt.yticks(np.linspace(0,1,11),fontsize=8)
    plt.title("G(*)",fontsize=8)
    plt.grid()
    plt.xlabel('d(R,B)/d(R,A)')
    plt.ylabel('P')
    fig.set_size_inches(3, 2.2)  # 设置图像大小为6x4英寸
    plt.show()
    # fig.savefig(savepath+'/G.png', dpi=600, bbox_inches='tight')
if __name__ == '__main__':
    path = r'D:\数据\20220722数据采集\20220722数据采集\A4纸10-铅笔HB\A-0-P-150\S100\R1-Acc.csv'
    # savepath=r'C:/Users/Administrator/Desktop/IEEE-Trans-Template/picture'
    savepath = r'C:\Users\Administrator\Desktop\硕士毕业论文picture'
    data = pd.read_csv(path, header=None, usecols=[1, 2, 3])
    plt_acc(data,savepath=savepath)
    plt_G()


