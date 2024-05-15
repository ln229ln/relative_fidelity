import os

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import signal
import torch.nn.functional as F
from scipy.fft import rfft, rfftfreq



def computecorrects(predis,label,margin):
    # predict=np.array(list((map(predictResult,predis))))
    # result=np.sum(predict==label)
    result=np.sum(np.abs(predis-label)<=margin)
    return result#predict,


def FIRfilter(signal_source, fs):
    nyq = fs / 2
    f1 = 10
    f2 = 600
    bandpass = signal.firwin(501, [f1 / nyq, f2 / nyq], pass_zero='bandpass', window="hann")
    filteData = signal.lfilter(bandpass, 1, signal_source)
    return filteData

def butter(signal_source, fs):
    sos = signal.butter(10, [20, 700], 'bandpass', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, signal_source)
    return filtered

def weibo(fredata,head=15,tail=700,q=1.1):#输入输出数据类型都是dataframe
    start=head
    bin=0
    bins=[]
    while start<tail:
        end = int(q * start)
        for i in range(start,min(end,tail)):
            bin+=fredata.iloc[i]#fredata是Series类型
        start=end
        bins.append(bin)
        bin=0
    return pd.DataFrame(bins)
def bins(fredata,head=15,tail=700,reso=1):
    start = head
    bin = 0
    bins = []
    while start < tail:
        end=start+reso
        for i in range(start, min(end,tail)):
            bin += fredata.iloc[i]  # fredata是Series类型
        start = end
        bins.append(bin)
        bin = 0
    return pd.DataFrame(bins)

def corrects(x1,x2,label,margin):
    dist = F.pairwise_distance(x1, x2)
    pre=dist.detach().apply_(lambda x:x>margin)
    label=label.detach()
    return (pre==label).sum()


class FFT():
    def __init__(self,sample_rate):
        self.sample_rate=sample_rate
    def fft321(self,acc):#将accfile三种分别做FFT->DFT321
        fft=pd.DataFrame(columns=['x','y','z'])
        fft['x']=self.fft(acc[1])  #fft变换
        fft['y'] = self.fft(acc[2])
        fft['z'] = self.fft(acc[3])
        dft321=(fft['x']**2+fft['y']**2+fft['z']**2)**1/2
        return dft321
    def fft(self,acc):#fft并且频率归一化
        acc=acc - acc.mean()
        acc=acc.to_numpy()
        N = self.sample_rate  # create N points in our fft, where N > length(az)
        X_raw = rfft(acc, N)  # get our raw fft transformation
        X = abs(X_raw)  # take the amplitude of this  data
        X=X/len(X)
        return X




