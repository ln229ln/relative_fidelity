import itertools
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from fun import*
import pandas as pd
import torch
import json
import math


class DataLoad(Dataset):
    def __init__(self, path):
        with open(path, encoding='utf8') as f1:
            # 加载文件的对象
            self.py_list = json.load(f1)
    def __getitem__(self,idx):
        d=self.py_list[idx]
        r_feature=torch.tensor(d["r_feature"],dtype=torch.float32)
        p1_feature = torch.tensor(d["p1_feature"], dtype=torch.float32)
        p2_feature = torch.tensor(d["p2_feature"], dtype=torch.float32)
        label=torch.tensor(d["label"], dtype=torch.float32)
        press=torch.tensor(d["pressure"], dtype=torch.float32)
        speed = torch.tensor(d["speed"], dtype=torch.float32)
        pen={"refer": d["refer"], "pen1": d["pen1"], "pen2": d['pen2']}
        return [p1_feature,r_feature,p2_feature],label,pen,press,speed
    def __len__(self):
        return len(self.py_list)


'''
class featureload(Dataset):
    '''
    set:'huawei'/'digital'
    settype:'train'/'test'
    datatype:'F'/'T'
    allFV:allfv拼接
    '''
    def __init__(self,path,set,settype,datatype='F',allFV=False):
        self.dlp=dataloadProduct(path,set,settype,allFV)
        self.datatype=datatype
        self.set=set
        self.allfv=allFV
        if(set.startswith('huawei')):
            self.samplerate=4096
            self.tar_coorpath=r'userData/huaweicoord2023.4.4.xlsx'
            self.tar_tripletpath=r'userData/triplet-label.xlsx'
        elif(set.startswith('digital')):
            self.samplerate = 2000
            self.tar_tripletpath = r'userData/detailed_study_named.csv'
        else:
            self.samplerate = 4096
            self.tar_coorpath = r''
            self.tar_tripletpath = r'userData/triplet-label.xlsx'
    def __getitem__(self,idx):
        data=self.dlp[idx]['data']
        pen=self.dlp[idx]['pen']
        press=self.dlp[idx]['pressure']
        speed=self.dlp[idx]['speed']
        press = torch.Tensor([press])
        speed = torch.Tensor([speed])
        r=self.dlp[idx]['repeat']
        label=self.dlp[idx]['label']
        label = torch.tensor(label, dtype=torch.float32)
        if(self.set.endswith('triplet')):
            p1_data=self.pretreat(data['p1_data'])
            r_data= self.pretreat(data['r_data'])
            p2_data= self.pretreat(data['p2_data'])
            #直接根据主观实验得到概率
            # return [p1_data,r_data,p2_data],label,pen,press,speed#这里的pen是字典
            res={"p1_feature":p1_data.tolist(),"r_feature":r_data.tolist(),"p2_feature":p2_data.tolist(),"label":label.item(),"pressure":press.tolist(),"speed":speed.tolist(),"refer":pen["refer"],"pen1":pen["pen1"],"pen2":pen['pen2']}

            return res
        else:
            data,tar=self.pretreat(data,pen)
            return data,tar,pen,press,speed

    def featurepro(self,data):#时域转化为频域并bins合并处理
        fft=FFT(self.samplerate)
        #data = pd.DataFrame(fft.fft(data[2]))#仅看z轴的频域
        data=fft.fft321(data)
        data=bins(data,head=10,tail=500,reso=2)
        return data
    def pretreat(self,data):#对加速度数据进行预处理并返回feature，并根据pen类型返回对应的坐标label
        if (self.datatype == 'F'):
            if(self.allfv):
                data=(data.apply(lambda x:bins(x,head=10,tail=500,reso=2).iloc[:,0],axis=1))
            else:
                data = self.featurepro(data).T  # 时域变频域并处理
        data = torch.tensor(data.values,dtype=torch.float32)  # 变成tensor类型,sequence*feature
        # 归一化
        # data=data/data.max()
        return data

    def __len__(self):
        return self.dlp.length




class dataloadProduct():
    def __init__(self,path,set,settype,allFV=False):#set数据集可选值：1.'huawei'，2.'huawei_triplet'，4.'digital——triplet'
        self.path = path;               #settype训练集还是测试集，可选值：1.'train'，2.'test'
        self.settype=settype
        self.length =0
        if(set == 'huawei_triplet'):
            if(allFV):
                self.set = HuaweiTripletAllFV(self.path)
            else:
                self.set = HuaweiTriplet(self.path)
        elif (set == 'digital_triplet'):
            if (allFV):
                self.set = DigitalAllFV(self.path)
            else:
                self.set=DigitalAllFV(self.path)
        else:
            print("没有该数据集")
            self.set = None
        self.length = self.set.length(settype)
    def __getitem__(self,idx):
        return self.set.getItem(idx,self.settype)



class HuaweiTripletParent():
    def __init__(self,path):
        self.path = path;
        self.rt = readtxt('userData/huawei.txt');
        self.l_trainpaper = self.rt.getTrainPaper();
        self.l_testpaper = self.rt.getTestPaper();
        self.train_triplet = self.buildTrainTriplet(self.l_trainpaper)
        # random.seed(4)
        # rows=random.sample(range(int(self.train_triplet.shape[0]/2)),25)
        # self.test_triplet =self.train_triplet.iloc[rows].copy()
        # self.train_triplet.drop(rows,inplace=True)
        self.test_triplet = self.buildTestTriplet(self.l_trainpaper, self.l_testpaper)
        self.added_train_triplet = self.addTrainTriplet()
        self.added_test_triplet=self.addTestTriplet()
    def buildTrainTriplet(self,l_trainpaper):
        df1=pd.read_excel(r'userData/triplet-label.xlsx')
        df1=df1[df1['pen1'].isin(l_trainpaper)&df1['pen2'].isin(l_trainpaper)&df1['refer'].isin(l_trainpaper)]
        df2 = df1.copy()
        df2.iloc[:, [0, 2]] = df2.iloc[:, [2, 0]]
        df2.iloc[:, [3, 4]] = df2.iloc[:, [4, 3]]
        df_concat = pd.concat([df1,df2],axis=0)
        df_concat['label']=(df_concat['picked1']/(df_concat['picked1']+df_concat['picked2']))#.apply(rtoc)
        return df_concat
    def buildTestTriplet(self,l_trainpaper,l_testpaper):#这里不包括以test做参考的情况，但包括pen1和pen2都是test的情况
        df1 = pd.read_excel(r'userData/triplet-label.xlsx')#sheet_name='high',usecols=[1,2,3,4,5]
        df1 = df1[df1['refer'].isin(l_trainpaper) & (df1['pen1'].isin(l_testpaper) | df1['pen2'].isin(l_testpaper))]
        df1['label'] = (df1['picked1'] / (df1['picked1'] + df1['picked2']))
        return df1
    def addTrainTriplet(self):#添加RRA
        df1=pd.DataFrame(columns=['pen1','refer','pen2','label'])
        for i in range(len(self.l_trainpaper)):
            for j in range(i+1,len(self.l_trainpaper)):
                p_a=self.l_trainpaper[i]
                p_r=p_a
                p_b=self.l_trainpaper[j]
                add={'pen1':p_a,'refer':p_r,'pen2':p_b,'label':0.9}
                df1=df1.append(add,ignore_index=True)
        df2 = df1.copy()
        df2.iloc[:, [0, 2]] = df2.iloc[:, [2, 0]]
        df2.iloc[:, 3] = 0.1
        df_concat = pd.concat([df1, df2], axis=0)
        print('added train_triplet:',df_concat.shape[0])
        return df_concat
    def addTestTriplet(self):
        df_added = pd.DataFrame(columns=['pen1', 'refer', 'pen2', 'label'])
        refer=['A4纸10-中性笔','A4纸10-铅笔HB','A4纸10-圆珠笔','A4纸10-马克笔']
        for p_a in refer:
            for p_b in self.l_testpaper:
                p_r = p_a
                add = {'pen1': p_a, 'refer': p_r, 'pen2': p_b, 'label': 1}
                df_added=df_added.append(add, ignore_index=True)
        print('added test_triplet:', df_added.shape[0])
        return df_added
    def gToN(self,g):
        if (g == '100'):
            return 0.8
        elif (g == '150'):
            return 1.12
        elif (g == '200'):
            return 1.4

class HuaweiTriplet(HuaweiTripletParent):
    l_pressure = ['A-0-P-150']  # ,'A-0-P-150','A-0-P-200'
    l_speed = ['S80','S100','S120']  # 'S60','S80','S100',,'S120','S140'
    repeat = ['R{}-Acc.csv'.format(i + 1) for i in range(0, 2)]
    def __init__(self,path):
        super(HuaweiTriplet,self).__init__(path);
    def getItem(self,i,train_or_test):
        i_pe = i // (len(self.l_pressure) * len(self.l_speed) * (len(self.repeat) ** 3))
        i_pr = (i // (len(self.l_speed) * (len(self.repeat) ** 3))) % len(self.l_pressure)
        i_s = (i // (len(self.repeat) ** 3)) % len(self.l_speed)
        i_r = i % (len(self.repeat) ** 3)
        r_r=(i_r//(len(self.repeat) ** 2))%len(self.repeat)
        p1_r=(i_r//len(self.repeat))%len(self.repeat)
        p2_r=i_r%len(self.repeat)
        label=0
        (refer, pen1, pen2) =('','','')
        if(train_or_test=='train'):
            (pen1,refer,pen2,picked1,picked2,label)=list(self.train_triplet.iloc[i_pe])
        elif(train_or_test=='test'):
            ( pen1,refer, pen2,picked1,picked2,label) = list(self.test_triplet.iloc[i_pe])
        r_path=os.path.join(self.path, refer, self.l_pressure[i_pr], self.l_speed[i_s], self.repeat[r_r])
        p1_path=os.path.join(self.path, pen1, self.l_pressure[i_pr], self.l_speed[i_s], self.repeat[p1_r])
        p2_path = os.path.join(self.path, pen2, self.l_pressure[i_pr], self.l_speed[i_s], self.repeat[p2_r])
        r_data=pd.read_csv(r_path, header=None, usecols=[1, 2, 3])
        p1_data = pd.read_csv(p1_path, header=None, usecols=[1, 2, 3])
        p2_data = pd.read_csv(p2_path, header=None, usecols=[1, 2, 3])
        result={'pen':{'refer':refer,'pen1':pen1,'pen2':pen2},'data':{'r_data':r_data,'p1_data':p1_data,'p2_data':p2_data},'pressure':gToN(self.l_pressure[i_pr].lstrip("A-0-P-"))
                ,'speed':float(self.l_speed[i_s].strip('S'))*0.001,'repeat':{'r_repeat':self.repeat[r_r],'p1_repeat':self.repeat[p1_r],'p2_repeat':self.repeat[p2_r]},'label':label}
        return result
    def length(self,train_or_test):
        if (train_or_test == 'train'):
            return self.train_triplet.shape[0]*(len(self.l_pressure) * len(self.l_speed) * (len(self.repeat) ** 3))
        elif(train_or_test == 'test'):
            return self.test_triplet.shape[0]*(len(self.l_pressure) * len(self.l_speed) * (len(self.repeat) ** 3))
#所有压力和速度拼接
class HuaweiTripletAllFV(HuaweiTripletParent):
    tmp=[i for i in range(0, 2)]
    repeat = ['R{}-FFT.csv'.format(i + 1) for i in tmp]
    l_rcmbn=list(itertools.permutations(tmp, 2))
    def __init__(self,path):
        super(HuaweiTripletAllFV,self).__init__(path);
    def getItem(self,i,train_or_test):
        label = 0
        (refer, pen1, pen2) = ('', '', '')
        if (train_or_test == 'train'):
            if(i<self.train_triplet.shape[0]*len(self.repeat) ** 3):#----R,A,B不重复
                i_pe = i // (len(self.repeat) ** 3)
                r_r=(i//(len(self.repeat) ** 2))%len(self.repeat)
                p1_r=(i//len(self.repeat))%len(self.repeat)
                p2_r=i%len(self.repeat)
                (pen1, refer, pen2, picked1, picked2, label) = list(self.train_triplet.iloc[i_pe])
            else:  # ------R,R,B
                i = i - self.train_triplet.shape[0] * (len(self.repeat) ** 3)
                i_pe = i // (len(self.l_rcmbn) * len(self.repeat))
                ind = (i % (len(self.l_rcmbn) * len(self.repeat))) // len(self.repeat)
                r_r = self.l_rcmbn[ind][0]
                p1_r = self.l_rcmbn[ind][1]
                p2_r = i % len(self.repeat)
                (pen1, refer, pen2,  label) = list(self.added_train_triplet.iloc[i_pe])
        elif (train_or_test == 'test'):
            if (i < self.test_triplet.shape[0] * len(self.repeat) ** 3):  # ----R,A,B不重复
                i_pe = i // (len(self.repeat) ** 3)
                r_r = (i // (len(self.repeat) ** 2)) % len(self.repeat)
                p1_r = (i // len(self.repeat)) % len(self.repeat)
                p2_r = i % len(self.repeat)
                (pen1, refer, pen2, picked1, picked2, label) = list(self.test_triplet.iloc[i_pe])
            else:  # ------R,R,B
                i = i - self.test_triplet.shape[0] * (len(self.repeat) ** 3)
                i_pe = i // (len(self.l_rcmbn) * len(self.repeat))
                ind = (i % (len(self.l_rcmbn) * len(self.repeat))) // len(self.repeat)
                r_r = self.l_rcmbn[ind][0]
                p1_r = self.l_rcmbn[ind][1]
                p2_r = i % len(self.repeat)
                (pen1, refer, pen2, label) = list(self.added_test_triplet.iloc[i_pe])
        r_path=os.path.join(self.path, refer, self.repeat[r_r])
        p1_path=os.path.join(self.path, pen1, self.repeat[p1_r])
        p2_path = os.path.join(self.path, pen2, self.repeat[p2_r])
        r_data=pd.read_csv(r_path)
        p1_data = pd.read_csv(p1_path)
        p2_data = pd.read_csv(p2_path)
        result={'pen':{'refer':refer,'pen1':pen1,'pen2':pen2},'data':{'r_data':r_data,'p1_data':p1_data,'p2_data':p2_data},'pressure':0.123
                ,'speed':0.12345,'repeat':{'r_repeat':self.repeat[r_r],'p1_repeat':self.repeat[p1_r],'p2_repeat':self.repeat[p2_r]},'label':label}
        return result
    def length(self,train_or_test):
        if (train_or_test == 'train'):
            return self.train_triplet.shape[0]*(len(self.repeat) ** 3)+4*self.added_train_triplet.shape[0]#-----added
        elif(train_or_test == 'test'):
            return self.test_triplet.shape[0]*(len(self.repeat) ** 3)+4*self.added_test_triplet.shape[0]

class  DigitalParent():
    def __init__(self):
        self.rt = readtxt('userData/digital.txt');
        self.l_trainpaper = self.rt.getTrainPaper();
        self.l_testpaper = self.rt.getTestPaper();
        self.train_triplet = self.buildTrainTriplet(self.l_trainpaper)
        self.test_triplet = self.buildTestTriplet(self.l_trainpaper, self.l_testpaper)

    def buildTrainTriplet(self, l_trainpaper):
        df1 = pd.read_csv(r'userData/detailed_study_named.csv')
        df1 = df1[df1['pen1'].isin(l_trainpaper) & df1['pen2'].isin(l_trainpaper) & df1['refer'].isin(l_trainpaper)]
        df2 = df1.copy()
        df2.iloc[:, [0, 2]] = df2.iloc[:, [2, 0]]
        df2.iloc[:, [3, 4]] = df2.iloc[:, [4, 3]]
        df_concat = pd.concat([df1, df2], axis=0)
        df_concat['label'] = (df_concat['picked1'] / (df_concat['picked1'] + df_concat['picked2']))  # .apply(rtoc)
        return df_concat

    def buildTestTriplet(self, l_trainpaper, l_testpaper):  # 这里不包括以test做参考的情况，但包括pen1和pen2都是test的情况
        df1 = pd.read_csv(r'userData/detailed_study_named.csv')
        df1 = df1[df1['refer'].isin(l_trainpaper) & (df1['pen1'].isin(l_testpaper) | df1['pen2'].isin(l_testpaper))]
        df2 = df1.copy()
        df2.iloc[:, [0, 2]] = df2.iloc[:, [2, 0]]
        df2.iloc[:, [3, 4]] = df2.iloc[:, [4, 3]]
        df_concat = pd.concat([df1, df2], axis=0)
        df_concat['label'] = (df_concat['picked1'] / (df_concat['picked1'] + df_concat['picked2']))  # .apply(rtoc)
        # df1['label']=(df1['picked1'] / (df1['picked1'] + df1['picked2']))
        return df_concat

class DigitalAllFV(DigitalParent):
    def __init__(self,path):
        super(DigitalAllFV,self).__init__()
        self.path=path;
    def getItem(self,i,train_or_test):
        label=0
        (refer, pen1, pen2) =('','','')
        if(train_or_test=='train'):
            (pen1,refer,pen2,picked1,picked2,label)=list(self.train_triplet.iloc[i])
        elif(train_or_test=='test'):
            ( pen1,refer, pen2,picked1,picked2,label) = list(self.test_triplet.iloc[i])
        r_path=os.path.join(self.path, refer)
        p1_path=os.path.join(self.path, pen1)
        p2_path = os.path.join(self.path, pen2)
        r_data=pd.read_excel(r_path+'.xlsx',header=None)
        p1_data = pd.read_excel(p1_path+'.xlsx',header=None)
        p2_data = pd.read_excel(p2_path+'.xlsx',header=None)
        result={'pen':{'refer':refer,'pen1':pen1,'pen2':pen2},'data':{'r_data':r_data,'p1_data':p1_data,'p2_data':p2_data},'pressure':0.1
                ,'speed':0.123456,'repeat':{'r_repeat':'1','p1_repeat':'1','p2_repeat':'1'},'label':label}
        return result
    def length(self,train_or_test):
        if (train_or_test == 'train'):
            return self.train_triplet.shape[0]
        elif(train_or_test == 'test'):
            return self.test_triplet.shape[0]

class readtxt():
    def __init__(self,path):
        self.path=path;
    def getTrainPaper(self):
        f=open(self.path,encoding='gbk');
        data=f.readline().strip("训练集：").strip('\n').split("，");
        return data
    def getTestPaper(self):
        f = open(self.path,encoding='gbk');
        f.readline()
        data = f.readline().strip("测试集：").strip('\n').split("，");
        return data

if __name__ == '__main__':
    # path = r"D:\数据\20220722数据采集\20220722数据采集"  # /root/autodl-tmp/data/20220722数据采集
    path=r'D:\数据\20220722数据采集\HapticDFT321_F\FiveRepeat'#所有速度和压力
    savepath = r'G:\Project\PycharmProjects\相对拟真度评测2024.2.1\Data\\'
    print(os.getcwd())
    # path = r'D:\数据\DigitalDraw\acc\realtools'
    train_data = featureload(path, 'huawei_triplet', 'train',allFV=True)
    test_data = featureload(path, 'huawei_triplet', 'test',allFV=True)
    # print('shape',len(train_data[10]['p1_feature']))
    # pd.DataFrame(train_data[962]['p1_feature']).T.plot()
    # plt.show()
    train_length = len(train_data)
    test_length = len(test_data)
    print('train_length', train_length)
    print('test_length',test_length)
    l_test = []
    l_train = []
    nums = 10000
    for i, item in enumerate(train_data):
        l_train.append(item)
        if ((i + 1) % nums == 0):  # 每50个数据样本存成一个文件
            with open(savepath + 'train_ireader_add{}.json'.format((i + 1) // nums), 'w', encoding='utf8') as f1:
                j_str = json.dump(l_train, f1, ensure_ascii=False, indent=2)
            l_train = []
    if l_train is not None:
        with open(savepath + 'train_ireader_add{}.json'.format((i + 1) // nums), 'w', encoding='utf8') as f1:
            j_str = json.dump(l_train, f1, ensure_ascii=False, indent=2)
    print(len(l_train))
    for i, item in enumerate(test_data):
        l_test.append(item)
        if ((i + 1) % nums == 0):  # 每50个数据样本存成一个文件
            with open(savepath + 'test_ireader_add{}.json'.format((i + 1) // nums + 1), 'w', encoding='utf8') as f2:
                j_str = json.dump(l_test, f2, ensure_ascii=False, indent=2)
            l_test = []
    if l_test is not None:
        with open(savepath + 'test_ireader_add{}.json'.format((i + 1) // nums + 1), 'w', encoding='utf8') as f2:
            j_str = json.dump(l_test, f2, ensure_ascii=False, indent=2)
    print(len(l_test))

    # with open(savepath +'huaweitest_透明笔尖1.json', encoding='utf8') as f1:
    #     # 加载文件的对象
    #     py_list = json.load(f1)
    #     print(len(py_list[0]["r_feature"]))
'''









