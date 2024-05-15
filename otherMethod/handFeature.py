import warnings

import pandas as pd
import scipy.optimize as optimize
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr,kendalltau

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=120
plt.rcParams['figure.figsize']=(10,8)

Chi_Eng={'iReader Smart X':'iReader Smart X','苹果Pencil':'Apple Pencil','华为Pencil-透明笔尖':'M-Pencil','微软surface':'Microsoft Surface Pen',
        'A4纸10-中性笔':"gel+A4",'A4纸10-圆珠笔':"ball-point+A4",'A4纸10-铅笔HB':"HB+A4",'A4纸10-马克笔':"Marker+A4"}
def fweibo(fredata,head=11,tail=500,q=1.1):
    start = head
    bins = []
    while start < tail:
        end = int(q * start)+1
        bin = 0
        for i in range(start, min(end, tail)):
            bin += fredata.iloc[i]  # fredata是Serie类型
        start = end
        bins.append(bin)
    return pd.Series(bins)#/max(bins)

#------------------客观距离计算--------------------
def metric_abs():
    read_path="D:\\数据\\20220722数据采集\\HapticDFT321_F\\FiveRepeat"
    save_path="..\\Result"
    l_PenPaper=['A4纸10-马克笔','A4纸10-铅笔HB','A4纸10-圆珠笔','A4纸10-中性笔','iReader Smart X','苹果Pencil','微软surface','华为Pencil-透明笔尖']
    l_refer=['A4纸10-马克笔','A4纸10-铅笔HB','A4纸10-圆珠笔','A4纸10-中性笔']
    df_rmse=pd.DataFrame(index=l_refer,columns=l_PenPaper,dtype=int)
    df_weibo=pd.DataFrame(index=l_refer,columns=l_PenPaper,dtype=int)
    df_snr=pd.DataFrame(index=l_refer,columns=l_PenPaper,dtype=int)
    R_name=["R{}-FFT.csv".format(i+1) for i in range(2)]
    work = pd.ExcelWriter(save_path+"\\"+'other_method_abs.xlsx')
    for refer_pen in l_refer:
        l_testPen=l_PenPaper.copy()
        l_testPen.remove(refer_pen)
        for test_pen in l_testPen:
            rmse_similar=0
            weibo_similar=0
            snr_similar=0
            for R in R_name:
                refer_data = pd.read_csv(os.path.join(read_path,refer_pen,R))
                test_data = pd.read_csv(os.path.join(read_path, test_pen, R))
                Nrefer_data = refer_data.apply(lambda row: row / row.max(), axis=1)
                Ntest_data = test_data.apply(lambda row: row / row.max(), axis=1)
                # print('real',refer_data.iloc[2,range(11,1000)])
                #rmse
                dist=((refer_data-test_data)**2)
                rmse_similar+=((dist.iloc[:,range(11,500)]).sum().sum())**(1/2)
                #snr
                signal=(refer_data**2).iloc[:, range(11, 500)].sum().sum()
                noise=dist.iloc[:, range(11, 500)].sum().sum()
                snr = signal/noise
                snr_similar += snr
                #weibo
                refer_weibo1 = Nrefer_data.apply(fweibo, axis=1)
                # print(refer_weibo1.iloc[2,:])
                test_weibo1 = Ntest_data.apply(fweibo, axis=1)
                refer_weibo2 = Nrefer_data.apply(fweibo, axis=1,head=12)
                test_weibo2 = Ntest_data.apply(fweibo, axis=1,head=12)
                # dist1=((refer_weibo1-test_weibo1)**2)**(1/2)
                # dist2=((refer_weibo2-test_weibo2)**2)**(1/2)
                dist1 = abs(refer_weibo1 - test_weibo1)/refer_weibo1
                dist2 = abs(refer_weibo2 - test_weibo2) /refer_weibo2
                weibo_similar+=(dist1.sum().sum()+dist2.sum().sum())/2
            rmse_similar/=len(R_name)
            weibo_similar /= len(R_name)
            snr_similar/=len(R_name)
            df_rmse.loc[refer_pen,test_pen]=rmse_similar
            df_weibo.loc[refer_pen,test_pen]=weibo_similar
            df_snr.loc[refer_pen,test_pen]=snr_similar
    df_rmse.fillna(0,inplace=True)
    df_rmse=df_rmse.apply(lambda x:x/x.max(),axis=1)
    df_weibo.fillna(0,inplace=True)
    df_weibo=df_weibo.apply(lambda x:1-x/x.max(),axis=1)
    df_snr.fillna(0,inplace=True)
    df_snr=df_snr.apply(lambda x:x/x.max(),axis=1)
    #保存
    df_rmse.to_excel(work,sheet_name='rmse')
    df_weibo.to_excel(work,sheet_name='weibo')
    df_snr.to_excel(work,sheet_name='snr')
    work.save()


#-----------计算相关系数----------
def corr_compute():
    real_scorefile=r'../Result/real_score.xlsx'
    file=r'../Result/other_method_abs.xlsx'
    real=pd.read_excel(real_scorefile,index_col=0)
    rmse=pd.read_excel(file,sheet_name='rmse',index_col=0)
    weibo=pd.read_excel(file,sheet_name='weibo',index_col=0)
    snr=pd.read_excel(file,sheet_name='snr',index_col=0)
    corr_save=pd.DataFrame(index=real.index,columns=['rmse','weibo','snr'])
    for refer in real.index:
        s_real=real.loc[refer,].sort_values()
        s_rmse=rmse.loc[refer,]
        s_weibo=weibo.loc[refer,]
        s_snr=snr.loc[refer,]
        del s_real[refer]
        del s_rmse[refer]
        del s_weibo[refer]
        del s_snr[refer]
        plt.figure(refer)
        plt.subplot(3,1,1)
        plt.scatter(s_real.index,s_real.tolist())
        plt.scatter(s_rmse.index,s_rmse.tolist())
        plt.title('rmse')
        plt.subplot(3, 1, 2)
        plt.scatter(s_real.index, s_real.tolist())
        plt.scatter(s_weibo.index,s_weibo.tolist())
        plt.title('weibo')
        plt.subplot(3, 1, 3)
        plt.scatter(s_real.index, s_real.tolist())
        plt.scatter(s_snr.index, s_snr.tolist())
        plt.title('snr')
        plt.show()
        rmse_corr=s_real.corr(s_rmse, method='kendall')
        weibo_corr=s_real.corr(s_weibo, method='kendall')
        snr_corr=s_real.corr(s_snr, method='kendall')
        corr_save.loc[refer,'rmse']=rmse_corr
        corr_save.loc[refer, 'weibo'] = weibo_corr
        corr_save.loc[refer, 'snr'] = snr_corr
    print(corr_save)

#-------绝对相似度和相对相似度映射关系---------


def target_func(x, a):
    return 1/(1+np.exp(-a*x))

def aboscore_relascore(df_real,df_metric,testpen):
    df_train = df_real[(df_real['pen1'] != testpen) & (df_real['pen2'] != testpen)]
    df_test = df_real[(df_real['pen1'] == testpen) | (df_real['pen2'] == testpen)]
    df_test['predict']=pd.Series()
    for refer, train_group in df_train.groupby('refer'):
        x = []
        y = []
        for index, row in train_group.iterrows():
            pen1=row['pen1']
            pen2=row['pen2']
            fij=df_metric.loc[refer,pen1]-df_metric.loc[refer,pen2]
            x.append(fij)
            y.append(row['label'])
        para, cov = optimize.curve_fit(target_func, x, y, p0=2)
        print('参数：',para)
        plt.scatter(x, y)
        tep_x = np.linspace(min(x), max(x))
        plt.plot(tep_x, [target_func(a, *para) for a in tep_x])
        plt.title('testpen:{}  refer:{}'.format(testpen,refer))
        plt.show()
        for index, row in df_test.iterrows():
            if(row['refer']==refer):
                pen1 = row['pen1']
                pen2 = row['pen2']
                fij = df_metric.loc[refer, pen1] - df_metric.loc[refer, pen2]
                df_test.loc[index,'predict']=target_func(fij, *para)
    return df_test


metric_abs()
file= '../Result/other_method_abs.xlsx'  #客观绝对打分
stylus=['iReader Smart X','苹果Pencil','微软surface','华为Pencil-透明笔尖']
rmse=pd.read_excel(file,sheet_name='rmse',index_col=0)
weibo=pd.read_excel(file,sheet_name='weibo',index_col=0)
snr=pd.read_excel(file,sheet_name='snr',index_col=0)
df = pd.read_excel('../userData/triplet-label.xlsx')#真实相对P
df['label'] = (df['picked1'] / (df['picked1'] + df['picked2']))
df.drop(['picked1','picked2'],inplace=True,axis=1)
save_path="..\\Result"
work = pd.ExcelWriter(save_path+"\\"+'other_method_Rela.xlsx')

for testpen in stylus:
    print('-------rmse--------')
    result1=aboscore_relascore(df,rmse,testpen)
    result1.rename(columns={'predict':'rmse_predict'},inplace=True)
    print('-------weibo--------')
    result2=aboscore_relascore(df,weibo,testpen)
    result2.rename(columns={'predict':'weibo_predict'},inplace=True)
    print('-------snr--------')
    result3=aboscore_relascore(df,snr,testpen)
    result3.rename(columns={'predict':'snr_predict'},inplace=True)
    merged_df = pd.merge(result1, result2, on=['pen1', 'refer', 'pen2','label'])
    merged_df = pd.merge(merged_df, result3, on=['pen1', 'refer', 'pen2','label'])
    merged_df.rename(columns={'label':'real'},inplace=True)
    merged_df.to_excel(work, sheet_name=Chi_Eng[testpen],index=None)
    print(merged_df)
work.save()
