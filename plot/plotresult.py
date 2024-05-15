import os
import random
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr,kendalltau
import matplotlib as mpl
plt.rcParams['font.sans-serif']=['Times New Roman']#SimHei，Times New Roman
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=600
mpl.rcParams['font.size'] = 8
plt.rcParams['figure.figsize']=(3.5,4)

pennames=['ireader','苹果Pencil','surface','透明笔尖']#'ireader','surface','苹果Pencil','透明笔尖','1','2','3','4'
Chi_Eng={'iReader Smart X':'iReader Smart X','苹果Pencil':'Apple Pencil','华为Pencil-透明笔尖':'M-Pencil','微软surface':'Microsoft Surface Pen',
        'A4纸10-中性笔':"gel+A4",'A4纸10-圆珠笔':"ball-point+A4",'A4纸10-铅笔HB':"HB+A4",'A4纸10-马克笔':"Marker+A4"}
Eng_Chi={}
for key,value in Chi_Eng.items():
    Eng_Chi[value]=key
stylus=['iReader Smart X','Apple Pencil','Microsoft Surface Pen','M-Pencil']

#-----------计算模型的各个指标-----------
def modelmetric(all_predict):
    real=all_predict['real']
    for colum_name in all_predict.columns[4:all_predict.shape[1]]:
        print('----------object method:',colum_name)

        predict = all_predict[colum_name]

        mean_deviation = (real - predict).abs().mean()

        pearson_corr, pearson_p = pearsonr(real, predict)
        kendall_corr, kendall_p  = kendalltau(real, predict)

        spearman_corr, spearman_corr_p = spearmanr(real, predict)
        print("Mean Deviation:", round(mean_deviation,2))
        print("Pearson Correlation Coefficient:", round(pearson_corr,2), pearson_p)
        print("Spearman's Rank Correlation Coefficient:", round(spearman_corr,2), spearman_corr_p)
        print("Kendall Tau相关系数:", round(kendall_corr,2), kendall_p)
#---------汇总结果---------
def save_Summery(savefile):
    myresult_file= '../Result/MyResultAver.xlsx'
    otherresult_file= '../Result/other_method_Rela_11hz.xlsx'
    tog_file= '../Result/TOG_result2.xlsx'
    worksheet=pd.ExcelWriter(savefile)
    for sheet_name in stylus:
        myresult=pd.read_excel(myresult_file,sheet_name=sheet_name,usecols=range(0,5))
        otherresult = pd.read_excel(otherresult_file,sheet_name=sheet_name)
        togresult = pd.read_excel(tog_file,sheet_name=sheet_name,usecols=range(0,5))
        merge = myresult.copy()
        merge['rmse_pridict']=pd.Series()
        merge['weibo_pridict']=pd.Series()
        merge['snr_pridict'] = pd.Series()
        merge['2D_space_predict'] = pd.Series()
        for idx,row in myresult.iterrows():
            refer=row['refer']
            pen1=row['pen1']
            pen2=row['pen2']
            right=otherresult[(otherresult['refer']==refer)&(otherresult['pen1']==pen1)&(otherresult['pen2']==pen2)]
            D_space=togresult[(togresult['refer']==Chi_Eng[refer])&(togresult['pen1']==Chi_Eng[pen1])&(togresult['pen2']==Chi_Eng[pen2])]
            merge.iloc[idx,[5,6,7]]=right.iloc[0,[4,5,6]]
            merge.iloc[idx, 8] = D_space.iloc[0, 4]
        print('Test Stylus:',sheet_name)
        print(merge)
        modelmetric(merge)
        merge.to_excel(worksheet,sheet_name=sheet_name,index=None)

    worksheet.save()



def plotcorrects(file):
    result=pd.read_excel('Result/togVSmy.xlsx')
    result['mydia']=(result.iloc[:,3]-result.iloc[:,4]).abs()
    result['togdia']=(result.iloc[:,3]-result.iloc[:,6]).abs()
    pennames=['ireader','surface','apple Pencil','huawei']
    my_mean=[]
    my_var=[]
    tog_mean=[]
    tog_var=[]
    d_percent={}
    for i,testpen in enumerate(pennames):
        data=result.iloc[i*24:(i+1)*24]
        my_mean.append(data['mydia'].mean())
        my_var.append(data['mydia'].var())
        tog_mean.append(data['togdia'].mean())
        tog_var.append(data['togdia'].var())
        my_l=[]
        tog_l=[]
        for row,margin in enumerate(margins):
            count = data['mydia'] < margin  # 返回一个Series，包含每个元素是否小于5的布尔值
            count = count.sum()/data.shape[0]  # 对布尔值进行求和，得到小于5的数量
            my_l.append(count)
            count = data['togdia'] < margin  # 返回一个Series，包含每个元素是否小于5的布尔值
            count = count.sum()/data.shape[0]  # 对布尔值进行求和，得到小于5的数量
            tog_l .append(count)
        d_percent[testpen]={'my':my_l,'tog':tog_l}
    fig = plt.figure(figsize=(3.5, 4),dpi=600)
    barwidth=0.4
    x1=[i for i in range(len(pennames))]
    x2=[i+barwidth for i in range(len(pennames))]
    print('my_mean',my_mean)
    plt.bar(x1, my_mean, width=barwidth, label='deep-net')
    plt.bar(x2 ,tog_mean, width=barwidth, label='hand-craft')
    plt.errorbar(x=x1
                ,y=my_mean
               ,yerr=my_var
               ,fmt='.'
               ,color='r'
               ,elinewidth=1#线宽
               ,capsize=3)#横线长度
    plt.errorbar(x=x2
               ,y=tog_mean
               ,yerr=tog_var
                ,fmt='.'
               ,color='r'
               ,elinewidth=1#线宽
               ,capsize=3)#横线长度
    plt.xticks([x+barwidth/2 for x in range(len(pennames))], pennames)
    plt.ylim(0,0.7)
    plt.ylabel('average deviation')
    plt.legend()
    plt.savefig('testT.png')
    # plt.show()
    fig = plt.figure(figsize=(3.5, 4),dpi=600)
    linestyle=['--','-.',':','-']
    for i,pen in enumerate(pennames):
        line1,=plt.plot(margins,d_percent[pen]['my'],color='royalblue',linestyle=linestyle[i],label=pen)
        plt.plot(margins,d_percent[pen]['tog'], color='orange',linestyle=linestyle[i])
    plt.xlabel(r'$\beta$')
    plt.ylabel('Percentage')
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # plot(grid=True,ylim=(0,1.05),xlim=(0,0.9))
    plt.legend()
    legend = plt.gca().get_legend()
    legend.get_lines()[0].set_color('black')
    legend.get_lines()[1].set_color('black')
    legend.get_lines()[2].set_color('black')
    legend.get_lines()[3].set_color('black')
    plt.savefig('test2T.png')


def new_plot_corrects(savepath,file='../Result/All.xlsx',save=False):
    for styl in stylus:
        df=pd.read_excel(file,sheet_name=styl,usecols=[3,4])
        corrects = []
        for threshold in margins:
            count = len(df[abs(df['real'] - df['predict']) <= threshold])
            correct = count / len(df)
            corrects.append(correct)
        # 绘制阈值与记录比例的图表
        plt.plot(margins, corrects,label=styl,linewidth=0.9)
    plt.xlabel(r'Tolerance error $\delta$',fontsize=8)#容错误差
    plt.ylabel('Accuracy',fontsize=8)#准确率
    plt.legend(prop={'size': 8})
    fig = plt.gcf()
    plt.grid(alpha=0.5)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],fontsize=8)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],fontsize=8)
    plt.tight_layout()

    fig.set_size_inches(3, 2)  # 设置图像大小为6x4英寸
    fig.savefig(savepath+'/Accuracy.png', dpi=600,bbox_inches='tight')

    plt.show()

def label_distribute(savepath,file=r'..\\userData/triplet-label.xlsx'):
    data = pd.read_excel(file, usecols=['picked1','picked2'])
    data['label']=data['picked1']/(data['picked1']+data['picked2'])
    counts, bins, patches=plt.hist(data['label'], bins=5, rwidth=0.6)
    fig = plt.gcf()
    fig.set_size_inches(3, 3)  # 设置图像大小为6x4英寸
    plt.xticks([])
    print(counts,bins)

    #在横坐标位置添加文本
    for i, xi in enumerate(bins[:-2]):
        plt.text(xi+0.1, -2, f'[{round(xi,2)},{round(bins[i+1],2)})', ha='center',fontsize=7)
    plt.text(0.9, -2.5, '[0.8,1]', ha='center', fontsize=7)
    plt.text(0.5, -5.5, 'Label', ha='center', fontsize=8)#
    plt.ylabel('Count')#
    plt.yticks(np.arange(0,30,5), fontsize=8)
    # plt.grid(alpha=0.5)
    plt.show()
    fig.savefig(savepath + '/label.png', dpi=600, bbox_inches='tight')
if __name__ == '__main__':
    savepath=r'C:/Users/Administrator/Desktop/IEEE-Trans-Template'
    margins = np.linspace(0, 0.8, 60)
    file= '../Result/MyResultAver.xlsx'
    # df_predict=pd.ExcelFile(file)
    new_plot_corrects(savepath=savepath,file=file,save=True)

    # save_Summery(savefile=file)
