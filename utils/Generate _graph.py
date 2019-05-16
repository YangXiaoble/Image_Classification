#coding=utf-8
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv


#需要绘制6中曲线图
# labels=["Corss_Entropy_Loss"
#         "Regularization_Loss",
#         "Total_Loss",
#         "Top1_Accuracy",
#         # "Learning_rate",
#         # "Weights_decay"
#         ]
"""
读取csv文件
"""

def readcsv(files):
    csvfile=open(files,'r')
    plots=csv.reader(csvfile,delimiter=',')
    x=[]
    y=[]

    for row in plots:
        # x.append((row[1]))
        # y.append((row[2]))
        x.append((row[1]))
        y.append((row[0]))

    return x,y

#设置字体属性
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']='NSimSun,Times New Roman'

# for i in range(labels):

plt.figure() #创建第i面板

    # ax1=plt.subplot(2,3,i)

    # plt.sca(ax1)

x, y = readcsv("../results/test/ResNet50/ResNet50_test.csv")
plt.plot(x,y,color='red',label='ResNet50')
    # x,y=readcsv("./results/"+labels[i]+"/resnet50.csv")
    # plt.plot(x,y,color='green',label='ResNet50')
    #
    # x2,y2=readcsv("./results/"+labels[i]+"/googlenet.csv")
    # plt.plot(x2,y2,color='red',label='GoogLeNet V3')
    #
    # x3,y3=readcsv("./results/"+labels[i]+"/vgg16.csv")
    # plt.plot(x3,y3,color='blue',label='VGG16')
    #
    # x4,y4=readcsv("./results/"+labels[i]+"/alexnet.csv")
    # plt.plot(x4,y4,color='yellow',label='AlexNet')
    #
    # x5,y5=readcsv("./results/"+labels[i]+"/densenet.csv")
    # plt.plot(x5,y5,color='black',label='DenseNet')

    #x,y轴范围
plt.ylim(0,1)
plt.xlim(0,1900) #这里根据实际情况变动

plt.xlabel('Steps',fontsize=20)
    # plt.ylabel(item,fontsize=16)
plt.ylabel('Val-Accuracy',fontsize=16)
plt.legend(fontsize=16)


plt.title('(a)')
    #设置绘图的标题
    # if labels[i]=="Corss_Entropy_Loss":
    #     plt.title('(a)',loc='bottom')
    #
    # elif labels[i]=="Regularization_Loss":
    #     plt.title('(b)',loc="bottom")
    #
    # elif labels[i]=="Total_Loss":
    #     plt.title('(c)',loc="bottom")
    #
    # elif labels[i]=="Top1_Accuracy":
    #     plt.title('(d)',loc="bottom")

    # elif labels[i]=="Learning_rate":
    #     plt.title('(e)',loc="bottom")
    #
    # elif labels[i]=="Weights_decay":
    #     plt.title('(f)',loc="bottom")

plt.show()

