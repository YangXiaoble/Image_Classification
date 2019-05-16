#coding=utf-8
import os
import cv2
import shutil
import random
import sys

"""
实现根据数据集进行比例分割，作为训练集与测试集(验证集方法一样)
rate:比例因子
@author:Xiaobo Yang
time:2019-4-16
"""

# 定义图片加载路径和保存路径
# oldPath = r"D:/Workspace/PycharmProjects/Image_Classification/data/train_data/"
oldPath=r"./data/train_val_test/"

dataSet_list = [ 'Apple',
                 'Bamboo',
                 'Blueberry',
                 'Camellia',
                 'Camphor',
                 'Cherry',
                 'Chestnut',
                 'Cyclobalanopsis_glauca',
                 'Cypress',
                 'Elaeocarpus_decipiens',
                 'Fir',
                 'Ginko_biloba',
                 'Grape',
                 'Locust',
                 'Mahogany',
                 'Masson_pine',
                 'Mulberry',
                 'Osmanthus',
                 'Peach',
                 'Pepper',
                 'Raspberry']

labels_list = [0,1,2,3,4,5,6,
               7,8,9,10,11,12,13,
               14,15,16,17,18,19,20
               ]

fileNumber=0
all_fileList=[]
#获取图片的总个数
for item in dataSet_list:
    data_dir=os.path.join(oldPath,item)
    fileList=os.listdir(data_dir)
    fileNumber += len(fileList)  # 图片的总个数
    all_fileList +=fileList #图片名称列表

print(fileNumber)
# print(all_fileList)

rate_1 = 0.2  # 定义抽取的比例
rate_2 =0.25
pickNumber_1 = int(fileNumber * rate_1)  # 按照rate比例从所有图片中抽取图片数量
pickNumber_2=int(fileNumber*rate_2*(1-rate_1)) #将剩下的图片分为3:1（即对于所有图片：6:2:2）
sample_1=random.sample(all_fileList,pickNumber_1) #随机选取测试样本图片

#删除测试样本，从余下图片中选取验证样本
for name in sample_1:
    all_fileList.remove(name)

sample_2=random.sample(all_fileList,pickNumber_2)#选取验证样本

label=0
f_test = open('test.txt', "w+")
f_train= open('train.txt',"w+")
f_val =open('val.txt',"w+")

for item in dataSet_list:
    data_dir=os.path.join(oldPath,item)

    # newPath=r"D:/Workspace/PycharmProjects/Image_Classification/data/test_data/"+item+'/'
    # newPath=r"./data/test_data/"+item+'/'
    # if not os.path.exists(newPath):
    #     os.mkdir(newPath)

    fileList=os.listdir(data_dir)

    for file in fileList:
        if file in sample_1:#这部分为测试集
            path = os.path.join(data_dir, file)
            append_item = path +' '+ str(labels_list[label])+'\n'
            f_test.writelines(append_item)
            print(path)
            # shutil.copy(path,newPath)#复制图片到newPath文件夹下，用于测试
        elif file in sample_2:#这部分为验证集
            append_item = os.path.join(data_dir, file) + ' ' + str(labels_list[label]) + '\n'
            f_val.writelines(append_item)

        else:#这部分为训练集
            append_item = os.path.join(data_dir, file) + ' ' + str(labels_list[label]) + '\n'
            f_train.writelines(append_item)

    label+=1

f_test.close()
f_train.close()
f_val.close()





