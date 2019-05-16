#coding=utf-8
import cv2
import numpy as np
import os

"""
图像上下左右平移
@author:Xiaobo Yang
created time:2019-4-16
"""

#定义图片的文件夹名称
dataSet_list=[
                'Apple',
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
                'Raspberry'
                ]

for item in dataSet_list:

    picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/"+item+'/'  #定义图片加载路径和保存路径
    savePath=r"C:/Users/Xiaobo Yang/Desktop/trainData_save3/"+item+'/'

    if not os.path.exists(savePath):
        os.mkdir(savePath)

    fileList=os.listdir(picPath)


    #定义四种操作
    count=[7,8,9,10]
    for file in fileList:
        filename = os.path.splitext(file)[0]
        filetye = os.path.splitext(file)[1]

        path = os.path.join(picPath, file)

        print(path) #测试
        # 加载图像
        image = cv2.imread(path)

        rows,cols,channel=image.shape

        # print(rows)
        # print(cols)

        #x为正，向右移动；y为正，向下移动
        M1=np.float32([[1,0,int(rows/4)],[0,1,0]]) #向右平移四分之一
        #args1:待平移图片,args2:平移矩阵,args3:平移后图像的大小
        shifted_1=cv2.warpAffine(image,M1,(image.shape[1],image.shape[0]))
        cv2.imwrite(savePath + filename + '_' + str(count[0]) + filetye,shifted_1)

        M2 = np.float32([[1, 0, -int(rows / 4)], [0, 1, 0]])  # 向左平移四分之一
        # args1:待平移图片,args2:平移矩阵,args3:平移后图像的大小
        shifted_2 = cv2.warpAffine(image, M2, (image.shape[1], image.shape[0]))
        cv2.imwrite(savePath + filename + '_' + str(count[1]) + filetye, shifted_2)

        M3 = np.float32([[1, 0, 0], [0, 1, int(cols / 4)]])  # 向下平移四分之一
        # args1:待平移图片,args2:平移矩阵,args3:平移后图像的大小
        shifted_3 = cv2.warpAffine(image, M3, (image.shape[1], image.shape[0]))
        cv2.imwrite(savePath + filename + '_' + str(count[2]) + filetye, shifted_3)

        M4 = np.float32([[1, 0, 0], [0, 1, -int(cols / 4)]])  # 向上平移四分之一
        # args1:待平移图片,args2:平移矩阵,args3:平移后图像的大小
        shifted_4 = cv2.warpAffine(image, M4, (image.shape[1], image.shape[0]))
        cv2.imwrite(savePath + filename + '_' + str(count[3]) + filetye, shifted_4)

