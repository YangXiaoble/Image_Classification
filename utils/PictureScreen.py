#coding=utf-8
from PIL import Image
import os
# import cv2
import shutil

"""
实现根据图片大小对图片进行筛选的功能
@author:Xiaobo Yang
time:2019-4-16
"""


dataSet_list = ['Apple',
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


for item in dataSet_list:
    picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/" + item + '/'  # 定义图片加载路径和保存路径
    savePath = r"C:/Users/Xiaobo Yang/Desktop/trainData_save7/" + item + '/'

    if not os.path.exists(savePath):
        os.mkdir(savePath)

    fileList = os.listdir(picPath)

    for file in fileList:

        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]

        data_dirs = os.path.join(picPath, file)
        # 加载图像
        image= Image.open(data_dirs)
        # image=cv2.imread(picPath)

        print(data_dirs)

        # reshape= image.shape
        # rows=reshape[0]
        # cols=reshape[1]

        cols,rows=image.size

        try:
            if cols <= 160 or rows <= 120:
                # args1:old position,args2:new position
                shutil.move(data_dirs, savePath)
        except Exception as e:
                print(e)
                continue



