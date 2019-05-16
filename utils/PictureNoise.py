#coding=utf-8
"""
实现图像高斯加噪与椒盐加噪
"""
import numpy as np
from numpy import shape
import random
import os
import cv2

#定义图片的文件夹名称
dataSet_list=[ 'Apple',
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


#椒盐噪声
def PepperAndSaltNoise(img,percentage):
    NoiseImage=img
    NOiseNum=int(percentage*img.shape[0]*img.shape[1])
    for i in range(NOiseNum):
        randX=random.randint(0,img.shape[0]-1)
        randY=random.randint(0,img.shape[1]-1)

        if random.random()<=0.5:
            NoiseImage[randX,randY]=0
        else:
            NoiseImage[randX,randY]=255
    return NoiseImage


#随机生成符合正太（高斯）分部的随机数，means,siama为两个参数
def GaussianNoise(img,means,sigma,percentage):
    NoiseImg=img
    NOiseNum = int(percentage * img.shape[0] * img.shape[1])

    for i in range(NOiseNum):
        randX=random.randint(0,img.shape[0]-1)
        randY=random.randint(0,img.shape[1]-1)

        #此处在原有像素值上加上随机数
        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)

        #若像素值小于0，则强制为0；若大于255,则强制为255
        if NoiseImg[randX,randY].all()< 0:
            NoiseImg[randX,randY]=0
        elif NoiseImg[randX,randY].all() > 255:
            NoiseImg[randX,randY]=255

    return NoiseImg

if __name__ == '__main__':

    inputNumber = input('请输入1或2:') #1：执行椒盐噪声操作 2：高斯噪声操作

    for item in dataSet_list:

        picPath=r"C:/Users/Xiaobo Yang/Desktop/trainData/"+item+'/'
        newPath=r"C:/Users/Xiaobo Yang/Desktop/trainData_save2/"+item+'/'

        if not os.path.exists(newPath):
            os.mkdir(newPath)

        filelist=os.listdir(picPath)

        for file in filelist:
            filename=os.path.splitext(file)[0]
            filetype=os.path.splitext(file)[1]

            path=os.path.join(picPath,file)

            print(path)

            img=cv2.imread(path)

            if inputNumber=='1':
                img1 = PepperAndSaltNoise(img, 0.03)
                cv2.imwrite(newPath + filename + '_' + str(5) + filetype, img1)
            elif inputNumber=='2':
                img2 = GaussianNoise(img, 2, 16, 0.5)
                cv2.imwrite(newPath + filename + '_' + str(6) + filetype, img2)
            else:
                break




