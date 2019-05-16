#coding=utf-8
""""
    date:2019-3-8
    @auther:Xiaobo Yang
    实现图片缩放的功能
"""
import cv2
import numpy as np
import os
"""
    利用缩放尺度因子进行缩放
    resize():args1:待缩放的图像，args2:缩放后的图像，args and args4:缩放尺度因子，args5:插值方法
    其中：
        插值方法:默认使用：cv2.INTER_LINEAR(双线性插值)
                 缩小是推荐使用：cv2.INTER_AREA（使用像素区域关系进行重采样）
                 扩大时推荐使用：cv2.INTER_CUBIC（4X4像素邻域的双三次插值） and cv2.INTER_LINEAR(前者比后者速度慢)
                 cv2.INTER_NEAREST:最近邻插值,cv2.INTER_LANCZOS4:8x8像素邻域的Lanczos插值

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


#利用缩放因子进行缩放
def scaleFactor(img):
    #None:输出图像的尺寸大小,fx和fy是缩放因子
    img1=cv2.resize(img,None,fx=3,fy=3,interpolation=cv2.INTER_LANCZOS4)
    return img1

#固定尺寸缩放
def fixedSize(img):
    height,width=img.shape[:2]
    img1=cv2.resize(img,(4*width,4*height),interpolation=cv2.INTER_LINEAR)
    return img1

if __name__ == '__main__':

    for item in dataSet_list:

        picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/" + item + '/'  # 定义图片加载路径和保存路径
        savePath = r"C:/Users/Xiaobo Yang/Desktop/trainData_save7/" + item + '/'

        if not os.path.exists(savePath):
            os.mkdir(savePath)

        fileList = os.listdir(picPath)

        for file in fileList:

            print(file)

            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]


            path = os.path.join(picPath, file)
            # 加载图像
            image= cv2.imread(path)

            try:
                reshape=image.shape
                rows=reshape[0]
                cols=reshape[1]
            except Exception as e:
                print(e)

            if rows<=120 or cols<=160:

                # img=scaleFactor(image)
                img1=fixedSize(image)
                cv2.imwrite(savePath + filename + '_' + str(22) + filetype,img1)

                img2=scaleFactor(image)
                cv2.imwrite(savePath + filename + '_' + str(23) + filetype,img2)