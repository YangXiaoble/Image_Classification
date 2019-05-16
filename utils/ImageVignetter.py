#coding=utf-8
"""
实现图片暗角渐晕的特效（灵感：来源于2345画图王）
@author: Xiaobo Yang
time:2019-4-15
"""
import cv2
import numpy as np
import os

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


def ImageVignette(img):
    rows, cols, channel = img.shape
    # 定义卷积核
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T  # 转置
    # 定义掩模
    mask = 255 * kernel / np.linalg.norm(kernel)
    # 复制一个同样的image
    output = np.copy(image)
    # 每一个通道都用掩模作用一次
    for i in range(channel):
        output[:, :, i] = output[:, :, i] * mask

    return output

if __name__ == '__main__':
    for item in dataSet_list:
        picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/" + item + '/'  # 定义图片加载路径和保存路径
        savePath = r"C:/Users/Xiaobo Yang/Desktop/trainData_save5/" + item + '/'

        if not os.path.exists(savePath):
            os.mkdir(savePath)

        fileList = os.listdir(picPath)

        for file in fileList:
            filename = os.path.splitext(file)[0]
            filetye = os.path.splitext(file)[1]

            path = os.path.join(picPath, file)

            print(path)

            # 加载图像
            image = cv2.imread(path)

            # # 图像渐晕
            # img = ImageVignetting(image)
            #
            # cv2.imwrite(savePath + filename + '_' + str(1) + filetye,img)
            img=ImageVignette(image)
            cv2.imwrite(savePath + filename + '_' + str(12) + filetye, img)