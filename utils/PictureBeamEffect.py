#coding=utf-8
'''
实现光束特效
author:Xiaobo Yang
time:2019-4-15
'''
import numpy as np
from PIL import Image
import os

ALPHA = 1.2
BETA = 30

#定义图片的文件夹名称
dataSet_list=[
                # 'Apple',
                # 'Bamboo',
                # 'Blueberry',
                # 'Camellia',
                # 'Camphor',
                # 'Cherry',
                # 'Chestnut',
                # 'Cyclobalanopsis_glauca',
                # 'Cypress',
                # 'Elaeocarpus_decipiens',
                # 'Fir',
                # 'Ginko_biloba',
                # 'Grape',
                'Locust',
                'Mahogany',
                'Masson_pine',
                'Mulberry',
                'Osmanthus',
                # 'Peach',
                # 'Pepper',
                # 'Raspberry'
                ]



#Alpha:决定对比度
#beta:决定亮度
def light_adjust(img, a, b):
    c, r = img.size
    arr = np.array(img)
    for i in range(r):
        for j in range(c):
            for k in range(3):
                temp = arr[i][j][k] * a + b
                if temp > 255:
                    arr[i][j][k] = 2 * 255 - temp
                else:
                    arr[i][j][k] = temp
    return arr


if __name__ == '__main__':
    for item in dataSet_list:

        picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/" + item + '/'  # 定义图片加载路径和保存路径
        savePath = r"C:/Users/Xiaobo Yang/Desktop/trainData_save4/" + item + '/'

        if not os.path.exists(savePath):
            os.mkdir(savePath)

        fileList = os.listdir(picPath)

        for file in fileList:
            filename = os.path.splitext(file)[0]
            filetye = os.path.splitext(file)[1]

            path = os.path.join(picPath, file)
            print(path)
            # 加载图像
            img= Image.open(path)

            arr = light_adjust(img, ALPHA, BETA)
            img1 = Image.fromarray(arr)
            img1.save(savePath + filename + '_' + str(11) + filetye)


