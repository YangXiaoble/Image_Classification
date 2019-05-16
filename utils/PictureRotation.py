#coding=utf-8
"""
实现图片的翻转与旋转
author:Xiaobo Yang
time:2019-4-10
"""
import PIL.Image as img
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

for item in dataSet_list:

    picPath = r"C:/Users/Xiaobo Yang/Desktop/trainData/"+item+'/'  #定义图片加载路径和保存路径
    newPath=r"C:/Users/Xiaobo Yang/Desktop/trainData_save6/"+item+'/'

    if not os.path.exists(newPath):
        os.mkdir(newPath)

    fileList=os.listdir(picPath)

    for file in fileList:

        filename=os.path.splitext(file)[0] #文件名
        filetype=os.path.splitext(file)[1] #文件类型

        #加载每一张图片
        path=os.path.join(picPath,file)

        print(path)

        im = img.open(path)

        #对图片执行的步骤：左右对称，上下对称，以及对图片每隔45度旋转一次，共9次操作。
        count=[13,14,15,16,17,18,19,20,21]

        ng=im.transpose(img.FLIP_LEFT_RIGHT)#图片左右对称
        ng.save(newPath+filename+'_'+str(count[0])+filetype)

        ng=im.transpose(img.FLIP_TOP_BOTTOM)#图片上下对称
        ng.save(newPath + filename + '_' + str(count[1]) + filetype)

        ng=im.rotate(45)
        ng.save(newPath + filename + '_' + str(count[2]) + filetype)

        ng=im.transpose(img.ROTATE_90)#旋转90度角
        ng.save(newPath + filename + '_' + str(count[3]) + filetype)

        ng = im.rotate(135)
        ng.save(newPath + filename + '_' + str(count[4]) + filetype)

        ng=im.transpose(img.ROTATE_180) #旋转 180 度角
        ng.save(newPath + filename + '_' + str(count[5]) + filetype)

        ng = im.rotate(225)
        ng.save(newPath + filename + '_' + str(count[6]) + filetype)
        ng=im.transpose(img.ROTATE_270) #旋转270度角
        ng.save(newPath + filename + '_' + str(count[7]) + filetype)

        ng = im.rotate(315)
        ng.save(newPath + filename + '_' + str(count[8]) + filetype)

        # ng = im.rotate(45) #逆时针旋转 45 度角。
