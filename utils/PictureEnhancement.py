#coding=utf-8
"""
实现图像增强的功能
author：Xiaobo Yang
time:2019-4-10
"""
#pip install piexif
import os
from PIL import Image
from PIL import ImageEnhance

#定义图片的文件夹名称
dataSet_list=[
                # 'Apple',
                'Bamboo',
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
                # 'Locust',
                # 'Mahogany',
                # 'Masson_pine',
                # 'Mulberry',
                # 'Osmanthus',
                # 'Peach',
                # 'Pepper',
                # 'Raspberry'
                ]


for item in dataSet_list:

    picPath = r"C:/Users/Xiaobo Yang/Desktop/train_test/"+item+'/'  #定义图片加载路径和保存路径
    savePath=r"C:/Users/Xiaobo Yang/Desktop/train_test1/"+item+'/'

    if not os.path.exists(savePath):
        os.mkdir(savePath)

    fileList=os.listdir(picPath)

    for file in fileList:


        filename=os.path.splitext(file)[0]
        filetye=os.path.splitext(file)[1]

        path=os.path.join(picPath,file)

        print(path)#用于测试


        #加载图像
        image = Image.open(path)

        #定义四种操作
        count=[1,2,3,4]

        # 亮度增强
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.5 #强度系数
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.save(savePath+filename+'_'+str(count[0])+filetye)

        # # 色度增强
        # enh_col = ImageEnhance.Color(image)
        # color = 1.5
        # image_colored = enh_col.enhance(color)
        # image_brightened.save(savePath + filename + '_' + str(count[1]) + filetye)
        #
        # # 对比度增强
        # enh_con = ImageEnhance.Contrast(image)
        # contrast = 2
        # image_contrasted = enh_con.enhance(contrast)
        # image_brightened.save(savePath + filename + '_' + str(count[2]) + filetye)
        #
        # # 锐度增强
        # enh_sha = ImageEnhance.Sharpness(image)
        # sharpness = 3.0
        # image_sharped = enh_sha.enhance(sharpness)
        # image_brightened.save(savePath + filename + '_' + str(count[3]) + filetye)


