#coding=utf-8

#这段代码实现的功能是对文件夹中的所有图片实现重命名
#以完成制作标签的功能
"""
create on 2019-1-11

@author:Xiaobo Yang
"""

import os

#定义图片的文件夹名称
derectoryName=[ 'Apple',
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


for i in range(len(derectoryName)):

    oldPath=r'C:\\Users\\Xiaobo Yang\\Desktop\\train_data1\\' +derectoryName[i] #图片原有保存路径

    print(oldPath)

    newPath = r'C:\\Users\\Xiaobo Yang\\Desktop\\train_data2\\' + derectoryName[i]  # 图片将要保存路径

    if not os.path.exists(newPath):
        os.mkdir(newPath)

    print(newPath)

    filelist = os.listdir(oldPath) #列出当前文件夹中的所有文件


    for file in filelist:
        print(file)

    count = 1
    for file in filelist:

        Olddir=os.path.join(oldPath,file)

        if os.path.isdir(Olddir):
            continue

        filename=os.path.splitext(file)[0] #文件名
        filetype=os.path.splitext(file)[1] #文件类型

        Newdir=os.path.join(newPath,derectoryName[i]+'_'+str(count).zfill(4)+'.jpg')
        os.rename(Olddir,Newdir)

        count+=1


