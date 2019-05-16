#coding=utf-8

#coding=utf-8

#这段代码实现的功能是对文件夹中的所有图片实现重命名
#以完成制作标签的功能
"""
create on 2019-1-11

@author:Xiaobo Yang
"""

import os


path = "./data/train_data/Apple"  #图片存放路径
print(path)
filelist = os.listdir(path) #列出当前文件夹中的所有文件

count=1
for file in filelist:
    print(file)

for file in filelist:
    Olddir=os.path.join(path,file)

    if os.path.isdir(Olddir):
        continue

    filename=os.path.splitext(file)[0] #文件名
    filetype=os.path.splitext(file)[1] #文件类型
    Newdir=os.path.join(path,"Apple_"+str(count).zfill(5)+filetype)
    os.rename(Olddir,Newdir)

    count+=1


