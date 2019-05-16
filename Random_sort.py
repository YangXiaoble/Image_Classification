#coding=utf-8
import random


def Random_Sort():
    FileNameList=[]
    f=open('./train.txt','r+')
    for i in f:
        i=i.rstrip('\n')
        FileNameList.append(i)


    random.shuffle(FileNameList)

    w=open('./train_1.txt','w+')
    for item in FileNameList:
        w.writelines(item+'\n')

    f.close()
    w.close()

if __name__=="__main__":
    Random_Sort()


