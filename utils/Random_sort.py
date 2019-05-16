#coding=utf-8
import random


def Random_Sort():
    FileNameList=[]
    f=open('../test.txt','r+')
    for i in f:
        i=i.rstrip('\n')
        FileNameList.append(i)


    random.shuffle(FileNameList)

    w=open('../test_1.txt','w+')
    for item in FileNameList:
        w.writelines(item+'\n')

    f.close()
    w.close()

if __name__=="__main__":
    Random_Sort()


