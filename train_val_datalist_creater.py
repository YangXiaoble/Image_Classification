# coding: utf-8
import os
import sys
import argparse

"""
    为所有图片生成标签。
"""
#命令行解析器
parser = argparse.ArgumentParser(description='define train and test data')

parser.add_argument('--create_data', default='val', type=str, action='store',
                    help='train or test')

args = parser.parse_args()


database_list = ['Apple',
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

label_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# root = os.getcwd()#返回当前工作目录
root=r'D:/Workspace/PycharmProjects/leaf_recognition/data/test_data/'

f=open('train.txt', "w+") if args.create_data == 'train' else open('val.txt',"w+")
# f=open('/val.txt', "a+")

label = 0
for item in database_list:
    data_dir = os.path.join(root, item)
    for dirs in os.listdir(data_dir):
       append_item = os.path.join(data_dir,dirs)+' '+str(label_list[label])+'\n'
       f.writelines(append_item)
    label+=1

f.close()