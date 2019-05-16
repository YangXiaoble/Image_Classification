# 将其它格式转位jpg
import os
from PIL import Image
import shutil
import sys

# 定义输入和输出文件夹
input_dirHR = '.\pic'
output_dirLR = '.\SourceImage'
if not os.path.exists(input_dirHR):
    os.mkdir(input_dirHR)
if not os.path.exists(output_dirLR):
    os.mkdir(output_dirLR)


def image2Jpg(dataset_dir, type):
    """"
     实现将图片转换为jpg格式
     dataset_dir:图片保存路径
     type:图片待转换格式
    """
    files = []
    image_list = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, _) for _ in image_list]
    for index, jpg in enumerate(files):
        if index > 100000:
            break
        try:
            sys.stdout.write('\r>>Converting image %d/100000 ' % (index))
            sys.stdout.flush()
            im = Image.open(jpg)
            pic= os.path.splitext(jpg)[0] + "." + type
            im.save(pic)
            # 将已经转换的图片移动到指定位置
            ''''' 
            if jpg.split('.')[-1] == 'jpg': 
              shutil.move(png,output_dirLR) 
            else: 
              shutil.move(png,output_dirHR) 
            '''
            shutil.move(pic, output_dirLR)
        except IOError as e:
            print('could not read:', jpg)
            print('error:', e)
            print('skip it\n')

    sys.stdout.write('Convert Over!\n')
    sys.stdout.flush()


if __name__ == "__main__":
    current_dir = os.getcwd()#返回当前工作目录
    print(current_dir)
    data_dir = ".\pic"

    image2Jpg(data_dir, 'jpg')