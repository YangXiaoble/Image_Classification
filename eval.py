"""Evaluating a trained model on the test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys
from PIL import Image
import imghdr


def evaluate(args):
    """
    评估网络模型
    param args:命令解析参数
    return:评估结果
    """
    # 构建graph
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):#设置默认的cpu设备

        # 获取图像和标签
        if args.save_predictions is None:
            images, labels = data_loader.read_inputs(False, args)
        else:
            images, labels, urls = data_loader.read_inputs(False, args)

        # 在GPU上进行图像计算
        with tf.device('/gpu:0'):
            #构建一个图形，用于计算推理模型中的logits预测
            logits = arch.get_model(images, 0.0, False, args)

            #计算top-1和top-5的准确度
            top_1_op = tf.nn.in_top_k(logits, labels, 1)
            top_n_op = tf.nn.in_top_k(logits, labels, args.top_n)

            if args.save_predictions is not None:
                topn = tf.nn.top_k(tf.nn.softmax(logits), args.top_n)
                topnind= topn.indices
                topnval= topn.values

            saver = tf.train.Saver(tf.global_variables())#初始化全局变量

            # Build the summary operation based on the TF collection of Summaries.
            #定义summary保存操作
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('./log', g)
            # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


        #启动session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            sess.run(tf.global_variables_initializer())#初始化变量
            sess.run(tf.local_variables_initializer())

            # summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('./log', sess.graph)

            ckpt = tf.train.get_checkpoint_state(args.log_dir)

            # Load the latest model
            #加载最后的模型
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                #从checkpoint恢复
                saver.restore(sess, ckpt.model_checkpoint_path) #恢复ckpt文件

            else:
                return

            # 启动队列运行程序
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            true_predictions_count = 0  #计算正确预测的数量
            true_topn_predictions_count = 0#计算topn正确预测的数量
            all_count = 0 #总的数量

            step = 0#步骤数(每一个batch,增加一个step)

            #定义预测输出样式
            predictions_format_str = ('%d,%s,%d,%s,%s\n')
            batch_format_str = ('Batch Number: %d, Top-1 Hit: %d, Top-'+str(args.top_n)+' Hit: %d, Top-1 Accuracy: %.3f, Top-'+str(args.top_n)+' Accuracy: %.3f')

            #保存预测的文件存在，则打开该文件
            if args.save_predictions is not None:
                out_file = open(args.save_predictions,'w')

            #用来输出结果
            w=open('./results/val/ResNet50/ResNet50_val.csv','w+')
            write_format=('%.4f,%.4f,%d\n')

            while step < args.num_batches and not coord.should_stop():#步骤数小于batch数
            # for step in range(args.num_batches):
                # if not coord.should_stop():

                    if args.save_predictions is None:#保存预测的文件为空，执行获得top1的预测与topn的预测。
                        top1_predictions, topn_predictions = sess.run([top_1_op, top_n_op])

                    else:
                        #预测正确的文件已经存在的前提下
                        top1_predictions, topn_predictions, urls_values, label_values, topnguesses, topnconf = sess.run([top_1_op, top_n_op, urls, labels, topnind, topnval])

                        for i in range(0,urls_values.shape[0]):

                            out_file.write(predictions_format_str%(step*args.batch_size+i+1, urls_values[i], label_values[i],
                            '[' + ', '.join('%d' % item for item in topnguesses[i]) + ']',
                            '[' + ', '.join('%.4f' % item for item in topnconf[i]) + ']'))
                            out_file.flush()#清空缓存


                    true_predictions_count += np.sum(top1_predictions) #计算top1预测正确的总次数
                    true_topn_predictions_count += np.sum(topn_predictions)#计算topn预测正确的总次数
                    all_count+= top1_predictions.shape[0]#计算所有的预测次数

                    #输出格式：step,top1预测正确次数，topn预测正确次数，top1正确率和topn正确率
                    print(batch_format_str%(step, true_predictions_count, true_topn_predictions_count, true_predictions_count / all_count, true_topn_predictions_count / all_count))

                    top1_accuracy=true_predictions_count/all_count
                    tf.summary.scalar("top1_accuracy",top1_accuracy) #记录top1精度
                    top5_accuracy=true_topn_predictions_count/all_count
                    tf.summary.scalar("top5_accuracy",top5_accuracy)#记录top5精度

                    sys.stdout.flush()#清空缓存

                    step += 1

                    w.write(write_format%(top1_accuracy,top5_accuracy,step))


                #
                # summary_str = sess.run(summary_op)
                # summary_writer.add_graph(sess.graph)
                # summary_writer.add_summary(summary_str,step)


            w.close()

            if args.save_predictions is not None:
                out_file.close()

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))

            coord.request_stop()
            coord.join(threads)



def get_sess(args):
    """
        启动session
    """
    with tf.Graph().as_default() as g, tf.device('/gpu:0'):#设置默认gpu

        input = tf.placeholder(dtype=tf.float32, shape=(None,args.load_size[0] ,args.load_size[1],args.num_channels))
        logits = arch.get_model(input, 0.0, False,args)#获取网络模型：（输入，权重，不是训练，命令解析参数）

        #设置内存需求自动增长，启动session
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        #加载checkpoint文件
        ckpt = tf.train.get_checkpoint_state('./ResNet_Run-12-05-2019_02-04-24')  #这里

        if ckpt and ckpt.model_checkpoint_path:
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print(e)
                print('Load Model ERROR ERROR!!!!!!!!!!!!!!!!')
                sys.exit()
        else:
            print('model does not exist!! ,The program is going to quit!')
            sys.exit()

        return sess, logits, input


def standardization(img,width=400,height=400,channel=3):
    """
    图像标准化
    param img:图像
    param width:图像的宽
    param height: 图像的高
    param channel:颜色通道
    return:标准化后的图像
    """
    num_compare = width*height*channel
    img_arr = np.array(img)
    img_std = (img_arr - np.mean(img_arr)) / max(np.std(img_arr), 1 / math.sqrt(num_compare))
    return img_std


#获取标签
def read_lables(file,delimiter):
    f = open(file, "r")  #以写的方式打开文件
    labels = []  #标签
    images=[]
    for line in f:
        tokens = line.split(delimiter) #以分隔符划分文件路径和文件的标签
        images.append(tokens[0])
        labels.append(int(tokens[1]))

    # f.close()
    return images,labels


def test(args,image_folder):
    """
    实现测试
    param args:命令解析参数
    param image_folder: 测试图片的路径
    return:输出最终结果
    """
    sess, logits, input = get_sess(args)

    # labels=read_lables(args.data_info,args.delimiter) #这里添加标签
    images,labels=read_lables('./test_1.txt',args.delimiter)  #这里加载标签

    print(images)
    print(labels)

    # f_error=open("error.txt","+w")
    true_predictions_count=0

    step=0

    all_count=0

    #将测试结果写入文件
    w=open('./results/test/ResNet50/ResNet50_test.csv','w+')

    write_format=('%.4f,%d\n')

    # summary_op = tf.summary.merge_all()

    # summary_writer = tf.summary.FileWriter('./test', sess)

    # for root, dirs, files in os.walk(image_folder):#遍历测试集中图片
    for i,item in enumerate(images):
        # for file in files:
        #     item = os.path.join(root, file)
            try:
                #item =sys.unicode(item, 'utf-8')
                 item =str(item)
            except:
                continue

            out = 0 #预测的结果

            try:
                #判断图像格式是否符合要求
                # if imghdr.what(os.path.join(root, file)) not in {'jpeg','jpg','rgb','gif','tif','bmp','png'}:
                #     print('invalid image: {}'.format(os.path.join(root, file)))
                #     continue
                if imghdr.what(item) not in {'jpeg','jpg','rgb','gif','tif','bmp','png'}:
                    print('invalid image: {}'.format(item))
                    continue

                img = Image.open(item) #加载图像
                img = img.resize(args.load_size, Image.ANTIALIAS)#插值进行图像缩放成固定大小

                # 图像进行标准化
                std_img = standardization(img)
                imgs = [std_img]

                test_list = sess.run([logits],feed_dict={input: imgs})

                out = np.argmax(test_list)#预测的结果

                # 计算正确率（将预测值与正确值比较）
                if int(out)==int(labels[i]):
                    true_predictions_count+=1

                # else:
                #     f_error.writelines(item+'\n')

                all_count+=1

                print('The out is:', out)
                print('The labels is:', labels[all_count])

                TestAccuracy = true_predictions_count / all_count

                # if all_count % 100==0:
                w.write(write_format%(TestAccuracy,all_count))
                # tf.summary.scalar('TestAccuracy', TestAccuracy)

                print('The last accuracy is:%.6f' %TestAccuracy)
                #
                print('the true predictions is:', true_predictions_count)
                print('the all account is:', all_count)

                # for step in range(all_count):
                #     # if step % 1000 == 0:
                #     summary_str = sess.run(summary_merge)
                #     summary_writer.add_summary(summary_str, step)
                # summary = tf.Summary()
                # summary_str=summary.ParseFromString(sess.run(summary_op))
                # summary_str = sess.run(summary_op)

                # if step % 1000==0: #测试一千张，记录一次
                #     summary_str=sess.run(summary_op)
                #     summary_writer.add_summary(summary_str,step)

            except Exception as e:
                print(e)
                print('evaluate ERROR!!')

            finally:
                print(item,' Test result: ',out)
                print("#######################")


    w.close()
    # summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    # f_error.close()

def main():
    
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')
    parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')#加载图片大小
    parser.add_argument('--crop_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images after random cropping')#随机裁剪图片大小
    parser.add_argument('--batch_size', default= 24, type= int, action= 'store', help= 'The testing batch size') #测试的batch大小
    parser.add_argument('--num_classes', default= 21, type= int, action= 'store', help= 'The number of classes')#类的数量
    parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images') #颜色通道数
    parser.add_argument('--num_batches' , default=-1 , type= int, action= 'store', help= 'The number of batches of data')
    parser.add_argument('--path_prefix' , default='./', action= 'store', help= 'The prefix address for images')
    parser.add_argument('--delimiter' , default=' ', action = 'store', help= 'Delimiter for the input files')
    parser.add_argument('--data_info'   , default= 'val_1.txt', action= 'store', help= 'File containing the addresses and labels of testing images')
    parser.add_argument('--num_threads', default= 4, type= int, action= 'store', help= 'The number of threads for loading data')
    parser.add_argument('--architecture', default= 'ResNet', help='The DNN architecture')
    # parser.add_argument('--architecture', default='googlenet', help='The DNN architecture')
    parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
    parser.add_argument('--log_dir', default= './ResNet_Run-12-05-2019_02-04-24', action= 'store', help='Path for saving Tensorboard info and checkpoints')
    parser.add_argument('--save_predictions', default= None, action= 'store', help= 'Save top-5 predictions of the networks along with their confidence in the specified file')
    parser.add_argument('--top_n', default= 5, type= int, action= 'store', help= 'Specify the top-N accuracy')
    parser.add_argument('--eval_model', default=True, type=bool, action='store', help='evaluate acc of  model')
    parser.add_argument('--test_images_path', default=None, type=str, action='store', help='test images')
    # parser.add_argument('--test_images_path', default='./data/train_data', type=str, action='store', help='val images')

    args = parser.parse_args()
    args.num_samples = sum(1 for line in open(args.data_info)) #测试样本个数
    if args.num_batches==-1: #计算batch的个数
        if(args.num_samples%args.batch_size==0):
            args.num_batches= int(args.num_samples/args.batch_size)
        else:
            args.num_batches= int(args.num_samples/args.batch_size)+1

    print(args)



    if args.eval_model:
        evaluate(args)
    if args.test_images_path != None:
        test(args,image_folder=args.test_images_path)
        # test(args,image_folder='./data/test_data/')


if __name__ == '__main__':
    main()