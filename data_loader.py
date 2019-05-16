#coding=utf-8

#导入需要的包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange
import tensorflow as tf

# Parse the input file name and label
"""
解析导入的文件名和标签

    Args:
        file:文件
        delimiter:分割符
"""
def _read_label_file(file, delimiter):
    f = open(file, "r")#以写的方式打开文件
    filepaths = [] #文件路径
    labels = [] #标签

    for line in f:
        tokens = line.split(delimiter) #以分隔符划分文件路径和文件的标签
        filepaths.append(tokens[0]) #
        labels.append(int(tokens[1]))
    return filepaths, labels


def read_inputs(is_training, args):
    """
    加载图像和标签
    param is_training: True or false
    param args: 命令行中输入的参数，（batch_size......）
    return:
    """
    filepaths, labels = _read_label_file(args.data_info, args.delimiter) #data_info:文件地址以及标签，delimiter:分隔符

    # get absolute path of images
    #获取图片的绝对路径
    filenames = [os.path.join(args.path_prefix,i) for i in filepaths]

    # Create a queue that produces the filenames to read.
    #创建一个队列以用来生成要读取的文件名
    if is_training:
        filename_queue = tf.train.slice_input_producer([filenames, labels], shuffle= args.shuffle, capacity= 1024)
    else:
        filename_queue = tf.train.slice_input_producer([filenames, labels], shuffle= False,  capacity= 1024, num_epochs =1)

    # Read examples from files in the filename queue.
    #从文件名队列中的文件中读取示例。
    file_content = tf.read_file(filename_queue[0])
    # Read JPEG or PNG or GIF image from file
    #从文件中读取JPEG,GIF,GIF图像
    reshaped_image = tf.to_float(tf.image.decode_jpeg(file_content, channels=args.num_channels))#通道的个数，RGB=3
    # Resize image to args.load_size, default= [width,height]
    #重新构造加载的图像的大小，采用默认的大小
    reshaped_image = tf.image.resize_images(reshaped_image, args.load_size)

    # change to tf.int64
    #改为int64类型
    label = tf.cast(filename_queue[1], tf.int64) #标签
    img_info = filename_queue[0] #文件信息

    if is_training:
        reshaped_image = _train_preprocess(reshaped_image, args)
    else:
        reshaped_image = _test_preprocess(reshaped_image, args)
     # Ensure that the random shuffling has good mixing properties.
     #确保随机改组具有良好的混合特性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(5000*min_fraction_of_examples_in_queue)

    #print(batch_size)
    # print ('Filling queue with %d images before starting to train. '
    #        'This may take some times.' % min_queue_examples)
    batch_size = args.chunked_batch_size if is_training else args.batch_size

    # Load images and labels with additional info
    # 加载图像和标签以及额外的信息
    if hasattr(args, 'save_predictions') and args.save_predictions is not None:

        images, label_batch, info = tf.train.batch(
            [reshaped_image, label, img_info],
            batch_size= batch_size,
            num_threads=args.num_threads,#线程数量
            capacity=min_queue_examples+3 * batch_size,#容量
            allow_smaller_final_batch=True if not is_training else False) #
        return images, label_batch, info
    else:
        images, label_batch = tf.train.batch(
            [reshaped_image, label],
            batch_size= batch_size,
            allow_smaller_final_batch= True if not is_training else False,
            num_threads=args.num_threads,
            capacity=min_queue_examples+3 * batch_size)
        return images, label_batch


#训练的预处理
def _train_preprocess(reshaped_image, args):
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    #用于训练网络的图像处理。 请注意应用于图像的许多随机失真。

    # Randomly crop a [height, width] section of the image.
    #随机裁剪固定大小的图像
    reshaped_image = tf.random_crop(reshaped_image, [args.crop_size[0], args.crop_size[1], args.num_channels]) #裁剪图像的大小以及RGB个数

    # Randomly flip the image horizontally.
    # 随机的水平翻转图像
    # reshaped_image = tf.image.random_flip_left_right(reshaped_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #  因为这些操作不是可交换的，所以考虑随机化操作顺序
    reshaped_image = tf.image.random_brightness(reshaped_image,
                                                 max_delta=63)
    # Randomly changing contrast of the image
    #随便改变图像的对比度
    reshaped_image = tf.image.random_contrast(reshaped_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    #减去均值并除于像素的方差
    reshaped_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    #设置张量的形状
    reshaped_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])
    #read_input.label.set_shape([1])
    return reshaped_image


#测试的预处理
def _test_preprocess(reshaped_image, args):
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    #图片处理用来评估
    #裁剪图片
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           args.crop_size[0], args.crop_size[1])

    # Subtract off the mean and divide by the variance of the pixels.
    #减去平均值并除于像素的方差
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    #设置张量的形状
    float_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])

    return float_image

