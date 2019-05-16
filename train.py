"""
A program to train different architectures(AlexNet,ResNet,...) using multiple GPU's with synchronous updates.
Usage:
Please refer to the readme file to compile the program and train the model.

一个使用多个GPU同步更新来训练不同架构（AlexNet，ResNet，......）的程序。
用法：
请参阅自述文件以编译程序并训练模型。

"""

#导入需要的库包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange

import tensorflow as tf

import data_loader
import arch
import sys
import argparse
# import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"#不会占用别的GPU的资源


def exclude():
  """
    exclude variables when loading a snapshot, this is useful for transfer learning
    加载快照时排除变量，这对迁移学习很有用
  """
  var_list = tf.global_variables() #全局变量
  to_remove = []
  for var in var_list:
    if var.name.find("output")>=0 or var.name.find("epoch_number")>=0: 
      to_remove.append(var) #把变量添加到数组中
  print(to_remove)
  for x in to_remove:
    var_list.remove(x) #删除变量
  return var_list #返回一个删除后的变量


#计算平均交叉熵
def loss(logits, labels):
  """
  Compute cross-entropy loss for the given logits and labels
  Add summary for the cross entropy loss

  Args:
    logits: Logits from the model
    labels: Labels from data_loader
  Returns:
    Loss tensor of type float.

  计算给定logits和标签的交叉熵损失
  添加交叉熵损失的summary

  参数:
      logits:源自model
      labels:源自data_loader

  返回值:
        浮点型的损失张量。

  """
  # Calculate the average cross entropy loss across the batch.
  # 计算批次中的平均交叉熵损失。
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= logits, name= 'cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name= 'cross_entropy')

  #Add a Tensorboard summary
  #添加一个Tensorboard summary
  tf.summary.scalar('Cross Entropy Loss', cross_entropy_mean)

  return cross_entropy_mean #返回平均交叉熵


#计算平均梯度
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all the GPUs.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.

  计算所有GPU中每个共享变量的平均梯度。

  参数：
      tower_grads：（gradient, variable）元组列表的列表。

  返回值：
    （gradient, variable）对的列表，其中梯度已在所有towers中平均。

  """
  average_grads = [] #平均梯度
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis= 0, values= grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



#训练
def train(args):

  """Train different architectures for a number of epochs.
    用一定数量的epochs训练不同的网络模型结构
  """

  #构建graph
  with tf.Graph().as_default(), tf.device('/cpu:0'): #设置默认的gpu设备

    #从磁盘中加载数据
    images, labels = data_loader.read_inputs(True, args)
    
    #epoch数量
    epoch_number = tf.get_variable('epoch_number', [], dtype= tf.int32, initializer= tf.constant_initializer(0), trainable= False)

    #降低学习率
    lr = tf.train.piecewise_constant(epoch_number, args.LR_steps, args.LR_values, name= 'LearningRate')

    #权重降低策略
    wd = tf.train.piecewise_constant(epoch_number, args.WD_steps, args.WD_values, name= 'WeightDecay')

    # transfer mode,1意味着使用模型作为特征抽取器，所以我们不需要批量正则化更新和dropout
    is_training= not args.transfer_mode[0]==1

    #创建一个优化器用于梯度下降，0.9是梯度下降中的momentum,即冲量（梯度下降算法中概念）
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    # opt=tf.train.AdamOptimizer(lr,0.9) #这里使用Adam算法

    #计算每个模型tower的梯度(多GPU运行中概念)
    tower_grads = []#本质是一个二元组列表list（梯度，变量）

    # transfer mode 3 所必须
    tower_auxgrads = []
    #多GPU时
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(args.num_gpus): #遍历GPU
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:

            #计算一个tower损失，该函数构造完整的模型，并且所有tower共享变量
            logits = arch.get_model(images, wd, is_training, args)

            #Top-1准确性(tf.reduce_mean():求一维度上的平均,tf.cast)
            top1acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))#类型转换

            # Top-n accuracy
            # topnacc = tf.reduce_mean(
            #     tf.cast(tf.nn.in_top_k(logits, labels, args.top_n), tf.float32))

            #构建图表中计算损失的部分.请注意，我们将使用下面的自定义函数合成total_loss
            #平均交叉损失熵
            cross_entropy_mean = loss(logits, labels)

            #获取所有正则化的损失，并将其添加
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)

            reg_loss = tf.add_n(regularization_losses)

            #添加一个tensorboard sunmmary
            tf.summary.scalar('Regularization Loss', reg_loss)#用tensorboard查看

            #计算所有的损失(cross entropy loss + regularization loss)
            total_loss = tf.add(cross_entropy_mean, reg_loss)

            # 为total loss,top-1与top-5,添加一个scalar summary
            tf.summary.scalar('Total Loss', total_loss)
            tf.summary.scalar('Top-1 Accuracy', top1acc)
            # tf.summary.scalar('Top-'+str(args.top_n)+' Accuracy', topnacc)

            # 为下一个tower重复使用变量
            tf.get_variable_scope().reuse_variables()

            # 从最后一个tower中保留(retain)summaries
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            #聚集(gather)批量正则化更新操作
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

            #计算此tower上的batch数据的梯度。
            if args.transfer_mode[0]== 3:
              #仅仅为最后一层计算梯度
              grads = opt.compute_gradients(total_loss, var_list= tf.get_collection(tf.GraphKeys.VARIABLES, scope='output'))

              #为所有层计算梯度
              auxgrads = opt.compute_gradients(total_loss)
              tower_auxgrads.append(auxgrads)
            elif args.transfer_mode[0]==1:
              grads = opt.compute_gradients(total_loss,var_list= tf.get_collection(tf.GraphKeys.VARIABLES, scope='output'))
            else:
              grads = opt.compute_gradients(total_loss)

            #保持对所有tower的追踪
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    #我们必须计算每一个梯度的平均值。注意到：这是所有towers的同步点
    grads = average_gradients(tower_grads)

    # average all gradients for transfer mode 3
    #为transfer mode 3 ,求取所有梯度的平均值
    if args.transfer_mode[0]== 3:
      auxgrads = average_gradients(tower_auxgrads)

    # Add a summary to track the learning rate and weight decay
    #添加一个summary去跟踪学习率和权重衰减
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('weight_decay', wd))

    #设置训练的操作
    if args.transfer_mode[0]==3:
      train_op = tf.cond(tf.less(epoch_number,args.transfer_mode[1]),
              lambda: tf.group(opt.apply_gradients(grads),*batchnorm_updates), lambda: tf.group(opt.apply_gradients(auxgrads),*batchnorm_updates))
    elif args.transfer_mode[0]==1:
      train_op = opt.apply_gradients(grads)
    else:
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      train_op = tf.group(opt.apply_gradients(grads), batchnorm_updates_op)

    # a loader for loading the pretrained model (it does not load the last layer)
    #用来加载预训练模型的的加载器（该加载器没有加载最后一层）
    if args.retrain_from is not None:
      if args.transfer_mode[0]==0:
        pretrained_loader = tf.train.Saver()
      else:
        pretrained_loader = tf.train.Saver(var_list= exclude())

    # Create a saver.
    #创建一个saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep= args.max_to_keep)

    # Build the summary operation from the last tower summaries.
    #从最后的tower summaries中构建summary操作
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    #定义初始化操作
    init = tf.global_variables_initializer()

    # Logging the runtime information if requested
    #如果需要，记录运行时的信息。
    if args.log_debug_info:
      run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    # Creating a session to run the built graph
    #创建一个session以运行构建的图
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement= True,
        log_device_placement= args.log_device_placement))

    # config=tf.ConfigProto(allow_soft_placement=True)
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)#最多占用GPU的70%
    #
    # config.gpu_options.allow_growth=True#不会分配所有GPU资源，设置GPU资源按需增加
    #
    # sess=tf.Session(config=config)

    sess.run(init)#执行初始化操作

    # Continue training from a saved snapshot, load a pre-trained model
    #从一个已经保存的快照(snapshot)中,加载预训练的模型继续训练
    if args.retrain_from is not None:
      ckpt = tf.train.get_checkpoint_state(args.retrain_from)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        #从checkpoint中恢复
        pretrained_loader.restore(sess, ckpt.model_checkpoint_path)
      else:
        return

    # Start the queue runners.
    #开启队列runners
    tf.train.start_queue_runners(sess= sess)
    
    # Setup a summary writer
    #设置一个summary writer
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    # Set the start epoch number
    #设置开始的epoch 数量
    start_epoch = sess.run(epoch_number + 1)

    # The main training loop
    #主要的训练循环
    for epoch in xrange(start_epoch, start_epoch + args.num_epochs):
      # update epoch_number
      #更新epoch_number
      sess.run(epoch_number.assign(epoch))



      # Trainig batches
      #训练batches
      for step in xrange(args.num_batches):
    
        start_time = time.time()
        _, loss_value, top1_accuracy = sess.run([train_op, cross_entropy_mean, top1acc], options= run_options, run_metadata= run_metadata)
        duration = time.time() - start_time

        # Check for errors
        #检查错误
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Logging and writing tensorboard summaries
        #记录日志,写入到tensorboard sumamaries
        if step % 10 == 0:
          num_examples_per_step = args.chunked_batch_size * args.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / args.num_gpus

          format_str = ('%s: epoch %d, step %d, loss = %.2f, Top-1 = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), epoch, step, loss_value, top1_accuracy,
                               examples_per_sec, sec_per_batch))
          sys.stdout.flush()
        if step % 2000 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(
              summary_str, args.num_batches * epoch + step)
          if args.log_debug_info:
            summary_writer.add_run_metadata(
                run_metadata, 'epoch%d step%d' % (epoch, step))

      # Save the model checkpoint periodically after each training epoch
      #在每个训练时期之后定期保存模型检查点
      checkpoint_path = os.path.join(args.log_dir, args.snapshot_prefix)
      saver.save(sess, checkpoint_path, global_step= epoch)


def main():  # pylint: disable=unused-argument
    #命令行解析
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')

    parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk') #从磁盘加载图像的大小
    parser.add_argument('--crop_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images after random cropping') #随机裁剪后图像的大小
    parser.add_argument('--batch_size', default=24, type= int, action= 'store', help= 'The training batch size') # #训练的batch size的大小
    parser.add_argument('--num_classes', default=21, type=int, action='store', help= 'The number of classes') #分类的数量
    parser.add_argument('--num_channels', default= 3 , type= int, action= 'store', help= 'The number of channels in input images')#图片的颜色通道数
    parser.add_argument('--num_epochs', default= 20, type= int, action= 'store', help= 'The number of epochs') #训练的批次数
    parser.add_argument('--path_prefix', default= '', action='store', help= 'the prefix address for images') #图像的前缀地址
    parser.add_argument('--data_info', default= 'train_1.txt', action= 'store', help= 'Name of the file containing addresses and labels of training images')#从这里加载图片路径
    parser.add_argument('--shuffle', default= True, type= bool, action= 'store',help= 'Shuffle training data or not')  #是否随机Shuffle数据
    parser.add_argument('--num_threads', default= 1, type= int, action='store', help= 'The number of threads for loading data')#线程数，可以开启多线程
    parser.add_argument('--log_dir', default= None, action= 'store', help= 'Path for saving Tensorboard info and checkpoints') #日志保存
    parser.add_argument('--snapshot_prefix', default= 'snapshot', action= 'store', help= 'Prefix for checkpoint files')
    parser.add_argument('--architecture', default= 'vgg', help= 'The DNN architecture') #在这里选取不同网络结构
    # parser.add_argument('--architecture', default='googlenet', help='The DNN architecture')  # 在这里选取不同网络结构
    parser.add_argument('--depth', default= 50, type= int, action= 'store', help= 'The depth of ResNet architecture')#网络结构的层数
    parser.add_argument('--run_name', default= 'Run'+str(time.strftime("-%d-%m-%Y_%H-%M-%S")), action= 'store', help= 'Name of the experiment') #训练实验的名字
    parser.add_argument('--num_gpus', default= 1, type= int, action= 'store', help= 'Number of GPUs') #gpu个数
    parser.add_argument('--log_device_placement', default= False, type= bool, help= 'Whether to log device placement or not')
    parser.add_argument('--delimiter', default= ' ', action= 'store', help= 'Delimiter of the input files')#分隔符
    parser.add_argument('--retrain_from', default= None, action= 'store', help= 'Continue Training from a snapshot file')
    parser.add_argument('--log_debug_info', default= False, action= 'store', help= 'Logging runtime and memory usage info')
    parser.add_argument('--num_batches', default= -1, type= int, action= 'store', help= 'The number of batches per epoch') #每个批次的batch个数
    parser.add_argument('--transfer_mode', default = [0], nargs='+', type= int, help= 'Transfer mode 0=None , 1=Tune last layer only , 2= Tune all the layers, 3= Tune the last layer at early epochs     (it could be specified with the second number of this argument) and then tune all the layers') #这里选择迁移学习
    # parser.add_argument('--LR_steps', type=int, nargs='+', default=[19, 30, 44, 53], help='LR change epochs')
    # parser.add_argument('--LR_steps', type=int, nargs='+', default=[3, 5, 7, 9], help='LR change epochs')
    parser.add_argument('--LR_steps', type=int, nargs='+', default=[6, 9, 12, 15], help='LR change epochs')
    # parser.add_argument('--LR_values', type=float, nargs='+', default=[0.01, 0.005, 0.001, 0.0005, 0.0001], help='LR change epochs')
    parser.add_argument('--LR_values', type=float, nargs='+', default=[0.012, 0.005, 0.001, 0.0005, 0.0001], help='LR change epochs')#学习率值
    # parser.add_argument('--WD_steps', type=int, nargs='+', default= [30], help='WD change epochs')
    parser.add_argument('--WD_steps', type=int, nargs='+', default=[12], help='WD change epochs')

    parser.add_argument('--WD_values', type=float, nargs='+', default=[0.0005, 0.0001], help='LR change epochs')
    parser.add_argument('--top_n', default= 5, type= int, action= 'store', help= 'Specify the top-N accuracy')#top-N精确度
    parser.add_argument('--max_to_keep', default= 3, type= int, action= 'store', help= 'Maximum number of snapshot files to keep')
    args = parser.parse_args()

    #为每个GPU分配训练样本
    args.chunked_batch_size = int(args.batch_size/args.num_gpus)

    #统计训练样本的总数量
    args.num_samples = sum(1 for line in open(args.data_info))

    #设置每一个epoch的batch数量
    if args.num_batches==-1:
        args.num_batches= int(args.num_samples/args.batch_size)+1

    #（为空）创建日志文件目录
    if args.log_dir is None:
      args.log_dir= args.architecture+"_"+args.run_name

    print(args)
    print("Saving everything in "+args.log_dir)

    #如果需要，重新创建日志文件目录
    if tf.gfile.Exists(args.log_dir):
      tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)
 
    train(args)


if __name__ == '__main__':
  main()
