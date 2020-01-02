# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:58:54 2018

@author: shirhe-lyh
"""

import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib.slim import nets

slim = tf.contrib.slim
import tensorflow as tf
def set_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    return gpuConfig
def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.  """
  with tf.name_scope(values=[image_buffer], name=scope,default_name='decode_jpeg'):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def read_and_decode(example_proto):
    features = tf.parse_single_example(
      example_proto,
      features={
          'image/height'       : tf.FixedLenFeature([], tf.int64),
          'image/width'        : tf.FixedLenFeature([], tf.int64),
          'image/colorspace'   : tf.FixedLenFeature([], tf.string),
          'image/channels'     : tf.FixedLenFeature([], tf.int64),
          'image/class/label'  : tf.FixedLenFeature([], tf.int64),
          'image/class/synset' : tf.FixedLenFeature([], tf.string),
          'image/class/text'   : tf.FixedLenFeature([], tf.string),
          'image/format'       : tf.FixedLenFeature([], tf.string),
          'image/filename'     : tf.FixedLenFeature([], tf.string),
          'image/encoded'      : tf.FixedLenFeature([], tf.string)
      })

    width = tf.cast(features['image/width'], dtype=tf.int32)
    height = tf.cast(features['image/height'], dtype=tf.int32)
    synset = tf.cast(features['image/class/synset'], dtype=tf.string)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    human = tf.cast(features['image/class/text'], dtype=tf.string)
    filename = tf.cast(features['image/filename'], dtype=tf.string)

    image_buffer=features['image/encoded']
    image = decode_jpeg(image_buffer)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [256,256,3])
    return image, label, synset,human,width, height,filename
    
batch_size=50
imageHeight=224
imageWidth=224

num_steps = int(50000/batch_size)
epoch=int(batch_size*num_steps/300000)+1
model_save_path =  "./pre-model/resnet_v1_50.ckpt"  # Path to the model.ckpt-(num_steps) will be saved
train_filename = "./ImageNettrain300tf/"
val_filename = "./imagenet50tf/"
train_files_names = os.listdir(train_filename)
train_files = [train_filename+ item for item in train_files_names]
print(train_files_names)
print(train_files)

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(read_and_decode).repeat(1)#10*batchsize
train_dataset = dataset.batch(batch_size)


train_iterator = train_dataset.make_initializable_iterator()
images, labels, synsets,humans,widths, heights,filenames= train_iterator.get_next()


val_files_names = os.listdir(val_filename)
val_files = [val_filename+ item for item in val_files_names]
print(val_files_names)
print(val_files)

tdataset = tf.data.TFRecordDataset(val_files)
tdataset = tdataset.map(read_and_decode).repeat(1)#10*batchsize
val_dataset = tdataset.batch(batch_size)


val_iterator = val_dataset.make_initializable_iterator()
vimages, vlabels, _,_,_, _,_= val_iterator.get_next()


if __name__ == '__main__':
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #resnet_model_path = '···/resnet_v1_50.ckpt'  # Path to the pretrained model
    X = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='TX')
    Y = tf.placeholder(tf.int64, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='IsTraining')
    #规范化数据
    #distorted_images = X - [123.68, 116.78, 103.94]
    #distorted_images = tf.image.per_image_standardization(X)
    #distorted_images=X
    #distorted_images = tf.image.resize_images(X, [224,224])
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(distorted_images, num_classes=1000,
                                                     is_training=is_training)
    
    logits= tf.squeeze(net, squeeze_dims=[1,2])
    logits1 = tf.nn.softmax(logits)
    classes = tf.argmax(logits1, axis=1, name='classes')
    epual0=tf.cast(tf.equal(tf.cast(classes, dtype=tf.int64), Y), dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int64), Y), dtype=tf.float32))

    init = tf.global_variables_initializer()
    restorer = tf.train.Saver(tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=10)
    checkpoints_dir="./ckpt/"
    cfg = set_config()
    with tf.Session(config=cfg) as sess:

        sess.run(init)
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        restorer.restore(sess,model_save_path)
        totalacc0=0
        avaacc0=0
        te=0
        totalacc=0
        avaacc=0
        
        try:
            
            for i in range(0,num_steps-1):
                """
                input_image, groundtruth_lists, syn, hs, _, _, fs = sess.run([images, labels,
                                                 synsets, humans, widths, heights, filenames])  # 在会话中取出image和label  syn, hs, _, _, fs
                train_dict = {X: input_image, Y: groundtruth_lists,is_training: True}
                #loss_, acc_ ,lr= sess.run([loss, accuracy,learning_rate], feed_dict=train_dict)
                acc_,classes0 = sess.run( [accuracy,classes], feed_dict=train_dict)
                #print(groundtruth_lists)
                #print(classes0)
                totalacc0+=acc_
                avaacc0=totalacc0 / (i+1)*1.0
                """
                vinput_image,vgroundtruth_lists = sess.run([vimages, vlabels])  # 在会话中取出image和label  syn, hs, _, _, fs
                val_dict = {X: vinput_image, Y: vgroundtruth_lists, is_training: False}
                vacc_ ,e0= sess.run([ accuracy,epual0], feed_dict=val_dict)
                #print(e0)
                te=te+np.sum(e0)
                totalacc+=vacc_
                avaacc=totalacc/(i+1)*1.0 #测试集准确率

                #train_text = 'Step: {},[Train]  Acc: {:.4f}, AveAcc: {:.4f} '.format(i, acc_,avaacc0)
                train_text = 'Step: {},[Val]  Acc: {:.4f}, AveAcc: {:.4f} ,te: {:.4f}'.format(i, vacc_,avaacc,te)
                print(train_text)
            #print(te)
      
        except tf.errors.OutOfRangeError:
            print("bug---------")
        


