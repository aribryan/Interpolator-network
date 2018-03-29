#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tensorpack import *
from networks import VGG,dumb_network1,U_net,vgg_ende,dumb_network2,some_func,Voxel3D
from SSIM_loss import SSIM

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

BATCH_SIZE = 16
SHAPE = 28
HEIGHT = 256
WIDTH = 256
CHANNELS = 3


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.ppm', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


class Model(ModelDesc):
    def _get_inputs(self):
        '''''''''
        return [InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'left'),
                InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'mid'),
                InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'right')]
                '''
        return [InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'left'),
                InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'mid'),
                InputDesc(tf.float32, (None, HEIGHT, WIDTH, CHANNELS), 'right')]

    def _build_graph(self, inputs):
        # left is image [HEIGHT, WIDTH, 3] with range [0, 255]
        left, middle,right = inputs
        # left is image [HEIGHT, WIDTH, 3] with range [-1, 1]
        left = left / 128 - 1
        middle = middle / 128 - 1
        right = right / 128 - 1

        # START HERE
        # put the network

        def some_func2(y):


            x = Conv2D('conv1', y, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv2', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv3', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv4', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv5', x, 32, kernel_shape=3, nl=tf.nn.relu)
            #x = Conv2D('conv6', x, 3, kernel_shape=3, nl=tf.identity) + y
            x = Conv2D('conv6', x, 3, kernel_shape=3, nl=tf.identity)
            return x

        def resnet50(y):

            x=Conv2D('conv1',y,64,kernel_shape=7,stride=1, nl=tf.nn.relu)
            x=MaxPooling('maxpool1',x,3)
            x=Conv2D('conv2',x,64,kernel_shape=3,stride=1,nl=tf.nn.relu)
            x=Conv2d('conv3',x,128,kernel_shape=4,stride=1,nl=tf.nn.relu)
            x = Conv2d('conv4', x, 256, kernel_shape=6,stride=1, nl=tf.nn.relu)
            x = Conv2d('conv5', x, 512, kernel_shape=3, stride=1,nl=tf.nn.relu)

            return x

        def vgg_en(y):

            x=Conv2D('conv1',y,32,kernel_shape=7,stride=1, nl=tf.nn.relu)
            x=MaxPooling('maxpool1',x,3)
            x=Conv2D('conv2',x,64,kernel_shape=5,stride=1,nl=tf.nn.relu)
            x=Conv2d('conv3',x,128,kernel_shape=3,stride=1,nl=tf.nn.relu)
            x = Conv2d('conv4', x, 256, kernel_shape=3,stride=1, nl=tf.nn.relu)
            x = Conv2d('conv5', x, 512, kernel_shape=3, stride=1,nl=tf.nn.relu)
            x = Conv2d('conv5', x, 512, kernel_shape=3, stride=1, nl=tf.nn.relu)
            x = Conv2d('conv5', x, 512, kernel_shape=3, stride=1, nl=tf.nn.relu)

            return x



        lB, lG, lR = tf.unstack(left, axis=-1)

        rB, rG, rR = tf.unstack(right, axis=-1)

        total_input = tf.stack([lB, lG, lR, rB, rG, rR],
                               axis=-1)  # 6 channel tensor for combined input of left and right


        new_mid = Voxel3D(total_input,3)
        print 'Prediction tensor is hereeeeeee 1111111111111111'
        print new_mid

        dummy = tf.identity(new_mid, name="important_tensor")
        print dummy.name
        corrected_new_mid = 128.*(new_mid + 1)

        '''''''''
        [la,lb,lc,ra,rb,rc] = tf.unstack(total, axis=-1)

        warped_left = tf.stack([la,lb,lc], axis=-1)
        warped_right = tf.stack([ra,rb,rc], axis=-1)
        '''

        error_mid = middle - new_mid
        corrected_error_mid = 128.*(error_mid +1)


        corrected_left = 128. * (left + 1)
        #corrected_warped_left = 128. * (warped_left + 1)
        #corrected_warped_right = 128. * (warped_right + 1)
        corrected_right = 128. * (right + 1)
        corrected_mid = 128. *(middle +1)

        # END HERE

        viz = tf.concat([corrected_left,corrected_right,corrected_mid, corrected_new_mid, corrected_error_mid], 2)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        xy=tf.summary.image('left,right,original_middle,predicted_middle,error_map', viz, max_outputs=max(30, BATCH_SIZE))
        print 'my predictor is',xy
        #writer = tf.summary.FileWriter("output", sess.graph)
        #self.cost1 = tf.reduce_mean(tf.squared_difference(middle, warped_left), name='total_cost1')
        #self.cost2 = tf.reduce_mean(tf.squared_difference(middle, warped_right), name='total_cost2')
        #self.cost = tf.add(self.cost1,self.cost2,name='total_costs')


        l1l2w= 1
        ssimw= 0.5
        self.costl2 = tf.reduce_mean(tf.squared_difference(new_mid,middle), name='total_costs1')


        self.costl1 = tf.reduce_mean(tf.abs(tf.subtract(new_mid, middle)), name='total_costs2')

        self.cost_ssim = ssimw * (tf.reduce_mean((SSIM(new_mid, middle)), name='ssim_costs'))


        self.cost1 = l1l2w * (self.costl1 + self.costl2)

        self.cost = tf.add(self.cost1,self.cost_ssim,name='total_costs')

        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-5, trainable=False)
        return tf.train.AdamOptimizer(lr)


def get_data():
    ds = LMDBDataPoint('./lmdb_files/fraunhoffer15000frames_new.lmdb', shuffle=True)
    #ds = LMDBDataPoint('./lmdb_files/fraunhoffer600_new.lmdb', shuffle=True)
    ds = ImageDecode(ds, index=0)
    ds = ImageDecode(ds, index=1)
    ds = ImageDecode(ds, index=2)

    def rescale(img):
        return cv2.resize(img, (WIDTH, HEIGHT))

    ds = MapDataComponent(ds, rescale, index=0)
    ds = MapDataComponent(ds, rescale, index=1)
    ds = MapDataComponent(ds, rescale, index=2)
    '''''''''
    augs = [
        # imgaug.MapImage(lambda x: x * 255.0),
        # imgaug.RandomResize((0.7, 1.2), (0.7, 1.2)),
        # imgaug.RotationAndCropValid(45)
        # imgaug.RandomPaste((WIDTH, HEIGHT)),
        # imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01)
        imgaug.CenterCrop((HEIGHT,WIDTH))]
    
    ds = AugmentImageComponent(ds, augs, index=0)
    ds = AugmentImageComponent(ds, augs, index=1)
    ds = AugmentImageComponent(ds, augs, index=2)
    '''

    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    # [(HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 3)]
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data()

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver()
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=4000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()

    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.load:
        config.session_init = SaverRestore(args.load)

    trainer = SyncMultiGPUTrainer(config.nr_tower)
    #trainer = train.SyncMultiGPUTrainerParameterServer(get_nr_gpu(),1)
    #launch_train_with_config(config, SimpleTrainer())
    launch_train_with_config(config, trainer)