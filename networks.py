#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tensorpack import *

from scipy import signal
from scipy.ndimage.filters import convolve
from tensorflow.python.framework import ops

from tensorpack import summary

import tensorflow.contrib.slim as slim





    ########################### VGG from mono ###############################################################

def VGG(image,length=2,reuse=False):
    with tf.variable_scope("VGG_autoEncoder", reuse=reuse):

        x1 = Conv2D('conv1', image, 32, kernel_shape=7, stride=2, nl=tf.nn.elu)  # H
        # x_1 = MaxPooling('maxpool1', x1, 2)  # H/2
        x2 = Conv2D('conv2', x1, 32, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_2 = MaxPooling('maxpool2', x2, 2)  # H/4
        x3 = Conv2D('conv3', x2, 64, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_3 = MaxPooling('maxpool3', x3, 2)  # H/8
        x4 = Conv2D('conv4', x3, 128, kernel_shape=6, stride=2, nl=tf.nn.elu)
        # x_4 = MaxPooling('maxpool4', x4, 2)  # H/16
        x5 = Conv2D('conv5', x4, 256, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_5 = MaxPooling('maxpool5', x5, 2)  # H/32
        x6 = Conv2D('conv6', x5, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_6 = MaxPooling('maxpool6', x6, 2)  # H/64
        x7 = Conv2D('conv7', x6, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_7 = MaxPooling('maxpool7', x7, 2)  # H/128
        x8 = Conv2D('conv8', x7, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)

        # new part of the network
        # x_8 = MaxPooling('maxpool8', x8, 2)  # H/256
        x9 = Conv2D('conv9', x8, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)
        # x_9 = MaxPooling('maxpool10', x9, 2)  # H/512

        # deconv new
        y9 = Deconv2D('deconv9', x9, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)  # H/512
        # y_9 = FixedUnPooling('unpooling9', y9, 2)  # H/256
        # concat9 = tf.concat([y9, x8], 3)

        # something like a decoder
        y8 = Deconv2D('deconv8', y9, 512, kernel_shape=7, stride=2, nl=tf.nn.elu)  # H/128

        # y_8 = FixedUnPooling('unpooling8', y8, 2)  # H/64

        concat8 = tf.concat([y8, x6], 3)

        # iconv8 = Conv2D(concat8, 512, kernel_shape=3, stride=1,nl=tf.nn.relu)
        # print 'iconv8 done 666666666666666666666666666666666666666666666666666666666666'

        # y7 = Deconv2D('deconv7', iconv8, 512, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/64
        y7 = Deconv2D('deconv7', concat8, 512, kernel_shape=4, stride=2, nl=tf.nn.elu)  # H/64
        print 'Deconv7 done 666666666666666666666666666666666666666666666666666666666666'
        # y_7 = FixedUnPooling('unpooling7', y7, 2)  # H/32
        concat7 = tf.concat([y7, x5], 3)
        print 'concat7 done 666666666666666666666666666666666666666666666666666666666666'
        iconv7 = Conv2D('conv_de7', concat7, 512, kernel_shape=3, stride=1, nl=tf.nn.relu)

        y6 = Deconv2D('deconv6', iconv7, 256, kernel_shape=5, stride=2, nl=tf.nn.relu)  # H/32
        # y6 = Deconv2D('deconv6', concat7, 256, kernel_shape=5, stride=2, nl=tf.nn.elu)  # H/32
        # y_6 = FixedUnPooling('unpooling6', y6, 2)  # H/16
        print 'unp6 done 666666666666666666666666666666666666666666666666666666666666'
        concat6 = tf.concat([y6, x4], 3)
        iconv6 = Conv2D('conv_de6', concat6, 256, kernel_shape=3, stride=1, nl=tf.nn.relu)

        # y5 = Deconv2D('deconv5', iconv6, 128, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/16
        y5 = Deconv2D('deconv5', iconv6, 128, kernel_shape=6, stride=2, nl=tf.nn.elu)  # H/16
        # y_5 = FixedUnPooling('unpooling5', y5, 2)  # H/8
        concat5 = tf.concat([y5, x3], 3)  #
        print 'unp5 done 666666666666666666666666666666666666666666666666666666666666'
        iconv5 = Conv2D('conv_de5', concat5, 128, kernel_shape=3, stride=1, nl=tf.nn.relu)
        disp5 = get_disp_pack(iconv5, name='disparity_5')

        # y4 = Deconv2D('deconv4', iconv5, 64, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/8
        y4 = Deconv2D('deconv4', disp5, 64, kernel_shape=6, stride=2, nl=tf.nn.elu)  # H/8
        # y_4 = FixedUnPooling('unpooling4', y4, 2)  # H/4
        concat4 = tf.concat([y4, x2], 3)  #
        iconv4 = Conv2D('conv_de4', concat4, 64, kernel_shape=3, stride=1, nl=tf.nn.relu)
        disp4 = get_disp_pack(iconv4, name='disparity_4')

        # y3 = Deconv2D('deconv3', iconv4, 32, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/4
        y3 = Deconv2D('deconv3', disp4, 32, kernel_shape=6, stride=2, nl=tf.nn.elu)  # H/4
        # y_3 = FixedUnPooling('unpooling3', y3, 2)  # H/2
        concat3 = tf.concat([y3, x1], 3)  #
        iconv3 = Conv2D('conv_de3', concat3, 32, kernel_shape=3, stride=1, nl=tf.nn.relu)
        disp3 = get_disp_pack(iconv3, name='disparity_3')

        y2 = Deconv2D('deconv2', disp3, 32, kernel_shape=6, stride=1, nl=tf.nn.elu)  # H/2
        # y_2 = FixedUnPooling('unpooling2', y2, 2)  # H
        #concat2 = tf.concat([y2, x1], 3)  #
        iconv2 = Conv2D('conv_de2', y2, 32, kernel_shape=3, stride=1, nl=tf.nn.relu)
        disp2 = get_disp_pack(iconv2, name='disparity_2')

        print 'unp2  done 666666666666666666666666666666666666666666666666666666666666'

        # x = Deconv2D('deconv1',y_2, 6, kernel_shape=3, stride=1, nl=tf.nn.relu) # for producing two images
        y1 = Deconv2D('deconv1', disp2, 16, kernel_shape=6, stride=1,
                      nl=tf.nn.relu)  # for producing two xmaps for dispa
        y_1 = FixedUnPooling('unpooling1', y1, 2)
        x = Deconv2D('deconv0', y_1, length, kernel_shape=8, stride=1, nl=tf.nn.elu)

        # x = Conv2D('conv6', x, 6, kernel_shape=3, nl=tf.identity)

    return x

########################## End of VGG #########################################################



####################### 3D voxelnet ########################

# https://arxiv.org/pdf/1702.02463.pdf

def Voxel3D(image,length=3,reuse=False):
    with tf.variable_scope("Voxel_auto_encoder", reuse=reuse):

        x1 = Conv2D('conv1', image, 64, kernel_shape=5, stride=1, nl=tf.nn.relu)  # H

        x_1 = MaxPooling('maxpool1', x1, 2)  # H/2
        x2 = Conv2D('conv2', x_1, 128, kernel_shape=5, stride=1, nl=tf.nn.relu)  # H/2

        x_2 = MaxPooling('maxpool2', x2, 2)  # H/4
        x3 = Conv2D('conv3', x_2, 256, kernel_shape=3, stride=1, nl=tf.nn.relu)  # H/4

        x_3 = MaxPooling('maxpool3', x3, 2)  # H/8
        x4 = Conv2D('conv4', x_3, 256, kernel_shape=3, stride=1, nl=tf.nn.relu)  # H/8



        y3 = Deconv2D('deconv3', x4, 256, kernel_shape=3, stride=2, nl=tf.nn.relu)  # H/4
        y_3 = Conv2D('conv5', y3, 256, kernel_shape=3, stride=1, nl=tf.nn.relu)  # H/4

        concat1 = tf.concat([x3, y_3], 3)

        y2 = Deconv2D('deconv2', concat1, 128, kernel_shape=3, stride=2, nl=tf.nn.relu)  # H/2
        y_2 = Conv2D('conv6', y2, 128, kernel_shape=3, stride=1, nl=tf.nn.relu)  # H/2

        concat2 = tf.concat([x2, y_2], 3)

        y1 = Deconv2D('deconv1', concat2, 64, kernel_shape=4, stride=2, nl=tf.nn.relu)  # H

        concat3 = tf.concat([x1, y1], 3)
        y_1 = Conv2D('conv7', concat3, 64, kernel_shape=4, stride=1, nl=tf.nn.relu)  # H



        op1 = Conv2D('conv8a', y_1, 2, kernel_shape=5, stride=1, nl=tf.nn.tanh)  # H output1
        op2 = Conv2D('conv8b', y_1, 2, kernel_shape=5, stride=2, nl=tf.nn.tanh)  # H/2 output2
        op3 = Conv2D('conv8c', y_1, 2, kernel_shape=5, stride=4, nl=tf.nn.tanh)  # H/4 output3


        z1 = Conv2D('conv9a', op1, 32, kernel_shape=4, stride=1, nl=tf.nn.tanh)  # H output1 final 32

        z2a = BilinearUpSample('upsample1',op2, 2)
        z2 = Conv2D('conv9b', z2a, 32, kernel_shape=5, stride=1, nl=tf.nn.tanh)  # H output1 final 32

        concat1 = tf.concat([z1, z2], 3)

        z3a = BilinearUpSample('upsample1', op3, 4)
        z3 = Conv2D('conv9c', z3a, 32, kernel_shape=5, stride=1, nl=tf.nn.tanh)  # H output1 final 32

        concat2 = tf.concat([concat1, z3], 3)

        y0 = Conv2D('conv10', concat2, 64, kernel_shape=5, stride=1, nl=tf.nn.relu)  # H output1 final 32
        final = Conv2D('conv11', y0, length, kernel_shape=5, stride=1, nl=tf.identity)  # H output1 final 32

    return final






##################### end voxelnet ##########################










########################### normal VGG autoencoder ##############################################

def vgg_ende(y,length=2,reuse=False):
    # something like an encoder
    with tf.variable_scope("Dumb_VGG", reuse=reuse):
        x1 = Conv2D('conv1', y, 32, kernel_shape=7, stride=1, nl=tf.nn.relu)  # H
        x_1 = MaxPooling('maxpool1', x1, 2)  # H/2
        x2 = Conv2D('conv2', x_1, 32, kernel_shape=4, stride=1, nl=tf.nn.relu)
        x_2 = MaxPooling('maxpool2', x2, 2)  # H/4
        x3 = Conv2D('conv3', x_2, 64, kernel_shape=4, stride=1, nl=tf.nn.relu)
        x_3 = MaxPooling('maxpool3', x3, 2)  # H/8
        x4 = Conv2D('conv4', x_3, 128, kernel_shape=6, stride=1, nl=tf.nn.relu)
        x_4 = MaxPooling('maxpool4', x4, 2)  # H/16
        x5 = Conv2D('conv5', x_4, 256, kernel_shape=4, stride=1, nl=tf.nn.relu)
        x_5 = MaxPooling('maxpool5', x5, 2)  # H/32
        x6 = Conv2D('conv6', x_5, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)
        x_6 = MaxPooling('maxpool6', x6, 2)  # H/64
        x7 = Conv2D('conv7', x_6, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)
        x_7 = MaxPooling('maxpool7', x7, 2)  # H/128
        x8 = Conv2D('conv8', x_7, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)

        # new part of the network
        x_8 = MaxPooling('maxpool8', x8, 2)  # H/256
        x9 = Conv2D('conv9', x_8, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)
        #x_9 = MaxPooling('maxpool10', x9, 2)  # H/512

        # deconv new
        y9 = Deconv2D('deconv9', x9, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)  # H/512
        y_9 = FixedUnPooling('unpooling9', y9, 2)  # H/256
        concat9 = tf.concat([y_9, x8], 3)

        # something like a decoder
        y8 = Deconv2D('deconv8', concat9, 512, kernel_shape=7, stride=1, nl=tf.nn.relu)  # H/128

        y_8 = FixedUnPooling('unpooling8', y8, 2)  # H/64

        concat8 = tf.concat([y_8, x7], 3)

        # iconv8 = Conv2D(concat8, 512, kernel_shape=3, stride=1,nl=tf.nn.relu)
        # print 'iconv8 done 666666666666666666666666666666666666666666666666666666666666'

        # y7 = Deconv2D('deconv7', iconv8, 512, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/64
        y7 = Deconv2D('deconv7', concat8, 512, kernel_shape=4, stride=1, nl=tf.nn.relu)  # H/64
        print 'Deconv7 done 666666666666666666666666666666666666666666666666666666666666'
        y_7 = FixedUnPooling('unpooling7', y7, 2)  # H/32
        concat7 = tf.concat([y_7, x6], 3)
        print 'concat7 done 666666666666666666666666666666666666666666666666666666666666'
        # iconv7 = Conv2D(concat7, 512, kernel_shape=3, stride=1, nl=tf.nn.relu)

        # y6 = Deconv2D('deconv6', iconv7, 256, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/32
        y6 = Deconv2D('deconv6', concat7, 256, kernel_shape=5, stride=1, nl=tf.nn.relu)  # H/32
        y_6 = FixedUnPooling('unpooling6', y6, 2)  # H/16
        print 'unp6 done 666666666666666666666666666666666666666666666666666666666666'
        concat6 = tf.concat([y_6, x5], 3)
        # iconv6 = Conv2D(concat6, 256, kernel_shape=3, stride=1, nl=tf.nn.relu)

        # y5 = Deconv2D('deconv5', iconv6, 128, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/16
        y5 = Deconv2D('deconv5', concat6, 128, kernel_shape=6, stride=1, nl=tf.nn.relu)  # H/16
        y_5 = FixedUnPooling('unpooling5', y5, 2)  # H/8
        concat5 = tf.concat([y_5, x4], 3)
        print 'unp5 done 666666666666666666666666666666666666666666666666666666666666'
        # iconv5 = Conv2D(concat5, 128, kernel_shape=3, stride=1, nl=tf.nn.relu)

        # y4 = Deconv2D('deconv4', iconv5, 64, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/8
        y4 = Deconv2D('deconv4', concat5, 64, kernel_shape=6, stride=1, nl=tf.nn.relu)  # H/8
        y_4 = FixedUnPooling('unpooling4', y4, 2)  # H/4
        concat4 = tf.concat([y_4, x3], 3)
        # iconv4 = Conv2D(concat4, 64, kernel_shape=3, stride=1, nl=tf.nn.relu)

        # y3 = Deconv2D('deconv3', iconv4, 32, kernel_shape=3,stride=1,nl=tf.nn.relu)  # H/4
        y3 = Deconv2D('deconv3', concat4, 32, kernel_shape=6, stride=1, nl=tf.nn.relu)  # H/4
        y_3 = FixedUnPooling('unpooling3', y3, 2)  # H/2
        concat3 = tf.concat([y_3, x2], 3)
        # iconv3 = Conv2D(concat3, 32, kernel_shape=3, stride=1, nl=tf.nn.relu)

        y2 = Deconv2D('deconv2', concat3, 32, kernel_shape=6, stride=1, nl=tf.nn.relu)  # H/2
        y_2 = FixedUnPooling('unpooling2', y2, 2)  # H
        concat2 = tf.concat([y_2, x1], 3)

        print 'unp2  done 666666666666666666666666666666666666666666666666666666666666'

        # x = Deconv2D('deconv1',y_2, 6, kernel_shape=3, stride=1, nl=tf.nn.relu) # for producing two images
        y1 = Deconv2D('deconv1', concat2, 16, kernel_shape=6, stride=1,
                      nl=tf.nn.relu)  # for producing two xmaps for dispa
        #y_1 = FixedUnPooling('unpooling1', y1, 2)
        x = Deconv2D('deconv0', y1, length, kernel_shape=8, stride=1, nl=tf.nn.relu)

        # x = Conv2D('conv6', x, 6, kernel_shape=3, nl=tf.identity)

    return x

########################### end of normal VGG autoencoder ##############################################
############### some_func ###########
def some_func(y,length=2, reuse=False):
    with tf.variable_scope("Dumb_Encoders", reuse=reuse):
        x = Conv2D('conv1', y, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv2', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv3', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv4', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv5', x, 32, kernel_shape=3, nl=tf.nn.relu)
        # x = Conv2D('conv6', x, 3, kernel_shape=3, nl=tf.identity) + y
        x = Conv2D('conv6', x, length, kernel_shape=3, nl=tf.identity)
    return x
#####################################


 ########################### dumb_network from mono ###############################################################
def dumb_network1(y,length=2, reuse=False):
    with tf.variable_scope("Dumb_Encoder1", reuse=reuse):
        x = Conv2D('conv1', y, 128, kernel_shape=4, nl=tf.nn.relu)
        x = Conv2D('conv2', x, 128, kernel_shape=4, nl=tf.nn.relu)
        x = Conv2D('conv3', x, 64, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv4', x, 64, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv6', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv7', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv8', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv9', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv10', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv11', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv12', x, length, kernel_shape=3, nl=tf.identity)

    return x

def dumb_network2(y,length=2, reuse=False):
    with tf.variable_scope("Dumb_Encoder2", reuse=reuse):
        x = Conv2D('conv1', y, 32, kernel_shape=4, nl=tf.nn.relu)
        x = Conv2D('conv2', x, 32, kernel_shape=4, nl=tf.nn.relu)
        x = Conv2D('conv3', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv4', x, 32, kernel_shape=3, nl=tf.nn.relu)
        #x = Conv2D('conv6', x, 32, kernel_shape=3, nl=tf.nn.relu)
        #x = Conv2D('conv7', x, 32, kernel_shape=3, nl=tf.nn.relu)
        #x = Conv2D('conv8', x, 32, kernel_shape=3, nl=tf.nn.relu)
        #x = Conv2D('conv9', x, 32, kernel_shape=3, nl=tf.nn.relu)
        #x = Conv2D('conv10', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv11', x, 32, kernel_shape=3, nl=tf.nn.relu)
        x = Conv2D('conv12', x, length, kernel_shape=3, nl=tf.identity)

    return x



########################## UNET #########################################################




def U_net(image,length=2,reuse=False):
    NF = 64
    with argscope(Conv2D, kernel_shape=4, stride=2):
        # encoder
        e1 = Conv2D('conv1', image, NF, nl=tf.nn.relu)
        e2 = Conv2D('conv2', e1, NF * 2)
        e3 = Conv2D('conv3', e2, NF * 4)
        e4 = Conv2D('conv4', e3, NF * 8)
        e5 = Conv2D('conv5', e4, NF * 8)
        e6 = Conv2D('conv6', e5, NF * 8)
        e7 = Conv2D('conv7', e6, NF * 8)
        e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
    with argscope(Deconv2D, nl=BNReLU, kernel_shape=7, stride=2):
        # decoder
        e8 = Deconv2D('deconv1', e8, NF * 8)
        e8 = Dropout(e8)
        e8 = tf.concat([e8, e7], 3)
        # e8 = ConcatWith(e8, 3, e7)

        e7 = Deconv2D('deconv2', e8, NF * 8)
        e7 = Dropout(e7)
        e7 = tf.concat([e7, e6], 3)
        # e7 = ConcatWith(e7, 3, e6)

        e6 = Deconv2D('deconv3', e7, NF * 8)
        e6 = Dropout(e6)
        e6 = tf.concat([e6, e5], 3)
        # e6 = ConcatWith(e6, 3, e5)

        e5 = Deconv2D('deconv4', e6, NF * 8)
        e5 = Dropout(e5)
        e5 = tf.concat([e5, e4], 3)
        # e5 = ConcatWith(e5, 3, e4)

        e4 = Deconv2D('deconv5', e5, NF * 4)
        e4 = Dropout(e4)
        # e4 = ConcatWith(e4, 3, e3)
        e4 = tf.concat([e4, e3], 3)

        e3 = Deconv2D('deconv6', e4, NF * 2)
        e3 = Dropout(e3)
        # e3 = ConcatWith(e3, 3, e2)
        e3 = tf.concat([e3, e2], 3)

        e2 = Deconv2D('deconv7', e3, NF * 1)
        e2 = Dropout(e2)
        # e2 = ConcatWith(e2, 3, e1)
        e2 = tf.concat([e2, e1], 3)

        prediction = Deconv2D('prediction', e2, length, nl=tf.nn.relu)

    return prediction

    #################################### end of UNET #################################################


############## a very basic 3D convolution network ###############################



################################ end #############################################






#################### extra functions ########################

def get_disp_pack(x, name):
    disp = 0.3 * conv_pack(name, x, 2, 3, 1, tf.nn.sigmoid)
    return disp


def conv_pack(name, x, num_out_layers, kernel_size, stri, activation_fn=tf.nn.elu):
    print 'here 1'
    # p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    print 'here 2'
    # p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    print 'here 3 '
    # Conv2D('conv1', y, 32, kernel_shape=7, stride=1, nl=tf.nn.elu)
    # return Conv2D(name,p_x, num_out_layers, kernel_shape=kernel_size, stride=stri, nl=activation_fn)
    return Conv2D(name, x, num_out_layers, kernel_shape=kernel_size, stride=stri, nl=activation_fn)