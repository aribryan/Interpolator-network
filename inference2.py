#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Will People Like Your Image?
tested with TensorFlow 1.1.0-rc1 (git rev-parse HEAD 45115c0a985815feef3a97a13d6b082997b38e5d) and OpenCV 3.1.0
EXAMPLE:
    python score.py --images "pattern/to/images/*.jpg"
"""

import tensorflow as tf
from tensorpack import *
import cv2
import numpy as np
import cPickle as pickle
import argparse
import glob
from networks import Voxel3D

if __name__ == '__main__':
    '''''''''
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='pattern for images')
    args = parser.parse_args()
    '''

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        with TowerContext("", is_training=False):
            left = tf.placeholder(tf.float32, shape=[1,256,256,3])
            right = tf.placeholder(tf.float32, shape=[1,256,256,3])

            lB, lG, lR = tf.unstack(left, axis=-1)
            rB, rG, rR = tf.unstack(right, axis=-1)
            total_input = tf.stack([lB, lG, lR, rB, rG, rR],
                                   axis=-1)  # 6 channel tensor for combined input of left and right

            embedding = Voxel3D(total_input, 3)

        # new_saver = tf.train.import_meta_graph('./train_log/dumb_network_inter_REAL/graph-0201-162928.meta')
        sess.run(tf.global_variables_initializer())
        # tf.train.Saver().restore(sess, './train_log/dumb_network_inter_REAL/model-90889')
        tf.train.Saver().restore(sess, './train_log/dumb_network_inter_REAL/model-2811')

        img1 = cv2.imread('/graphics/scratch/parallax/captured/all_frames/frame_000008_00.ppm')
        img2 = cv2.imread('/graphics/scratch/parallax/captured/all_frames/frame_000008_02.ppm')

        print(img1.max())

        img1 = cv2.resize(img1, (256, 256))[None, :, :, :]
        img2 = cv2.resize(img2, (256, 256))[None, :, :, :]
        # im = np.concatenate((im2_small, im1_small), axis=-1).astype('float32')
        encodings = sess.run(embedding, {left: img1, right:img2})
        print encodings.shape
        cv2.imwrite('inference2.png', ((encodings[0] + 1)*128).clip(0, 255).astype(np.uint8))
        #scores = np.linalg.norm(encodings, axis=1)

        #print fn, scores
        '''''''''
        
        for fn in glob.glob(args.images):
            img = cv2.imread(fn)
            img = cv2.resize(img, (256,256))
            encodings = sess.run(embedding, {feed_node: img[None, :, :, :]})
            scores = np.linalg.norm(encodings, axis=1)

            print fn, scores
        '''