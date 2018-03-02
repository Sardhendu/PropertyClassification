from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

def init_weights(shape, name):
    w = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.02), name=name)
    return w

def init_bias(shape, name):
    b = tf.get_variable(shape=shape, initializer=tf.constant_initializer(0.0), name=name)
    return b
    
def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def conv2d(X, w, b, s):
    conv = tf.nn.conv2d(X, w, strides=s, padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def mean_substract_image(image, mean_pixel):
    return image - mean_pixel

def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
        
        
