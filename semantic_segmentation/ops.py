from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging

seed_arr = [553, 292, 394, 874, 445, 191, 161, 141, 213,436,754,991,302,992,223,645,724,944,32,123,321, 909,784,239,337,888,666, 400,912,255,983,902,846,345,
854,989,291,486,444,101,202,304,505,607,707,808,905, 900, 774,272]

weight_seed_idx = [0]



def conv_layer(X, k_shape, stride=1, padding='SAME', w_init='tn', w_decay=None, scope_name='conv_layer', add_smry=True):
    # m, h, w, f = X.get_shape().as_list()
    
    if weight_seed_idx[0] == len(seed_arr) - 1:
        weight_seed_idx[0] = 0
    
    logging.info('SEED for scope: %s', str(seed_arr[weight_seed_idx[0]]))
    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer(seed=seed_arr[weight_seed_idx[0]])
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.02, seed=seed_arr[weight_seed_idx[0]]
        )
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype=tf.float32,
                shape=k_shape,
                initializer=wght_init,
                name="w",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=[k_shape[-1]],
                initializer=tf.constant_initializer(1),
                name='b',
                trainable=True
        )
    
    if w_decay:
        weight_decay = tf.multiply(tf.nn.l2_loss(weight), w_decay, name='weight_loss')
        tf.add_to_collection('loss_w_decay', weight_decay)
    
    weight_seed_idx[0] += 1
    
    if add_smry:
        tf.summary.histogram("conv_weights", weight)
        tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=padding) + bias

def activation(X, type='relu', scope_name='relu'):
    if type == 'relu':
        return tf.nn.relu(X, name=scope_name)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X, name=scope_name)
    elif type == 'tanh':
        return tf.nn.tanh(X, name=scope_name)
    else:
        raise ValueError('Provide proper Activation function')


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
        
        
