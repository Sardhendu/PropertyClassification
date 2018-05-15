from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging

seed_arr = [553, 292, 394, 874, 445, 191, 161, 141, 213,436,754,991,302,992,223,645,724,944,32,123,321, 909,784,239,337,888,666, 400,912,255,983,902,846,345,
854,989,291,486,444,101,202,304,505,607,707,808,905, 900, 774,272]

global weight_seed_idx
weight_seed_idx = [0]




#########
# For better understanding, look at:
# https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py

#########



##################################################################################################
def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
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


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# def get_tensor_size(tensor):
#     from operator import mul
#     return reduce(mul, (d.value for d in tensor.get_shape()), 1)

##################################################################################################

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



def conv2D_transposed_strided(X, k_shape, stride=2, padding='SAME', w_init='tn', out_shape=None,
                              scope_name='conv_layer', add_smry=True):
    '''
    :param X:           The input
    :param k_shape:     The shape for weight filter
    :param stride:      Strides, It should take the value 2 if upsampling double
    :param padding:     Should be same
    :param w_init:      which weight initializer (Glorot, truncated etc.)
    :param out_shape:   The output shape of the upsampled data, provide when you know
    :param scope_name:
    :param add_smry:
    :return:
    '''
    weight_seed_idx= []
    weight_seed_idx.append(2)
    if weight_seed_idx[0] == len(seed_arr) - 1:
        weight_seed_idx[0] = 0
    
    logging.info('SEED for scope: %s', str(seed_arr[weight_seed_idx[0]]))
    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer(seed=seed_arr[weight_seed_idx[0]])
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1, seed=seed_arr[weight_seed_idx[0]]
        )
    
    hght, wdth, in_ch, out_ch = k_shape
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype=tf.float32,
                shape=[hght, wdth, out_ch, in_ch],  # Note : We swap the in_out_channels
                initializer=wght_init,
                name="w",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=out_ch,  # k_shape[-1],
                initializer=tf.constant_initializer(1),
                name='b',
                trainable=True
        )
    
    weight_seed_idx[0] += 1
    
    if add_smry:
        tf.summary.histogram("dconv_weights", weight)
        tf.summary.histogram("dconv_bias", bias)
    # print ('dasdasdas', X.shape)
    # print (weight.shape, bias.shape)
    if out_shape is None:
        out_shape = list(X.get_shape().as_list())
        out_shape[1] *= 2  # Should be doubled when upsampling
        out_shape[2] *= 2
        out_shape[3] = out_ch  # weight.get_shape().as_list()[3]
    # print (out_shape)
    conv = tf.nn.conv2d_transpose(X, weight,
                                  tf.stack([tf.shape(X)[0], int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]),
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
    # print (conv.shape)
    return tf.nn.bias_add(conv, bias)





def mean_substract_image(image, mean_pixel):
    return image - mean_pixel

def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
        
        
