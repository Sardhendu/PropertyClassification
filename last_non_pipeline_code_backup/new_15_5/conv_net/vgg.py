from __future__ import division, print_function, absolute_import

import logging

import tensorflow as tf

from config import netParams
from conv_net.ops import conv_layer, batch_norm, activation, fc_layers, loss_optimization
from config import myNet




####################################################################
## Main
#####################################################################
def conv_1(X):
    name = 'conv1'
    k_shape = netParams[name]['conv_stride']
    stride = netParams[name]['conv_stride']
    pad = netParams[name]['conv_pad']
    
    with tf.variable_scope('conv1_layer'):
        logging.info('CONV_1 .........................')
        X = conv_layer(X, k_shape, stride, pad, scope_name='conv_1')
        X = batch_norm(X, axis=[0, 1, 2], scope_name='bn_1')
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('CONV_1: shape %s', str(X.shape))
        
        X = tf.layers.max_pooling2d(X, pool_size=netParams['conv1']['pool_size'],
                                    padding=netParams['conv4']['pool_pad'],
                                    strides=netParams['conv1']['pool_stride'], data_format='channels_last')
        logging.info('MAXPOOL_1: shape %s', str(X.shape))
        # X = tf.nn.dropout(X, netParams['conv1']['keep_prob'], seed=config.seed_arr[1])
        #
    return X


def conv_2(X):
    name = 'conv2'
    k_shape = netParams[name]['conv_stride']
    stride = netParams[name]['conv_stride']
    pad = netParams[name]['conv_pad']
    
    with tf.variable_scope('conv2_layer'):
        logging.info('CONV_2 .........................')
        X = conv_layer(X, k_shape, stride, pad, scope_name='conv_1')
        X = batch_norm(X, axis=[0, 1, 2], scope_name='bn_1')
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('CONV_2: shape %s', str(X.shape))
        
        X = tf.layers.max_pooling2d(X, pool_size=netParams['conv2']['pool_size'],
                                    padding=netParams['conv4']['pool_pad'],
                                    strides=netParams['conv2']['pool_stride'], data_format='channels_last')
        logging.info('MAXPOOL_2: shape %s', str(X.shape))
        # X = tf.nn.dropout(X, netParams['conv2']['keep_prob'], seed=config.seed_arr[2])
    
    return X


def conv_3(X):
    name = 'conv3'
    k_shape = netParams[name]['conv_stride']
    stride = netParams[name]['conv_stride']
    pad = netParams[name]['conv_pad']
    
    with tf.variable_scope('conv3_layer'):
        logging.info('CONV_3 .........................')
        X = conv_layer(X, k_shape, stride, pad, scope_name='conv_1')
        X = batch_norm(X, axis=[0, 1, 2], scope_name='bn_1')
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('CONV_3 shape %s', str(X.shape))
        
        X = tf.layers.max_pooling2d(X, pool_size=netParams['conv3']['pool_size'],
                                    padding=netParams['conv4']['pool_pad'],
                                    strides=netParams['conv3']['pool_stride'], data_format='channels_last')
        logging.info('MAXPOOL_3: shape %s', str(X.shape))
        # X = tf.nn.dropout(X, netParams['conv2']['keep_prob'], seed=config.seed_arr[2])
    
    return X


def conv_4(X):
    name = 'conv4'
    k_shape = netParams[name]['conv_stride']
    stride = netParams[name]['conv_stride']
    pad = netParams[name]['conv_pad']
    
    with tf.variable_scope('conv4_layer'):
        logging.info('CONV_4 .........................')
        X = conv_layer(X, k_shape, stride, pad, scope_name=name)
        X = batch_norm(X, axis=[0, 1, 2], scope_name='bn4')
        X = activation(X, type='relu', scope_name='relu')
        logging.info('CONV_4: shape %s', str(X.shape))
        
        X = tf.layers.max_pooling2d(X, pool_size=netParams['conv4']['pool_size'],
                                    padding=netParams['conv4']['pool_pad'],
                                    strides=netParams['conv4']['pool_stride'], data_format='channels_last')
        logging.info('MAXPOOL_4: shape %s', str(X.shape))
        # X = tf.nn.dropout(X, netParams['conv2']['keep_prob'], seed=config.seed_arr[2])
    
    return X


def fc1(X, dropout=True):
    k_shape = netParams['fc1']['shape']
    with tf.variable_scope('fc1_layer'):
        logging.info('FC_1 .........................')
        X = fc_layers(X, k_shape, scope_name='fc_1')
        logging.info('FC_1: shape %s', str(X.shape))
        # X = batch_norm(X, X.get_shape().as_list()[-1], axis=[0, 1, 2], scope_name='bn2')
        # logging.info('batch_norm2: shape %s', str(X.shape))
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('RELU_5: shape %s', str(X.shape))
        if dropout:
            X = tf.nn.dropout(X, netParams['fc1']['keep_prob'])#, seed=config.seed_arr[10])
    return X


def fc2(X, dropout=True):
    k_shape = netParams['fc2']['shape']
    with tf.variable_scope('fc2_layer'):
        logging.info('FC_2 .........................')
        X = fc_layers(X, k_shape, scope_name='fc_1')
        logging.info('FC_2: shape %s', str(X.shape))
        # X = batch_norm(X, X.get_shape().as_list()[-1], axis=[0, 1, 2], scope_name='bn2')
        # logging.info('batch_norm2: shape %s', str(X.shape))
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('RELU_6: shape %s', str(X.shape))
        if dropout:
            X = tf.nn.dropout(X, netParams['fc2']['keep_prob'])#, seed=config.seed_arr[11])
    return X

def fc3(X, dropout=True):
    k_shape = netParams['fc3']['shape']
    with tf.variable_scope('fc3_layer'):
        logging.info('FC_3 .........................')
        X = fc_layers(X, k_shape, scope_name='fc_1')
        logging.info('FC_3: shape %s', str(X.shape))
        # X = batch_norm(X, X.get_shape().as_list()[-1], axis=[0, 1, 2], scope_name='bn2')
        # logging.info('batch_norm2: shape %s', str(X.shape))
        X = activation(X, type='relu', scope_name='relu_1')
        logging.info('RELU_7: shape %s', str(X.shape))
        if dropout:
            X = tf.nn.dropout(X, netParams['fc3']['keep_prob'])#, seed=config.seed_arr[12])
    return X

def softmax(X):
    k_shape = netParams['softmax']['shape']
    with tf.variable_scope('softmax_layer'):
        logging.info('SOFTMAX .........................')
        logits = fc_layers(X, k_shape, scope_name='softmax')
        logging.info('Softmax: shape %s', str(X.shape))
        probs = tf.nn.softmax(logits)
    return logits, probs


def vgg(training=True):
    inpX = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['crop_shape'][0], myNet['crop_shape'][1], myNet['crop_shape'][2]],
                          name='X')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['num_labels']],
                          name='Y')
    
    X = conv_1(inpX)
    X = conv_2(X)
    X = conv_3(X)
    X = conv_4(X)
    X = tf.contrib.layers.flatten(X, scope='flatten')
    netParams['fc1']['shape'][0] = X.get_shape().as_list()[1]
    
    X = fc1(X, dropout=True)
    X = fc2(X, dropout=True)
    X = fc3(X, dropout=True)
    logging.info('The X till this point is: shape %s', str(X.get_shape().as_list()))
    
    logits, probs = softmax(X)
    logging.info('Logits: shape %s', str(logits.get_shape().as_list()))
    logging.info('Probabilities: shape %s', str(probs.get_shape().as_list()))

    if training:
        lossCE, optimizer, l_rate = loss_optimization(X=logits, y=inpY, learning_rate_decay=True)
    
        return dict(inpX=inpX, inpY=inpY, outProbs=probs,
                    loss=lossCE, optimizer=optimizer, l_rate=l_rate)
    else:
        return dict(inpX=inpX, inpY=inpY, outProbs=probs)


# def vgg_test_graph():
#     inpX = tf.placeholder(dtype=tf.float32,
#                           shape=[None, myNet['crop_shape'][0], myNet['crop_shape'][1], myNet['crop_shape'][2]],
#                           name='X')
#     inpY = tf.placeholder(dtype=tf.float32,
#                           shape=[None, myNet['num_labels']],
#                           name='Y')
#
#     X = conv_1(inpX)
#     X = conv_2(X)
#     X = conv_3(X)
#     X = conv_4(X)
#     X = tf.contrib.layers.flatten(X, scope='flatten')
#     netParams['fc1']['shape'][0] = X.get_shape().as_list()[1]
#
#     X = fc1(X, dropout=False)
#     X = fc2(X, dropout=False)
#     X = fc3(X, dropout=False)
#     logging.info('The X till this point is: shape %s', str(X.get_shape().as_list()))
#
#     logits, probs = softmax(X)
#     logging.info('Logits: shape %s', str(logits.get_shape().as_list()))
#     logging.info('Probabilities: shape %s', str(probs.get_shape().as_list()))
#
#     return dict(inpX=inpX, inpY=inpY, outProbs=probs)