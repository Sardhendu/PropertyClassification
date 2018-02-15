from __future__ import division, print_function, absolute_import

import logging

import tensorflow as tf

import config
from config import netParams, myNet, vars


def conv_layer(X, scope_name):
    logging.info('Running Scope %s: ', str(scope_name))
    stride = netParams[scope_name]['conv_stride']
    pad = netParams[scope_name]['conv_pad']
    k_shape = netParams[scope_name]['conv_shape']

    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0
    
    logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype = tf.float32,
                shape = k_shape,
                initializer=tf.truncated_normal_initializer(
                        stddev=0.1, seed=config.seed_arr[config.weight_seed_idx]
                ),
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
        
    config.weight_seed_idx += 1
    
    tf.summary.histogram("conv_weights", weight)
    tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=pad) + bias


def batch_norm(X, numOUT, axis=[0,1,2], scope_name=None):
    '''
    :param X:            The RELU output to be normalized
    :param numOUT:       Number of output channels (neurons)
    :param decay:        Exponential weighted average
    :param axis:         Normalization axis
    :param scope_name:
    :param trainable:
    :return:
    '''
    logging.info('Running Scope %s: ', str(scope_name))
    decay = config.myNet['batch_norm_decay']
    
    with tf.variable_scope(scope_name):
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="b",  # offset (bias)
                trainable=True
        )
        gamma = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="w",  # scale(weight)
                trainable=True)
        
        expBatchMean_avg = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="m",  # offset (bias)
                trainable=False)
        
        expBatchVar_avg = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="v",  # scale(weight)
                trainable=False)
        
        batchMean, batchVar = tf.nn.moments(X, axes=axis, name="moments")
        trainMean = tf.assign(expBatchMean_avg,
                              decay * expBatchMean_avg + (1 - decay) * batchMean)
        trainVar = tf.assign(expBatchVar_avg,
                             decay * expBatchVar_avg + (1 - decay) * batchVar)
        
        with tf.control_dependencies([trainMean, trainVar]):
            bn = tf.nn.batch_normalization(X,
                                           mean=batchMean,
                                           variance=batchVar,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-5,
                                           name=scope_name)
        return bn
    
    
def fc_layers(X, scope_name):
    logging.info('Running Scope %s: ', str(scope_name))
    k_shape = netParams[scope_name]['shape']
    # print (k_shape)

    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0
        
    logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(dtype=tf.float32,
                                 shape=k_shape,
                                 initializer=tf.truncated_normal_initializer(
                                         stddev=0.1, seed=config.seed_arr[config.weight_seed_idx]
                                 ),
                                 name='w',
                                 trainable=True
                                 )
        bias = tf.get_variable(dtype=tf.float32,
                               shape=[k_shape[-1]],
                               initializer=tf.constant_initializer(1.0),
                               name='b',
                               trainable=True)
    
    config.weight_seed_idx += 1
    
    X = tf.add(tf.matmul(X, weight), bias)

    tf.summary.histogram("fc_weights", weight)
    tf.summary.histogram("fc_bias", bias)
    return X
    


def activation(X, type='relu'):
    if type == 'relu':
        return tf.nn.relu(X)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X)


def loss_optimization(X, y, learning_rate_decay=True):
    globalStep = tf.Variable(0, dtype=tf.float32)
    if learning_rate_decay:
        
        l_rate = tf.train.exponential_decay(myNet['learning_rate'],
                                                  globalStep * vars['batch_size'],  # Used for decay computation
                                                  vars['train_size'],  # Decay steps
                                                  myNet['learning_rate_decay_rate'],  # Decay rate
                                                  staircase=True)  # Will decay the learning rate in discrete interval
        tf.summary.scalar('learning_rate', l_rate)
    else:
        l_rate = myNet['learning_rate']
    
    # We would like to store the summary of the loss to watch the decrease in loss.
    logging.info('INITIALIZING LOSS: .....')
    with tf.name_scope("Loss"):
        lossCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X, labels=y))
        tf.summary.scalar('cross_entropy_loss', lossCE)

    logging.info('INITIALIZING OPTIMIZATION WITH %s: .....', str(myNet['optimizer']))
    with tf.name_scope("Optimizer"):
        if myNet['optimizer'] == 'ADAM':
            optimizer = (tf.train.AdamOptimizer(learning_rate=l_rate)
                         .minimize(lossCE, global_step=globalStep))
        
        elif myNet['optimizer'] == 'RMSPROP':
            optimizer = (tf.train.RMSPropOptimizer(learning_rate=l_rate,
                                                   momentum=myNet['momentum'])
                         .minimize(lossCE, global_step=globalStep)
                         )
        else:
            optimizer = None
            raise ValueError('Your provided optimizers do not match with any of the initialized optimizers')
    
    return lossCE, optimizer, l_rate


def accuracy(labels, yPred):
    with tf.name_scope("Accuracy"):
        pred = tf.equal(tf.argmax(yPred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        return accuracy


def summary_builder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath, sess.graph)
    # writer.add_graph(sess.graph)
    return mergedSummary, writer



#