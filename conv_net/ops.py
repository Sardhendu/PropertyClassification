from __future__ import division, print_function, absolute_import

import logging

import tensorflow as tf

import config
from config import netParams, myNet, vars


def conv_layer(X, k_shape, stride=1, padding='SAME',  w_init='tn', scope_name='conv_layer'):
    m, h, w, f = X.get_shape().as_list()
    
    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0
    
    logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))
    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer(seed=config.seed_arr[config.weight_seed_idx])
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1, seed=config.seed_arr[config.weight_seed_idx]
        )
        
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype = tf.float32,
                shape = k_shape,
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
        
    config.weight_seed_idx += 1
    
    tf.summary.histogram("conv_weights", weight)
    tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=padding) + bias


def batch_norm(X, axis=[0,1,2], scope_name=None):
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
    numOUT = X.get_shape().as_list()[-1]
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
    
    
def fc_layers(X, k_shape, w_init='tn', scope_name='fc_layer'):
    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0
        
    logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))
    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer(seed=config.seed_arr[config.weight_seed_idx])
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1, seed=config.seed_arr[config.weight_seed_idx]
        )
        
        
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(dtype=tf.float32,
                                 shape=k_shape,
                                 initializer=wght_init,
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
    


def activation(X, type='relu', scope_name='relu'):
    if type == 'relu':
        return tf.nn.relu(X, name=scope_name)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X, name=scope_name)


def loss_optimization(X, y, learning_rate_decay=True, add_smry=True):
    print ('Learning Rate: Initial: ', myNet['learning_rate'])
    globalStep = tf.Variable(0, dtype=tf.float32)
    if learning_rate_decay:
        
        l_rate = tf.train.exponential_decay(myNet['learning_rate'],
                                                  globalStep * vars['batch_size'],  # Used for decay computation
                                                  vars['train_size'],  # Decay steps
                                                  myNet['learning_rate_decay_rate'],  # Decay rate
                                                  staircase=True)  # Will decay the learning rate in discrete interval
        
    else:
        l_rate = myNet['learning_rate']
    
    # We would like to store the summary of the loss to watch the decrease in loss.
    logging.info('INITIALIZING LOSS: .....')
    with tf.name_scope("Loss"):
        lossCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X, labels=y))
        

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
    
    if add_smry:
        tf.summary.scalar('learning_rate', l_rate)
        tf.summary.scalar('cross_entropy_loss', lossCE)
        
    return lossCE, optimizer, l_rate


def accuracy(labels, logits, type='training', add_smry=True):
    with tf.name_scope("Accuracy"):
        pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    # if add_smry:
    #     tf.summary.scalar('%s_accuracy'%str(type), acc)
    return acc


def summary_builder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath, sess.graph)
    # writer.add_graph(sess.graph)
    return mergedSummary, writer



#