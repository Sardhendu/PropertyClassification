from __future__ import division, print_function, absolute_import

import logging

import tensorflow as tf

import config
from config import myNet


def conv_layer(X, k_shape, stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_layer', add_smry=True):
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
    
    if w_decay:
        weight_decay = tf.multiply(tf.nn.l2_loss(weight), w_decay, name='weight_loss')
        tf.add_to_collection('loss_w_decay', weight_decay)
        
    config.weight_seed_idx += 1
    
    if add_smry:
        tf.summary.histogram("conv_weights", weight)
        tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=padding) + bias

def conv2D_transposed_strided(X, k_shape, stride=2, padding='SAME',  w_init='tn', out_shape=None, scope_name='conv_layer', add_smry=True):
    '''
    :param X:           The input
    :param k_shape:     The shape for weight filter
    :param stride:      Strides, It should take the value 2 if upsampling double
    :param padding:     Should be same
    :param w_init:      which weight initializer (Glorot, truncated etc.)
    :param out_shape:   The output shape of the upsampled data
    :param scope_name:
    :param add_smry:
    :return:
    '''

    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0

    logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))

    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer(seed=config.seed_arr[config.weight_seed_idx])
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1, seed=config.seed_arr[config.weight_seed_idx]
        )

    hght, wdth, in_ch, out_ch = k_shape
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype = tf.float32,
                shape = [hght, wdth, out_ch, in_ch],  # Note : We swap the in_out_channels
                initializer=wght_init,
                name="w",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=out_ch,#k_shape[-1],
                initializer=tf.constant_initializer(1),
                name='b',
                trainable=True
        )

    config.weight_seed_idx += 1
    
    if add_smry:
        tf.summary.histogram("dconv_weights", weight)
        tf.summary.histogram("dconv_bias", bias)
    # print ('dasdasdas', X.shape)
    # print (weight.shape, bias.shape)
    if out_shape is None:
        out_shape = list(X.get_shape().as_list())
        out_shape[1] *= 2  # Should be doubled when upsampling
        out_shape[2] *= 2
        out_shape[3] = out_ch#weight.get_shape().as_list()[3]
    # print (out_shape)
    conv = tf.nn.conv2d_transpose(X, weight,
                                  tf.stack([tf.shape(X)[0], int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]),
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
    # print (conv.shape)
    return tf.nn.bias_add(conv, bias)



def batch_norm(X, axis=[0,1,2], scope_name=None):
    '''
    :param X:            The RELU output to be normalized
    :param numOUT:       Number of output channels (neurons)
    :param decay:        Exponential weighted average
    :param axis:         Normalization axis
    :param scope_name:
    :param trainable:
    :return:
    
    Why have: exponential decay with batch norm? In real sense taking the mean and variance of the complete training set
    makes more sense. Since we use batches we would wanna maintain a moving average of the mean and variance. This
    after many batches would approximately equall to the overall mean. Also during the test data collecting the mean
    and variance of the test data is not a good plan, what if test data is from  different distribution. So we apply
    the train mean and variance (calculated by moving average) to the test data too.
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
    
    
def fc_layers(X, k_shape, w_init='tn', scope_name='fc_layer', add_smry=True):
    if config.weight_seed_idx == len(config.seed_arr) - 1:
        config.weight_seed_idx = 0
        
    #logging.info('SEED for scope: %s', str(config.seed_arr[config.weight_seed_idx]))
    
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

    if add_smry:
        tf.summary.histogram("fc_weights", weight)
        tf.summary.histogram("fc_bias", bias)
    return X
    


def activation(X, type='relu', scope_name='relu'):
    if type == 'relu':
        return tf.nn.relu(X, name=scope_name)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X, name=scope_name)


def get_loss(y_true, y_logits, which_loss, lamda=None):
    logging.info('INITIALIZING LOSS: .....')
    loss = None
    with tf.name_scope("Loss"):
        if which_loss == 'sigmoid_cross_entropy':
            # -[y_true log(y_logit) + (1-y_true) log(1-y_logit)]
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, labels=y_true))
        elif which_loss == 'softmax_cross_entropy':
            # An extension of sigmoid_cross_entropy for multiple class
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_true))
        elif which_loss == 'mean_squared_error':
            loss = tf.reduce_mean(tf.pow(y_true - y_logits, 2))
        else:
            raise ValueError('Provide a valid Loss Function')
    
    return loss


def optimize(loss, learning_rate_decay=True, add_smry=True):
    print ('Learning Rate: Initial: ', myNet['learning_rate'])
    globalStep = tf.Variable(0, dtype=tf.float32)
    if learning_rate_decay:
        
        l_rate = tf.train.exponential_decay(myNet['learning_rate'],
                                                  globalStep * myNet['batch_size'],  # Used for decay computation
                                                  myNet['lr_decay_steps'],  # Decay steps,
                                                  myNet['learning_rate_decay_rate'],  # Decay rate
                                                  staircase=True)  # Will decay the learning rate in discrete interval
        
    else:
        l_rate = myNet['learning_rate']
    
    # We would like to store the summary of the loss to watch the decrease in loss.

    logging.info('INITIALIZING OPTIMIZATION WITH %s: .....', str(myNet['optimizer']))
    with tf.name_scope("Optimizer"):
        if myNet['optimizer'] == 'ADAM':
            optimizer = (tf.train.AdamOptimizer(learning_rate=l_rate)
                         .minimize(loss, global_step=globalStep))
        
        elif myNet['optimizer'] == 'RMSPROP':
            optimizer = (tf.train.RMSPropOptimizer(learning_rate=l_rate,
                                                   momentum=myNet['momentum'])
                         .minimize(loss, global_step=globalStep)
                         )
        else:
            optimizer = None
            raise ValueError('Your provided optimizers do not match with any of the initialized optimizers')
    
    if add_smry:
        tf.summary.scalar('learning_rate', l_rate)
        tf.summary.scalar('cross_entropy_loss', loss)
        
    return optimizer, l_rate


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