from __future__ import division, print_function, absolute_import

import tensorflow as tf
from config import myNet
from conv_net import ops
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


# Courtesy #
# https://github.com/dalgu90/resnet-18-tensorflow



def conv_1(X, filters, scope_name):
    f = filters
    with tf.variable_scope(scope_name):
        X = ops.conv_layer(X, k_shape=[7, 7, 3, f], stride=2, padding='SAME', w_init='tn', scope_name='conv_1',
                           add_smry=False)
        X = ops.batch_norm(X, scope_name='bn_1')
        X = ops.activation(X, "relu")
        X = tf.layers.max_pooling2d(X, pool_size=3, padding='SAME', strides=2)
        logging.info('%s : conv_1 shape: %s', str(scope_name), str(X.shape))
    
    return X


def residual_block(X, filters, block_num, dropout, scope_name):
    f0 = X.get_shape().as_list()[-1]
    f1, f2 = filters
    X_shortcut = X
    
    with tf.variable_scope(scope_name):
        X = ops.conv_layer(X, k_shape=[3, 3, f0, f1], stride=1, padding='SAME', w_init='tn', scope_name='conv_1',
                           add_smry=False)
        X = ops.batch_norm(X, scope_name='bn_1')
        X = ops.activation(X, "relu")
        logging.info('%s : conv_1 shape: %s', str(scope_name), str(X.shape))
        
        if dropout is not None:
            logging.info('%s : dropout = %s shape: %s', str(scope_name), str(dropout), str(X.shape))
            X = tf.nn.dropout(X, dropout)
        
        X = ops.conv_layer(X, k_shape=[3, 3, f1, f2], stride=1, padding='SAME', w_init='tn', scope_name='conv_2',
                           add_smry=False)
        X = ops.batch_norm(X, scope_name='bn_2')
        logging.info('%s : conv_2 shape: %s', str(scope_name), str(X.shape))
        
        # Add skip connection
        X = X + X_shortcut
        X = ops.activation(X, 'relu')
        logging.info('%s : Skip add shape: %s', str(scope_name), str(X.shape))
        
        return X


def residual_block_first(X, filters, block_num, dropout, scope_name):
    '''
    Why need this? Normally we have skip connections between 2 layers in one residual block.
    When going from 1 residual block to another we decrease in the image size, In-order to maintain skip connection
    between the layers, we need to have the same dimension for input and output.
    '''
    f0 = X.get_shape().as_list()[-1]
    f1, f2 = filters
    
    with tf.variable_scope(scope_name):
        # We perform a 1x1 conv increasing the num_out channels to equal the number of dimensions. We also perform
        # down sampling of convolutional layer by using a stride of 2
        X_shortcut = ops.conv_layer(X, [1, 1, f0, f1], stride=2, padding='SAME', w_init='tn', scope_name='X_Shortcut',
                                    add_smry=False)
        logging.info('%s : conv_shortcut shape: %s', str(scope_name), str(X_shortcut.shape))
        
        X = ops.conv_layer(X, [3, 3, f0, f1], stride=2, padding='SAME', w_init='tn', scope_name='conv_1',
                           add_smry=False)
        X = ops.batch_norm(X, scope_name='bn_1')
        X = ops.activation(X, 'relu', scope_name='relu_1')
        logging.info('%s : conv_1 shape: %s', str(scope_name), str(X.shape))
        
        #if dropout is not None:
         #   logging.info('%s : dropout = %s shape: %s', str(scope_name), str(dropout), str(X.shape))
          #  X = tf.nn.dropout(X, dropout)
        
        X = ops.conv_layer(X, [3, 3, f1, f2], stride=1, padding='SAME', w_init='tn', scope_name='conv_2',
                           add_smry=False)
        X = ops.batch_norm(X, scope_name='bn_2')
        logging.info('%s : conv_2 shape: %s', str(scope_name), str(X.shape))
        
        X = X + X_shortcut
        X = ops.activation(X, 'relu', scope_name='relu_2')
        logging.info('%s : Skip add shape: %s', str(scope_name), str(X.shape))
    
    return X


def embeddings(inpX, use_dropout):
    filters = [64, 64, 128, 256, 512]
    
    if use_dropout:
        dropout_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    else:
        dropout_prob = [None, None, None, None, None, None, None, None]
    
   
    # Convolution Layer
    X = conv_1(inpX, filters[0], scope_name='conv_layer')
    logging.info('conv_layer : conv shape: %s', str(X.get_shape().as_list()))
    
    # Residual Block 1,2
    X = residual_block(X, [filters[1], filters[1]], block_num=1, dropout=dropout_prob[0],
                       scope_name='residual_block_1_1')
    X = residual_block(X, [filters[1], filters[1]], block_num=2, dropout=dropout_prob[1],
                       scope_name='residual_block_1_2')
    
    # Residual Block 3,4
    X = residual_block_first(X, [filters[2], filters[2]], block_num=3, dropout=dropout_prob[2], scope_name='residual_block_2_1')
    X = residual_block(X, [filters[2], filters[2]], block_num=4, dropout=dropout_prob[3],
                       scope_name='residual_block_2_2')
    
    # Residual block 5,6
    X = residual_block_first(X, [filters[3], filters[3]], block_num=5, dropout=dropout_prob[4], scope_name='residual_block_3_1')
    X = residual_block(X, [filters[3], filters[3]], block_num=6, dropout=dropout_prob[5],
                       scope_name='residual_block_3_2')
    
    # Residual block 7,8
    X = residual_block_first(X, [filters[4], filters[4]], block_num=7, dropout=dropout_prob[6], scope_name='residual_block_4_1')
    X = residual_block(X, [filters[4], filters[4]], block_num=8, dropout=dropout_prob[7],
                       scope_name='residual_block_4_2')
    
    # Flatten (dropout?)
    X = tf.contrib.layers.flatten(X, scope='flatten')
    logging.info('X - flattened: %s', str(X.get_shape().as_list()))
    
    # if use_dropout:
    #     X = tf.nn.dropout(X, 0.7)
    #     logging.info('Flattened : dropout = %s shape: %s', str(0.7), str(X.shape))
    
    # FC-Layer : Get a good 512 encoding to build ensemble
    embeddings = ops.fc_layers(X, [X.get_shape().as_list()[-1], 512], w_init='tn', scope_name='fc_layer1', add_smry=False)
    return embeddings


def resnet(img_shape, device_type, use_dropout):
    inpX = tf.placeholder(dtype=tf.float32,
                          shape=[None, img_shape[0], img_shape[1], img_shape[2]],
                          name='X')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['num_labels']],
                          name='Y')
    
    with tf.device(device_type):
        X_embeddings = embeddings(inpX, use_dropout)
        X_embeddings = ops.activation(X_embeddings, 'relu', scope_name='relu_fc')
        logging.info('X - FC Layer (RELU): %s', str(X_embeddings.get_shape().as_list()))
        
        # SOFTMAX Layer
        X_logits = ops.fc_layers(X_embeddings, [512, 2], w_init='tn', scope_name='fc_layer2', add_smry=False)
        logging.info('LOGITS - Softmax Layer: %s', str(X_logits.get_shape().as_list()))

        Y_probs = tf.nn.softmax(X_logits)
        logging.info('Softmax Y-Prob shape: shape %s', str(Y_probs.shape))

        loss = ops.get_loss(y_true=inpY, y_logits=X_logits, which_loss='sigmoid_cross_entropy', lamda=None)

        optimizer, l_rate = ops.optimize(loss=loss, learning_rate_decay=True, add_smry=False)

        acc = ops.accuracy(labels=inpY, logits=X_logits, type='training', add_smry=False)

    return dict(inpX=inpX, inpY=inpY, outProbs=Y_probs, accuracy=acc, loss=loss, optimizer=optimizer, l_rate=l_rate)


def mixture_of_experts(img_shape, device_type, use_dropout):
    inpX1 = tf.placeholder(dtype=tf.float32,
                          shape=[None, img_shape[0], img_shape[1], img_shape[2]],
                          name='expert1')
    inpX2 = tf.placeholder(dtype=tf.float32,
                          shape=[None, img_shape[0], img_shape[1], img_shape[2]],
                          name='expert2')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['num_labels']],
                          name='Y')

    with tf.device(device_type):
        logging.info('Expert 1: Creating Computation graph for Expert 1 ............... ')
        with tf.variable_scope('Expert1'):
            embeddings_m1 = embeddings(inpX1, use_dropout)
        
        logging.info('Expert 2: Creating Computation graph for Expert 2 ............... ')
        with tf.variable_scope('Expert2'):
            embeddings_m2 = embeddings(inpX2, use_dropout)

        expert_embeddings = tf.concat(values=[embeddings_m1, embeddings_m2], axis=-1)
        expert_embeddings = ops.activation(expert_embeddings, type='sigmoid', scope_name='sigmoid')
        logging.info('EMBEDDINGS: Stacked (sigmoid Gate) %s', str(expert_embeddings.get_shape().as_list()))
    
        # SOFTMAX Layer
        X_logits = ops.fc_layers(expert_embeddings, [1024, 2], w_init='tn', scope_name='softmax', add_smry=False)
        logging.info('LOGITS - Softmax Layer: %s', str(X_logits.get_shape().as_list()))
    
        Y_probs = tf.nn.softmax(X_logits)
        logging.info('Softmax Y-Prob shape: shape %s', str(Y_probs.shape))
    
        loss = ops.get_loss(y_true=inpY, y_logits=X_logits, which_loss='sigmoid_cross_entropy', lamda=None)
    
        optimizer, l_rate = ops.optimize(loss=loss, learning_rate_decay=True, add_smry=False)
    
        acc = ops.accuracy(labels=inpY, logits=X_logits, type='training', add_smry=False)

    return dict(inpX1=inpX1, inpX2=inpX2, inpY=inpY, outProbs=Y_probs, accuracy=acc, loss=loss,
         optimizer=optimizer, l_rate=l_rate)
    