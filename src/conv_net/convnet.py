
import logging
import time

import numpy as np
import tensorflow as tf

from src.conv_net import ops

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

def conv_net(conf, img_shape, device_type):
    inpX = tf.placeholder(dtype=tf.float32,
                          shape=[None, img_shape[0], img_shape[1], img_shape[2]],
                          name='X')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, conf['myNet']['num_labels']],
                          name='Y')
    # is_training = tf.placeholder(tf.bool)
    
    filters = [64, 64, 128, 384, 192, 2]
    
    with tf.device(device_type):
        ## LAYER 1
        logging.info('Input shape: %s', str(inpX.shape))
        X = ops.conv_layer(conf, inpX, k_shape=[5, 5, 3, filters[0]], stride=1, padding='SAME', w_init='tn', w_decay=None, scope_name='conv_1', add_smry=False)
        X = ops.batch_norm(conf, X, axis=[0, 1, 2], scope_name='bn_1')
        X = ops.activation(X, 'relu', 'relu_1')
        logging.info('Conv1 shape: %s', str(X.shape))
        X = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
        logging.info('Pool1 shape: %s', str(X.shape))
        
        ## LAYER 2
        X = ops.conv_layer(conf, X, k_shape=[5, 5, filters[0], filters[1]], stride=1, padding='SAME', w_init='tn', w_decay=None, scope_name='conv_2', add_smry=False)
        X = ops.batch_norm(conf, X, axis=[0, 1, 2], scope_name='bn_2')
        X = ops.activation(X, 'relu', 'relu_2')
        logging.info('Conv2 shape: %s', str(X.shape))
        X = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
        logging.info('Pool2 shape: %s', str(X.shape))

        ## LAYER 3
        X = ops.conv_layer(conf, X, k_shape=[5, 5, filters[1], filters[2]], stride=1, padding='SAME', w_init='tn',
                           w_decay=None, scope_name='conv_3', add_smry=False)
        X = ops.batch_norm(conf, X, axis=[0, 1, 2], scope_name='bn_3')
        X = ops.activation(X, 'relu', 'relu_3')
        logging.info('Conv3 shape: %s', str(X.shape))

        X = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
        logging.info('Pool3 shape: %s', str(X.shape))
        
        ## FLATTENED
        X = tf.contrib.layers.flatten(X, scope='flatten')
        logging.info('Flattened shape: %s', str(X.shape))
        
        
        ## LAYER 4
        X = ops.fc_layers(conf, X, k_shape=[X.get_shape().as_list()[-1], filters[3]], w_init='tn', scope_name='fc_layer_1',
                          add_smry=False)
        X = ops.activation(X, 'relu', 'relu_4')
        logging.info('Dense1 shape: %s', str(X.shape))
        
        ## LAYER 5
        X = ops.fc_layers(conf, X, k_shape=[filters[3], filters[4]], w_init='tn', scope_name='fc_layer_2', add_smry=False)
        X = ops.activation(X, 'relu', 'relu_4')
        logging.info('Dense2 shape: %s', str(X.shape))
        
        ## LAYER 6
        X_logits = ops.fc_layers(conf, X, k_shape=[filters[4], filters[5]], w_init='tn', scope_name='fc_layer_4',
                                 add_smry=False)
        logging.info('Output shape: %s', str(X.shape))
        
        Y_probs = tf.nn.softmax(X_logits)

        loss = ops.get_loss(y_true=inpY, y_logits=X_logits,
                            which_loss='softmax_cross_entropy', lamda=None)

        optimizer, l_rate = ops.optimize(loss=loss, learning_rate_decay=True, add_smry=False)

        acc = lambda: ops.accuracy(labels=inpY, logits=X_logits, type='training', add_smry=False)


    return dict(inpX=inpX, inpY=inpY, outProbs=Y_probs, accuracy=acc, loss=loss, optimizer=optimizer, l_rate=l_rate)








def to_one_hot(y):
    y = np.array(y, dtype=int)
    n_values = int(np.max(y)) + 1
    y = np.eye(n_values)[y]
    return y


def OUT(which_device):
    if which_device == 'gpu':
        device_type = '/gpu:0'
    else:
        device_type = '/cpu:0'

    tf.reset_default_graph()
    computation_graph = conv_net([96, 96, 3], device_type)

    config_ = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config_) as sess:
        sess.run(tf.global_variables_initializer())
        x = np.random.random((128, 96, 96, 3))
        y = np.append(np.ones(64), np.zeros(64))
        np.random.shuffle(y)
        y_1hot = to_one_hot(y)

        start_time = time.time()
        for i in range(0, 100):
            feed_dict = {computation_graph['inpX']:x, computation_graph['inpY']:y_1hot}
            loss, _ = sess.run([computation_graph['loss'], computation_graph['optimizer']], feed_dict=feed_dict)

            if (i % 10) == 0:
                print(loss)
        print('Total time ', str(time.time() - start_time))


#
# debugg = False
# if debugg:
#     OUT(which_device='cpu')