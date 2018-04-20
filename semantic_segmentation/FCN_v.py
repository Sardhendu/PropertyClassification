
'''
 Code taken from : https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py
 weights: http://www.vlfeat.org/matconvnet/models/
 MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
'''



from __future__ import division, print_function, absolute_import
import os
import scipy.io
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from semantic_segmentation import ops
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s "
                                                                                      "%(message)s")
import datetime
from six.moves import xrange


num_classes = 151

FLAGS = tf.flags.FLAGS
parent_dir = '/Users/sam/All-Program/App-DataSet/DeepNeuralNets/Models/VGG'
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
# tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
# tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
# tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
# tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", os.path.join(parent_dir, 'imagenet-vgg-verydeep-19.mat'), "Path to vgg model mat")

# tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")





def vgg_net(X):
    
    '''
    Code taken from : https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py
    :param weights:
    :param image:
    :return:

    Here we change the matconv net to tensorflow weight shape
    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]
    '''
    
    layers = (
        'conv1_1', 'conv1_2', 'pool_1',

        'conv2_1', 'conv2_2', 'pool_2',

        'conv3_1', 'conv3_2', 'conv3_3', 'pool_3',

        'conv4_1', 'conv4_2', 'conv4_3', 'pool_4'
    )
    
    shapes = [[3, 3, 3, 64], [3, 3, 64, 64], [],
              [3, 3, 64, 128], [3, 3, 128, 128], [],
              [3, 3, 128, 256], [3, 3, 256, 256], [3, 3, 256, 256], [],
              [3, 3, 256, 512], [3,3,512,512], [3,3,512,512], []]
    
    net = OrderedDict()
    
    for i, name in enumerate(layers):
        type_ = name[:4]
        if type_ == 'conv':
            X = ops.conv_layer(X, shapes[i], stride=1, padding='SAME', scope_name=name)
            X = ops.activation(X, 'relu', 'relu_%s'%(str(i)))

        elif type_ == 'pool':
            X = tf.nn.avg_pool(X, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding="SAME")
        
        logging.info('%s : shape = %s', str(name), str(X.shape))
        net[name] = X
    
    return net



# get_variable(weights, name)
def vgg_fcn_addition(net, keep_prob):
    
    conv5 =  ops.conv_layer(net['pool_4'], [3,3,512,4096], stride=1, padding='SAME', scope_name='conv5')
    conv5 = ops.activation(conv5, 'relu', 'relu_5')
    conv5 = tf.nn.dropout(conv5, 0.5)
    logging.info('conv5 shape: %s', str(conv5.shape))
    
    conv6 = ops.conv_layer(conv5, [3, 3, 4096, 4096], stride=1, padding='SAME', scope_name='conv6')
    conv6 = ops.activation(conv5, 'relu', 'relu_6')
    logging.info('conv6 shape: %s', str(conv6.shape))
    
    #
    # conv4_3_layer = net["conv4_3"]
    # logging.info('conv5_3 shape: %s', str(conv4_3_layer.shape))
    # pool5 = tf.nn.max_pool(conv5_3_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # logging.info('pool5 shape: %s', str(pool5.shape))
    #
    # # Lyer 7x7x4096
    # W6 = ops.init_weights(shape = [7, 7, 512, 4096],  name = 'w6')
    # b6 = ops.init_bias(shape=[4096], name='b6')
    # conv6 = ops.conv2d(pool5, W6, b6, s=[1 ,1 ,1 ,1])
    # logging.info('conv6 shape: %s', str(conv6.shape))
    # relu6 = tf.nn.relu(conv6, name="relu6")
    # if FLAGS.debug:
    #     ops.add_activation_summary(relu6)
    # relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
    #
    # # Layer 1x1x4096
    # W7 = ops.init_weights(shape=[1, 1, 4096, 4096], name='w7')
    # b7 = ops.init_bias(shape=[4096], name='b7')
    # conv7 = ops.conv2d(relu_dropout6, W7, b7, s=[1 ,1 ,1 ,1])
    # logging.info('conv7 shape: %s', str(conv7.shape))
    # relu7 = tf.nn.relu(conv7)
    # if FLAGS.debug:
    #     ops.add_activation_summary(relu7)
    # relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
    #
    # # Layer 1x1xnum_of_classes
    # W8 = ops.init_weights(shape=[1, 1, 4096, num_classes], name='w8')
    # b8 = ops.init_bias(shape=[num_classes], name='b8')
    # conv8 = ops.conv2d(relu_dropout7, W8, b8, s=[1 ,1 ,1 ,1])
    # logging.info('conv8 shape: %s', str(conv8.shape))
    #
    #
    #
    # ############ Upscale - Transposed
    # deconv1_shape = net["pool4"].get_shape()
    # W_t1 = ops.init_weights([4, 4, deconv1_shape[3].value, num_classes], name="w_t1")
    # b_t1 = ops.init_bias([deconv1_shape[3].value], name="b_t1")
    # conv_t1 = ops.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(net["pool4"]))
    # logging.info('conv_t1 shape: %s', str(conv_t1.shape))
    # fuse_1 = tf.add(conv_t1, net["pool4"], name="fuse_1")
    #
    # deconv2_shape = net["pool3"].get_shape()
    # W_t2 = ops.init_weights([4, 4, deconv2_shape[3].value, deconv1_shape[3].value], name="w_t2")
    # b_t2 = ops.init_bias([deconv2_shape[3].value], name="b_t2")
    # conv_t2 = ops.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net["pool3"]))
    # logging.info('conv_t2 shape: %s', str(conv_t2.shape))
    # fuse_2 = tf.add(conv_t2, net["pool3"], name="fuse_2")
    #
    #
    # deconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], num_classes])
    # W_t3 = ops.init_weights([16, 16, num_classes, deconv2_shape[3].value], name="W_t3")
    # b_t3 = ops.init_bias([num_classes], name="b_t3")
    # conv_t3 = ops.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv3_shape, stride=8)
    # logging.info('conv_t3 shape: %s', str(conv_t3.shape))
    # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    #
    # return tf.expand_dims(annotation_pred, dim=3) ,conv_t3

def inference(image, keep_prob):
    with tf.variable_scope("inference"):
        image_net = vgg_net(image)
        vgg_fcn_addition(image_net, keep_prob)
        # print (image_net)




def main():
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, 400, 400, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, 224, 224, 1], name="annotation")
    inference(image, keep_prob)


main()
