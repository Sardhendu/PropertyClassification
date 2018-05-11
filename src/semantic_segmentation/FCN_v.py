
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
from src.semantic_segmentation import ops
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
def vgg_fcn_addition(net, keep_prob, inp_shape):
    
    conv5 =  ops.conv_layer(net['pool_4'], [3,3,512,4096], stride=1, padding='SAME', scope_name='conv5')
    conv5 = ops.activation(conv5, 'relu', 'relu_5')
    conv5 = tf.nn.dropout(conv5, 0.5)
    logging.info('conv5 shape: %s', str(conv5.shape))
    
    conv6 = ops.conv_layer(conv5, [3, 3, 4096, 4096], stride=1, padding='SAME', scope_name='conv6')
    conv6 = ops.activation(conv5, 'relu', 'relu_6')
    logging.info('conv6 shape: %s', str(conv6.shape))
    
    

    # Additional CONV layers for UPSCALE
    conv_u1 = ops.conv_layer(net['pool_2'], [3,3,128,3], stride=1, padding='SAME', scope_name='up_conv1')
    logging.info('conv_u1 shape: %s', str(conv_u1.shape))
    conv_u2 = ops.conv_layer(net['pool_3'], [3,3,256,3], stride=1, padding='SAME', scope_name='up_conv2')
    logging.info('conv_u2 shape: %s', str(conv_u2.shape))
    conv_u3 = ops.conv_layer(net['pool_4'], [3,3,512,3], stride=1, padding='SAME', scope_name='up_conv3')
    logging.info('conv_u3 shape: %s', str(conv_u3.shape))
    conv_u4 = ops.conv_layer(conv6, [3,3,4096,3], stride=1, padding='SAME', scope_name='conv_u4')
    logging.info('conv_u4 shape: %s', str(conv_u4.shape))
    
    # Fractionally Strided Convolutions:
    # dconv_u1 = ops.conv2d_transpose_strided(conv_u4, W_t1, b_t1, output_shape=tf.shape(net["pool4"]))

    
    # We deconv on con_u4 and make a outshape of conv_u3 becase we would like to fuse their activation
    dconv_u1 = ops.conv2D_transposed_strided(conv_u4, k_shape=[3,3,3,3], stride=2, padding='SAME', w_init='tn', out_shape=conv_u3.get_shape().as_list(), scope_name='dconv_layer1', add_smry=True)
    #
    logging.info('dconv_u1 shape: %s', str(dconv_u1.shape))
    # fuse_1 = tf.add(conv_u3, dconv_u1, name="fuse_1")
    # logging.info('fuse_1 shape: %s', str(fuse_1.shape))
    #
    # dconv_u2 = ops.conv2D_transposed_strided(fuse_1, k_shape=[3, 3, 3, 3], stride=2, padding='SAME', w_init='tn', out_shape=conv_u2.get_shape().as_list(), scope_name='dconv_layer2', add_smry=True)
    #
    # logging.info('dconv_u2 shape: %s', str(dconv_u2.shape))
    # fuse_2 = tf.add(conv_u2, dconv_u2, name="fuse_2")
    # logging.info('fuse_2 shape: %s', str(fuse_2.shape))
    #
    # dconv_u3 = ops.conv2D_transposed_strided(
    #         fuse_2, k_shape=[3, 3, 3, 3], stride=2,
    #         padding='SAME', w_init='tn',
    #         out_shape=conv_u1.get_shape().as_list(), scope_name='dconv_layer3', add_smry=True)
    #
    # logging.info('dconv_u2 shape: %s', str(dconv_u3.shape))
    # fuse_3 = tf.add(conv_u1, dconv_u3, name="fuse_2")
    # logging.info('fuse_3 shape: %s', str(fuse_3.shape))

    # dconv_u4 = ops.conv2D_transposed_strided(
    #         fuse_3, k_shape=[3, 3, 3, 3], stride=2,
    #         padding='SAME', w_init='tn', out_shape=inp_shape,
    #         scope_name='dconv_layer4', add_smry=True)
    # logging.info('dconv_u4 shape: %s', str(dconv_u4.shape))

    dconv_u4 = ops.conv2D_transposed_strided(
            conv_u4, k_shape=[3, 3, 3, 3], stride=2,
            padding='SAME', w_init='tn', out_shape=inp_shape,
            scope_name='dconv_layer4', add_smry=True)
    logging.info('dconv_u4 shape: %s', str(dconv_u4.shape))

    #
    return dconv_u4


def optimize(loss, l_rate):
    
    # We would like to store the summary of the loss to watch the decrease in loss.
    
    logging.info('INITIALIZING OPTIMIZATION WITH %s: .....', str('ADAM'))
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

    return optimizer

def inference(inp_shape, annotation_shape):
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=inp_shape, name="input_image")
    annotation = tf.placeholder(tf.float32, shape=annotation_shape, name="annotation")
    
    with tf.variable_scope("inference"):
        image_net = vgg_net(image)
        print (image_net)
        logits = vgg_fcn_addition(image_net, keep_prob, inp_shape=[None, 400, 400, 3])
        print ('asdadadas ', logits.shape, annotation.shape)
        loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                              labels=annotation,
                                                                              name="entropy")))
        opt = optimize(loss, l_rate=0.005)
        
    return dict(image=image, annotation=annotation, out_img=logits, loss=loss, optimizer=opt)




def main():
    computation_graph = inference(inp_shape=[None, 400, 400, 3], annotation_shape=[None, 400, 400, 3])

    inp_image = np.random.random((5, 400, 400, 3))
    seg_image = np.random.random((5, 400, 400, 3))
    
    epochs = 10
    with tf.Session() as sess:
        
        for _ in epochs:
            feed_dict = {computation_graph['image']:inp_image, computation_graph['annotation']:seg_image}
            out_image, ls, _ = sess.run([computation_graph['out_img'], computation_graph['loss'],
                                           computation_graph['optimizer']], feed_dict=feed_dict)
            print(out_image)
            print (ls)
            
        
    
    


main()
