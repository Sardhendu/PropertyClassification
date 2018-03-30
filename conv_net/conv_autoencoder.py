
from __future__ import division, print_function, absolute_import
import time
from conv_net import ops
import tensorflow as tf

# TO'DOS
# For assessor images we reshape the image to 224x400 (Do this while creating the batches), then perform central crop
#  of 224x224 and then reshape the image to respected size

import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


# THINGS TO NOTE:
## Why use Sigmoid activations at the end ?
# Scenario 1: Because sigmoid outputs value between 0 and 1 and Images pixels can be
# interpolated from 0-1 to original pixel values. If we use tanh then the output may result in a negative number and
# hence may not be represented properly as the input.

# Scenario: Since we are reconstructing output form input, we would want teh input and ourput represent the same
# distribution. Since we divide the input by 255 we bound the input space between [0,1]. So we use a sigmoid
# activation function to bound the output (logits) at the range [0,1].

def encoder(X, encoding_filters):
    logging.info('INPUT shape: %s', str(X.shape))
    # Downsampling 1
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[0], encoding_filters[1]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_1', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_1')
    logging.info('%s : conv_1 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_1')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
    logging.info('%s : pool_1 shape: %s', 'ENCODER: ', str(X.shape))
    # print(X.shape)
    
    # Downsampling 2
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[1], encoding_filters[2]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_2', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_2')
    logging.info('%s : conv_2 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_2')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
    logging.info('%s : pool_2 shape: %s', 'ENCODER: ', str(X.shape))
    # print(X.shape)
    
    # Downsampling 3
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[2], encoding_filters[3]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_3', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_3')
    logging.info('%s : conv_3 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_3')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
    logging.info('%s : pool_3 shape: %s', 'ENCODER: ', str(X.shape))
    # print(X.shape)

    # Downsampling 4
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[3], encoding_filters[4]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_4', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_4')
    logging.info('%s : conv_4 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_4')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')
    logging.info('%s : pool_4 shape: %s', 'ENCODER: ', str(X.shape))

    # Downloading 5
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[4], encoding_filters[5]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_5', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_5')
    logging.info('%s : conv_5 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_5')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_5')
    logging.info('%s : pool_5 shape: %s', 'ENCODER: ', str(X.shape))

    # Downloading 5
    X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[5], encoding_filters[6]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_6', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_6')
    logging.info('%s : conv_6 shape: %s', 'ENCODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_5')
    # print(X.shape)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_6')
    logging.info('%s : pool_6 shape: %s', 'ENCODER: ', str(X.shape))

    # # Downloading 2: Downsample to 2 features, to plot data points
    # X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[6], encoding_filters[7]], stride=1, padding='SAME',
    #                    w_init='tn', w_decay=None, scope_name='conv_6', add_smry=False)
    # logging.info('%s : conv_7 shape: %s', 'ENCODER: ', str(X.shape))
    # X = ops.activation(X, 'relu', 'relu_7')
    # # print(X.shape)
    # X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_7')
    # logging.info('%s : pool_7 shape: %s', 'ENCODER: ', str(X.shape))
    #
    return X


def decoder(X, enc_fn, decoding_filters):
    # Upsample 1
    # X = ops.conv_layer(X, k_shape=[3,3,enc_fn,decoding_filters[0]],  stride=1, padding='SAME',  w_init='tn',
    # w_decay=None, scope_name='conv_4', add_smry=False)
    # print (X.shape)
    X = ops.conv2D_transposed_strided(X, k_shape=[3 ,3, enc_fn, decoding_filters[0]], stride=2, padding='SAME',
                                      w_init='tn', out_shape=None, scope_name='dconv_1', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
    logging.info('%s : dconv_1 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_6')
    print (X.shape)
    
    # Upsample 2
    # X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[0], decoding_filters[1]], stride=1, padding='SAME',
    # w_init='tn', w_decay=None, scope_name='conv_5', add_smry=False)
    # print(X.shape)
    X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[0], decoding_filters[1]], stride=2,
                                      padding='SAME', w_init='tn', out_shape=None, scope_name='dconv_2', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_8')
    logging.info('%s : dconv_2 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_7')
    # print(X.shape)
    
    # Upsampling 3
    # X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[1], decoding_filters[2]], stride=1, padding='SAME',
    # w_init='tn', w_decay=None, scope_name='conv_6', add_smry=False)
    # print(X.shape)
    X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[1], decoding_filters[2]], stride=2,
                                      padding='SAME', w_init='tn', out_shape=None, scope_name='dconv_3', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_9')
    logging.info('%s : dconv_3 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_8')
    # print(X.shape)
    
    # Upsampling 4
    X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[2], decoding_filters[3]], stride=2,
                                      padding='SAME', w_init='tn', out_shape=None, scope_name='dconv_4', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_10')
    logging.info('%s : dconv_4 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_9')

    # Upsampling 5
    X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[3], decoding_filters[4]], stride=2,
                                      padding='SAME', w_init='tn', out_shape=None, scope_name='dconv_5', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_11')
    logging.info('%s : dconv_5 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_10')

    # Upsampling 6
    X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[4], decoding_filters[5]], stride=2,
                                      padding='SAME', w_init='tn', out_shape=None, scope_name='dconv_6', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_12')
    logging.info('%s : dconv_6 shape: %s', 'DECODER: ', str(X.shape))
    X = ops.activation(X, 'relu', 'relu_11')
    
    
    # Decoding
    X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[5], decoding_filters[6]], stride=1, padding='SAME',
                       w_init='tn', w_decay=None, scope_name='conv_7', add_smry=False)
    X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_13')
    logging.info('%s : conv_7 shape: %s', 'DECODER: ', str(X.shape))
    
    sigmoid_logits = ops.activation(X, 'sigmoid', 'sigmoid_1')
    # print(X.shape)
    
    return X, sigmoid_logits



def conv_autoencoder(img_shape, device_type=None):
    inpX = tf.placeholder(dtype=tf.float32, shape=(None, img_shape[0], img_shape[1], img_shape[2]))
    # enc_filter = [3, 16, 32, 32, 128, 256, 256]  # 3 Represents the number of channels
    # dec_filter = [256, 256, 128, 64, 32, 16, 3]

    enc_filter = [3, 8, 16, 32, 64, 32, 16]  # 3 Represents the number of channels
    dec_filter = [16, 32, 64, 32, 16, 8, 3]

    
    with tf.device(device_type):
        encoded = encoder(inpX, encoding_filters=enc_filter)
        decoded, sigmoid_logits = decoder(encoded, enc_filter[-1], dec_filter)
        
        loss = ops.get_loss(y_true=inpX, y_logits=decoded, which_loss='sigmoid_cross_entropy', lamda=None)
        reconstruction_mse = tf.reduce_mean(tf.pow(inpX - decoded, 2), [1, 2, 3])  # Basically we want to find for each
        reconstruction_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded, labels=inpX)
                                                , [1,2,3])
        # image a reconstruction error, so we calculate mse across [img_width, img_height, img_channel]
        # print ('sdssdsdds ', reconstructionMSE.shape)
        
        optimizer, l_rate = ops.optimize(loss=loss, learning_rate_decay=True, add_smry=False)

        encoded = ops.activation(encoded, 'tanh', 'tanh_1')
        encoded_flattened =tf.layers.flatten(encoded)
        print (encoded_flattened.shape)
    
    return dict(inpX=inpX, encoded=encoded_flattened, decoded=decoded, sigmoid_logits=sigmoid_logits,
                loss=loss, reconstructionMSE=reconstruction_mse, reconstructionEntropy=reconstruction_entropy,
                optimizer=optimizer, learning_rate=l_rate)
#
#
#


# import numpy as np
# from conv_net import utils
# import os
# from data_transformation.data_prep import dumpStratifiedBatches_balanced_class
#
# assessor_code_house_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/input_images
# /assessor_code_images/house'
#
# assessor_code_land_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/input_images/assessor_code_images
# /land'
#
# cmn_house_pins = [pin.split('.')[0] for pin in os.listdir(assessor_code_house_path) if pin != '.DS_Store']
#
# cmn_land_pins = [pin.split('.')[0] for pin in os.listdir(assessor_code_land_path) if pin != '.DS_Store']
#
#
# tr_batch_size = 128
# ts_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
# cv_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
# dumpStratifiedBatches_balanced_class(cmn_land_pins=cmn_land_pins, cmn_house_pins=cmn_house_pins, img_resize_shape=[
# 224, 400, 3],
#                                      image_type='assessor_code',
#                                      ts_batch_size=ts_batch_size,
#                                      cv_batch_size=cv_batch_size,
#                                      tr_batch_size=tr_batch_size,
#                                      shuffle_seed=873,
#                                      get_stats=True,
#                                      max_batches=None)







#
# import numpy as np
# from conv_net import utils
# import matplotlib.pyplot as plt
# import matplotlib
#
# from data_transformation.preprocessing import Preprocessing
# from config import pathDict
# from data_transformation.data_io import getH5File
#
#
# out_img_shape = [64 ,64 ,3]
# def run_preprocessor(sess, dataIN, preprocess_graph, is_training):
#     out_shape = [dataIN.shape[0]] + out_img_shape
#     pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
#     for img_no in np.arange(dataIN.shape[0]):
#         feed_dict = {
#             preprocess_graph['imageIN']: dataIN[img_no, :],
#             preprocess_graph['is_training']: is_training
#         }
#         pp_imgs[img_no, :] = sess.run(
#                 preprocess_graph['imageOUT'],
#                 feed_dict=feed_dict
#         )
#
#     return pp_imgs
#
#
# def load_batch_data(image_type, which_data='cvalid'):
#     if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside',
#                           'google_overlayed', 'assessor_code']:
#         raise ValueError('Can not identify the image type %s, Please provide a valid one' % (str(image_type)))
#
#     data_path = pathDict['%s_batch_path' % (str(image_type))]
#     batch_file_name = '%s' % (which_data)
#
#     # LOAD THE TRAINING DATA FROM DISK
#     dataX, dataY = getH5File(data_path, batch_file_name)
#
#     return dataX, dataY
#
#
# x = np.random.random((100 ,224 ,400 ,3))
# y = np.append(np.ones(50), np.zeros(50))
# np.random.shuffle(y)
# y_1hot = utils.to_one_hot(y)
#
# preprocess_graph = Preprocessing(inp_img_shape=[224, 400 ,3],
#                                  crop_shape=[128 ,128 ,3],
#                                  out_img_shape=[64 ,64 ,3]).preprocessImageGraph()
#
# computation_graph = conv_autoencoder(img_shape=[64 ,64 ,3])


#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     cvalidX, cvalidY = load_batch_data(image_type='assessor_code', which_data='cvalid')
#     print ('cvalid_shape: ', cvalidX.shape)
#
#
#     cvpreprocessed_data = run_preprocessor(sess, cvalidX, preprocess_graph, is_training=False)
#
#
#     for i in range(0 ,100):
#         for batch_num in range(0, 2+ 1):
#             batchX, batchY = load_batch_data(image_type='assessor_code', which_data='train_%s' % (batch_num))
#             preprocessed_data = run_preprocessor(sess, batchX, preprocess_graph, is_training=True)
#             # print(preprocessed_data.shape)
#             feed_dict = {computation_graph['inpX']: preprocessed_data}
#             ls, opt = sess.run([computation_graph['loss'],
#                                           computation_graph['optimizer']], feed_dict=feed_dict)
#             print('Batch_loss : ', ls)
#         feed_dict = {computation_graph['inpX']: cvpreprocessed_data}
#         cv_dec, sig_logits, cv_ls = sess.run([computation_graph['decoded'],
#                                   computation_graph['sigmoid_logits'],
#                                   computation_graph['loss']], feed_dict=feed_dict)
#         # print (cv_dec)
#
#
#         print(cv_dec.shape, cv_ls)
#         if (((i+1)%3) == 0):
#
#             print (cvpreprocessed_data[0, :])
#             print ('')
#             print('')
#             print(cv_dec[0, :])
#             #
#             n = 10
#             # plt.figure(figsize=(20, 4))
#             fig, ax = plt.subplots(2, n, figsize=(20, 4))
#             ax = ax.ravel()
#
#             imaga_indexlist = np.arange(25, 35)
#             for i in range(0, n):
#                 ax[i].imshow(cvpreprocessed_data[imaga_indexlist[i]])
#
#             for i in range(10, 20):
#                 ax[i].imshow(sig_logits[imaga_indexlist[i - 10]])
#
#             fig.show()
#             # time.sleep(10)
#             # plt.close()
#             plt.pause(10)
#             plt.close()
