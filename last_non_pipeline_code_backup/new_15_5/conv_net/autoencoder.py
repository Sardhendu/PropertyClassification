from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print (x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print (x.shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print (x.shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print (x.shape)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print ('ENC ', encoded.shape)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
print (x.shape)
x = UpSampling2D((2, 2))(x)
print (x.shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
x = UpSampling2D((2, 2))(x)
print (x.shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
x = UpSampling2D((2, 2))(x)
print (x.shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
print (x.shape)
x = UpSampling2D((2, 2))(x)
print (x.shape)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
print ('DASDAS ', decoded.shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#
# ENCODER:  : conv_1 shape: (?, 64, 64, 16)
# 2018-03-27 01:37:25,914 INFO     ENCODER:  : pool_1 shape: (?, 32, 32, 16)
# 2018-03-27 01:37:25,915 INFO     SEED for scope: 292
# 2018-03-27 01:37:25,937 INFO     ENCODER:  : conv_2 shape: (?, 32, 32, 32)
# 2018-03-27 01:37:25,940 INFO     ENCODER:  : pool_2 shape: (?, 16, 16, 32)
# 2018-03-27 01:37:25,940 INFO     SEED for scope: 394
# 2018-03-27 01:37:25,960 INFO     ENCODER:  : conv_3 shape: (?, 16, 16, 64)
# 2018-03-27 01:37:25,963 INFO     ENCODER:  : pool_3 shape: (?, 8, 8, 64)
# 2018-03-27 01:37:25,963 INFO     SEED for scope: 874
# 2018-03-27 01:37:25,984 INFO     ENCODER:  : conv_4 shape: (?, 8, 8, 128)
# 2018-03-27 01:37:25,987 INFO     ENCODER:  : pool_4 shape: (?, 4, 4, 128)
# 2018-03-27 01:37:25,987 INFO     SEED for scope: 445
# 2018-03-27 01:37:26,019 INFO     DECODER:  : dconv_1 shape: (?, 8, 8, 128)
# 2018-03-27 01:37:26,020 INFO     SEED for scope: 191
# 2018-03-27 01:37:26,055 INFO     DECODER:  : dconv_2 shape: (?, 16, 16, 64)
# 2018-03-27 01:37:26,057 INFO     SEED for scope: 161
# 2018-03-27 01:37:26,091 INFO     DECODER:  : dconv_3 shape: (?, 32, 32, 32)
# 2018-03-27 01:37:26,092 INFO     SEED for scope: 141
# 2018-03-27 01:37:26,134 INFO     DECODER:  : dconv_4 shape: (?, 64, 64, 16)
# 2018-03-27 01:37:26,135 INFO     SEED for scope: 213
# 2018-03-27 01:37:26,161 INFO     DECODER:  : conv_5 shape: (?, 64, 64, 3)
#


from config import pathDict
from data_transformation.data_io import getH5File
import tensorflow as tf
from data_transformation.preprocessing import Preprocessing
out_img_shape = [64, 64, 3]


def run_preprocessor(sess, dataIN, preprocess_graph, is_training):
    out_shape = [dataIN.shape[0]] + out_img_shape
    pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
    for img_no in np.arange(dataIN.shape[0]):
        feed_dict = {
            preprocess_graph['imageIN']: dataIN[img_no, :],
            preprocess_graph['is_training']: is_training
        }
        pp_imgs[img_no, :] = sess.run(
                preprocess_graph['imageOUT'],
                feed_dict=feed_dict
        )
    
    return pp_imgs

def load_batch_data(image_type, which_data='cvalid'):
    if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside',
                          'google_overlayed', 'assessor_code']:
        raise ValueError('Can not identify the image type %s, Please provide a valid one' % (str(image_type)))
    
    data_path = pathDict['%s_batch_path' % (str(image_type))]
    batch_file_name = '%s' % (which_data)
    
    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY = getH5File(data_path, batch_file_name)
    
    return dataX, dataY

from keras.datasets import mnist
import numpy as np

preprocess_graph = Preprocessing(inp_img_shape=[224, 400 ,3],
                                 crop_shape=[128 ,128 ,3],
                                 out_img_shape=[64 ,64 ,3]).preprocessImageGraph()



ds = [0, 128,256, 480]
x_train = np.ndarray((128+128+224, 64,64,3))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    cv_dataX, cv_batchY = load_batch_data(image_type='assessor_code', which_data='cvalid')
    x_test = run_preprocessor(sess, cv_dataX, preprocess_graph, is_training=False)
    for batch_num in range(0, 2 + 1):
        # print ('train_%s' % (batch_num))
        batchX, batchY = load_batch_data(image_type='assessor_code', which_data='train_%s' % (batch_num))
        
        
        preprocessed_data = run_preprocessor(sess, batchX, preprocess_graph, is_training=True)
        # print (preprocessed_data.shape)
        # print (ds[batch_num],ds[batch_num+1])
        x_train[ds[batch_num]:ds[batch_num+1], :] = preprocessed_data
        
print (x_train.shape, x_test.shape)
# print (x_train[400:480,:])
# x_train = x_train.astype('float32') #/ 255.
# x_test = x_test.astype('float32') #/ 255.
# x_train = np.reshape(x_train, (len(x_train), 64, 64, 3))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 64, 64, 3))  # adapt this if using `channels_first` image data format


autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)

n = 10
# plt.figure(figsize=(20, 4))
fig, ax = plt.subplots(2, n, figsize=(20, 4))
ax = ax.ravel()

imaga_indexlist = np.arange(25, 35)
for i in range(0,n):
    ax[i].imshow(x_test[imaga_indexlist[i]])


for i in range(10, 20):
    ax[i].imshow(decoded_imgs[imaga_indexlist[i-10]])

fig.show()







# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# def get_weight(shape, name):
#     return tf.get_variable(dtype=tf.float32, shape=shape, initializer=tf.truncated_normal_initializer(0.01), name=name)
#
# from conv_net import ops
# print ('')
#
# import tensorflow as tf
#
# encoding_filters = [16,8,8]
# decoder_filter = [8,8,16]
#
# # Encoder
# X = tf.placeholder(dtype=tf.float32, shape=(None, 28,28,1))
#
# X = ops.conv_layer(X, k_shape=[3,3,1,encoding_filters[0]], stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_1', add_smry=True)
# print (X.shape)
# X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
# print (X.shape)
#
# X = ops.conv_layer(X, k_shape=[3,3,encoding_filters[0],encoding_filters[1]], stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_2', add_smry=True)
# print (X.shape)
# X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
# print (X.shape)
#
#
# X = ops.conv_layer(X, k_shape=[3,3,encoding_filters[1],encoding_filters[2]],
#                    stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_3', add_smry=True)
# print (X.shape)
# encoded = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
# print (encoded.shape)
#
#
# # Decoder with Fractionally Strided Convolutions
# X = ops.conv_layer(encoded, k_shape=[3,3,encoding_filters[2],decoder_filter[0]], stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_4', add_smry=True)
# print (X.shape)
# w1 = get_weight(shape=[3,3,encoding_filters[2],decoder_filter[0]], name='w1')
# # print (w1.shape)
# X = tf.nn.conv2d_transpose(X,
#                            w1,
#                            output_shape=tf.stack([tf.shape(X)[0], 8, 8, 8]),
#                            strides=[1,2,2,1],
#                            padding='SAME')
# print (X.shape)
#
# X = ops.conv_layer(X, k_shape=[3,3,decoder_filter[0],decoder_filter[1]], stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_5', add_smry=True)
# print (X.shape)
#
#
# w2 = get_weight(shape=[3,3,decoder_filter[0],decoder_filter[1]], name='w2')
# # print (w1.shape)
# X = tf.nn.conv2d_transpose(X,
#                            w2,
#                            output_shape=tf.stack([tf.shape(X)[0], 16, 16, decoder_filter[1]]),
#                            strides=[1,2,2,1],
#                            padding='SAME')
# print (X.shape)
#
# X = ops.conv_layer(X, k_shape=[3,3,decoder_filter[1],decoder_filter[2]], stride=1, padding='VALID',  w_init='tn', w_decay=None, scope_name='conv_6', add_smry=True)
# print (X.shape)  # VALID padding becasue we would like to shrink the input with stride 2. Also because, the encoder was downsampled from 28x28 to 14x14
#
# w2 = get_weight(shape=[3,3,decoder_filter[1],decoder_filter[2]], name='w3')
# # print (w1.shape)
# X = tf.nn.conv2d_transpose(X,
#                            w2,
#                            output_shape=tf.stack([tf.shape(X)[0], 28, 28, decoder_filter[2]]),
#                            strides=[1,2,2,1],
#                            padding='SAME')
# print (X.shape)
#
# X = ops.conv_layer(X, k_shape=[3,3,decoder_filter[2],1], stride=1, padding='SAME',  w_init='tn', w_decay=None, scope_name='conv_7', add_smry=True)
# print (X.shape)
# #




# tf.nn.conv2d_transpose(
#                 current_input, W,
#                 tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
#                 strides=[1, 2, 2, 1], padding='SAME'), b))