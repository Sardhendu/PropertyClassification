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
import keras.backend as K
from keras import metrics

# THINGS TO NOTE:
## Why use Sigmoid activations at the end ?
# Scenario 1: Because sigmoid outputs value between 0 and 1 and Images pixels can be
# interpolated from 0-1 to original pixel values. If we use tanh then the output may result in a negative number and
# hence may not be represented properly as the input.

# Scenario: Since we are reconstructing output form input, we would want teh input and ourput represent the same
# distribution. Since we divide the input by 255 we bound the input space between [0,1]. So we use a sigmoid
# activation function to bound the output (logits) at the range [0,1].


#
#
#
# def sample_gaussian_1var(z_mean, z_log_var, epsilon_sd):
#     ''' WE SAMPLE FROM A RANDOM UNIT GAUSSIAN :
#         In order to estimate z (latent space) from x (input space), we need to sample z from p(z|x). Since we learn
#         analytically by gradient updating. This means that the notation should be differentiable, which even means
#         that the model is deterministic i.e a fized value of parameter to the network will always output the same
#         output. Therefore, the only souce to add stocasticity (randomness) is via the input x.
#
#         Denoising Autoencoder: In denoising Autoencoders we learn that adding random noise to the input image actually
#         controls overfitting by trying to learn a reconstruction to map output (learned from noisy input) to the
#         actual input. Variational Autoencoder: In this case we incorporate a probabilistic sampling node (
#         random_normal) which makes the model stocastic. Here, epsilon id the randomness (p(epsilon)). The Encoding
#         layer send to the decoder is just the addition of Mean and variance latent representation. Here,
#         we just multiply the randomness to the variance
#         latent representation and add it to the mean latent representation.
#         And thus we bring Radomness to out model and the inference and generation become entirely differentiable and
#         learnable.
#
#         Why do we do unit Gaussian: Theoratically, the encodings can take any values, squashing the
#         encodings with activation doesn't seem a right. Having a gaussian with unit variance is sort of normalizing
#         and adding random noise to control overfitting. For example if we are learning mnist data, say digit 1 has a
#         mean 5.2 and digit 9 has a mean 6.4. Now suppose, digit 1 comes in and the mean is found to be 6,
#         then the network may say than its 9. By sampling method we learn the variance representation. So VAE method
#         would say digit 1 = N(mean=5.2, sd = 1) and digit 9 = N(mean=6.4, sd=0.1). Using this phenomena the network
#         would learn the right classification. IMAGE MIXTURES OF GAUSSIAN.
#     '''
#
#     # latent_dim = 32
#     with tf.name_scope("sample_gaussian"):
#         epsilon = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=epsilon_sd, dtype=tf.float32)
#         sum_mean_std = z_mean + epsilon * tf.exp(z_log_var / 2)
#     return sum_mean_std
#
#
# def vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var, input_dimension):
#     '''
#     :param x:
#     :param x_decoded_mean:
#     :return:
#
#     The loss si defined by two quantities:
#
#     1. Reconstruction loss: Cna be binary_cross_entropy (sigmoid_cross_entropy) or mean_squared_error
#
#     2. The regularized loss: kl_loss = The kl_loss can be thought as the relative entropy between the prior p(z) and
#     the posterior distribution q(z). The posterior q(z).
#     The formula seems a little weird because we strategically choose certain conjugate priors over z that let us
#     analytically integrate the KL divergence, yielding a closed form equation. So,
#     -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + log(z_var) - z_mean^2 -z_var), which can also be written as
#     -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + z_var_log - z_mean^2 - exp(z_var_log))
#     '''
#
#     # Compute VAE loss
#     with tf.name_scope("binary_cross_entropy_loss"):
#         xent_loss = input_dimension * metrics.binary_crossentropy(
#                 K.flatten(x),
#                 K.flatten(x_decoded_mean_squash))
#         # b_xent_loss = input_dimension * ops.get_loss(y_true=x, y_logits=x_decoded_mean,
# which_loss='sigmoid_cross_entropy', lamda=None)  # Entropy loss of input and
#         # # b_xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
#         # print('12121212121212', b_xent_loss.shape)
#         #
#         # b_xent_loss = input_dimension * metrics.binary_crossentropy(
#         #         K.flatten(x),
#         #         K.flatten(x_decoded_mean))
#     with tf.name_scope("KL_divergence_loss"):
#         """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
#         kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#
#     vae_loss = K.mean(xent_loss + kl_loss)
#
#     # generated_flat = tf.reshape(self.generated_images, [self.batchsize, 224 * 224 * 3])
#     # kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_log_var) - tf.log(tf.square(z_log_var)) - 1, 1)
#     # vae_loss = K.mean(xent_loss + kl_loss)
#
#     # The KL-divergence loss tries to bring the latent variables closer to a unit gaussian distribution. and the
#     # binary_cross_entropy minimizes the reconstruction error. Former is trying to make things seems a bit random,
#     # whereas the later is trying to learn the input properly
#     # vae_loss = tf.reduce_mean(b_xent_loss + kl_loss, name="VAE_total_cost")
#
#     return vae_loss, xent_loss, kl_loss
#
#
# def encoder(X, encoding_filters):
#     logging.info('INPUT shape: %s', str(X.shape))
#     # Downsampling 1
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[0], encoding_filters[1]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='conv_1', add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_1')
#     logging.info('ENCODER: conv_1 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'enc_relu_1')
#     # print(X.shape)
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
#     logging.info('ENCODER : pool_1 shape: %s'%(str(X.shape)))
#     # print(X.shape)
#
#     # Downsampling 2
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[1], encoding_filters[2]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='conv_2', add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_2')
#     logging.info('ENCODER : conv_2 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'enc_relu_2')
#     # print(X.shape)
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
#     logging.info('ENCODER : pool_2 shape: %s'%(str(X.shape)))
#     # print(X.shape)
#
#     # Downsampling 3
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[2], encoding_filters[3]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='conv_3', add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_3')
#     logging.info('ENCODER : conv_3 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'enc_relu_3')
#     # print(X.shape)
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
#     logging.info('ENCODERpool_3 shape: %s'%(str(X.shape)))
#     # print(X.shape)
#
#     # Downsampling 4
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[3], encoding_filters[4]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='conv_4', add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_4')
#     logging.info('ENCODER : conv_4 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'enc_relu_4')
#     # print(X.shape)
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')
#     logging.info('ENCODER : pool_4 shape: %s'%(str(X.shape)))
#
#
#     # FULLY CONNECTED LAYERS
#     X_flat = tf.layers.flatten(X, name='enc_flattened')
#     logging.info('ENCODER : Flattened shape: %s'%(str(X_flat.shape)))
#
#     # FC 1
#     X_fc1 = ops.fc_layers(X_flat, k_shape = [X_flat.get_shape().as_list()[-1], 8*8*64], w_init='gu',
#                           scope_name='enc_fc_layer1', add_smry=False)
#     X_fc1 = ops.activation(X_fc1, 'relu', 'enc_relu_5')
#     logging.info('ENCODER : FC shape: %s'%(str(X_fc1.shape)))
#
#     # FC 2
#     X_fc2 = ops.fc_layers(X_fc1, k_shape=[8*8*64, 512], w_init='gu',
#                           scope_name='enc_fc_layer2', add_smry=False)
#     X_fc2 = ops.activation(X_fc2, 'relu', 'enc_relu_6')
#     logging.info('ENCODER : FC shape: %s'%(str(X_fc2.shape)))
#
#     ##### Latent Layer
#     z_mean = ops.fc_layers(X_fc2, k_shape=[512, 64], w_init='gu',
#                           scope_name='enc_z_mean', add_smry=False)
#     logging.info('ENCODER : z_mean shape: %s'%(str(z_mean.shape)))
#
#     z_log_var = ops.fc_layers(X_fc2, k_shape=[512, 64], w_init='gu',
#                            scope_name='enc_z_log_var', add_smry=False)
#     logging.info('ENCODER : z_log_var shape: %s'%(str(z_log_var.shape)))
#
#     # # Downsampling 5
#     # X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[4], encoding_filters[5]], stride=1, padding='SAME',
#     #                    w_init='gu', w_decay=None, scope_name='conv_5', add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_5')
#     # logging.info('%s : conv_5 shape: %s', 'ENCODER: ', str(X.shape))
#     # X = ops.activation(X, 'relu', 'relu_5')
#     # # print(X.shape)
#     # X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_5')
#     # logging.info('%s : pool_5 shape: %s', 'ENCODER: ', str(X.shape))
#     #
#     # # Downsampling 5
#     # X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[5], encoding_filters[6]], stride=1, padding='SAME',
#     #                    w_init='gu', w_decay=None, scope_name='conv_6', add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_6')
#     # logging.info('%s : conv_6 shape: %s', 'ENCODER: ', str(X.shape))
#     # X = ops.activation(X, 'relu', 'relu_5')
#     # # print(X.shape)
#     # X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_6')
#     # logging.info('%s : pool_6 shape: %s', 'ENCODER: ', str(X.shape))
#
#     # # Downloading 2: Downsample to 2 features, to plot data points
#     # X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[6], encoding_filters[7]], stride=1, padding='SAME',
#     #                    w_init='gu', w_decay=None, scope_name='conv_6', add_smry=False)
#     # logging.info('%s : conv_7 shape: %s', 'ENCODER: ', str(X.shape))
#     # X = ops.activation(X, 'relu', 'relu_7')
#     # # print(X.shape)
#     # X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_7')
#     # logging.info('%s : pool_7 shape: %s', 'ENCODER: ', str(X.shape))
#     #
#     return z_mean, z_log_var
#
#
# def decoder(sampled_z, decoding_filters):
#     # Upsample 1
#     # X = ops.conv_layer(X, k_shape=[3,3,enc_fn,decoding_filters[0]],  stride=1, padding='SAME',  w_init='gu',
#     # w_decay=None, scope_name='conv_4', add_smry=False)
#     # print (X.shape)
#     # X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, enc_fn, decoding_filters[0]], stride=2, padding='SAME',
#     #                                   w_init='gu', out_shape=None, scope_name='dconv_1', add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     # logging.info('%s : dconv_1 shape: %s', 'DECODER: ', str(X.shape))
#     # X = ops.activation(X, 'relu', 'relu_6')
#     # print(X.shape)
#     #
#     # # Upsample 2
#     # # X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[0], decoding_filters[1]], stride=1, padding='SAME',
#     # # w_init='gu', w_decay=None, scope_name='conv_5', add_smry=False)
#     # # print(X.shape)
#     # X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[0], decoding_filters[1]], stride=2,
#     #                                   padding='SAME', w_init='gu', out_shape=None, scope_name='dconv_2',
# add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_8')
#     # logging.info('%s : dconv_2 shape: %s', 'DECODER: ', str(X.shape))
#     # X = ops.activation(X, 'relu', 'relu_7')
#     # print(X.shape)
#
#     X_fc1 = ops.fc_layers(sampled_z, k_shape=[64, 512], w_init='gu', scope_name='dec_fc_layer1', add_smry=False)
#     X_fc1 = ops.activation(X_fc1, 'relu', 'dec_relu_1')
#
#     X_fc2 = ops.fc_layers(X_fc1, k_shape=[512, 8*8*64], w_init='gu', scope_name='dec_fc_layer2', add_smry=False)
#     X_fc2 = ops.activation(X_fc2, 'relu', 'dec_relu_2')
#
#     rsp_X_fc2 = tf.reshape(X_fc2, [tf.shape(X_fc2)[0], 8, 8, 64])
#
#     # Upsampling 1
#     # X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[1], decoding_filters[2]], stride=1, padding='SAME',
#     # w_init='gu', w_decay=None, scope_name='conv_6', add_smry=False)
#     # print(X.shape)
#     X = ops.conv2D_transposed_strided(rsp_X_fc2, k_shape=[3, 3, decoding_filters[0], decoding_filters[1]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dconv_3',
# add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_9')
#     logging.info('DECODER : dconv_1 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'dec_relu_3')
#     # print(X.shape)
#
#     # Upsampling 2
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[1], decoding_filters[2]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dconv_4',
# add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_10')
#     logging.info('DECODER : dconv_2 shape: %s'%(str(X.shape)))
#     X = ops.activation(X, 'relu', 'dec_relu_4')
#
#     # Upsampling 3
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[2], decoding_filters[3]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dconv_5',
# add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_11')
#     X = ops.activation(X, 'relu', 'dec_relu_5')
#     logging.info('DECODER : dconv_3 shape: %s' % (str(X.shape)))
#
#     # Upsampling 4
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[3], decoding_filters[4]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dconv_6',
# add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_12')
#     # X = ops.activation(X, 'relu', 'dec_relu_6')
#     logging.info('DECODER : dconv_4 shape: %s' % (str(X.shape)))
#
#     # Decoding
#     # X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[4], decoding_filters[5]], stride=1, padding='SAME',
#     #                    w_init='gu', w_decay=None, scope_name='conv_7', add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_13')
#     # logging.info('DECODER : dconv_5 shape: %s' % (str(X.shape)))
#
#     sigmoid_logits = ops.activation(X, 'sigmoid', 'sigmoid_1')
#     # print(X.shape)
#
#     return X, sigmoid_logits
#
#
# def conv_autoencoder(img_shape, device_type=None):
#     inpX = tf.placeholder(dtype=tf.float32, shape=(None, img_shape[0], img_shape[1], img_shape[2]))
#     # enc_filter = [3, 16, 32, 32, 128, 256, 256]  # 3 Represents the number of channels
#     # dec_filter = [256, 256, 128, 64, 32, 16, 3]
#
#     enc_filter = [3, 64, 64, 64, 64]  # 3 Represents the number of channels
#     dec_filter = [64, 64, 64, 64, 3]
#
#     with tf.device(device_type):
#         z_encoded_mean, z_encoded_log_var = encoder(inpX, encoding_filters=enc_filter)
#         sampled_z = sample_gaussian_1var(z_encoded_mean, z_encoded_log_var, epsilon_sd=1.0)
#         _, x_reconstructed = decoder(sampled_z, dec_filter)
#
#         tot_ls, xent_ls, kl_ls = vae_loss(x=inpX, x_decoded_mean_squash=x_reconstructed, z_mean=z_encoded_mean,
#                                           z_log_var=z_encoded_log_var,
#                                           input_dimension=128 * 128)
#         opt, l_rate = ops.optimize(tot_ls, learning_rate_decay=False, add_smry=False)
#
#     return dict(inpX=inpX, latent_features=z_encoded_mean, reconstruction=x_reconstructed, loss=tot_ls,
#                     xent_loss=xent_ls, kl_loss=kl_ls, optimizer=opt, learning_rate=l_rate)
#







import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from conv_net.utils import Score
import matplotlib.pyplot as plt

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute, Lambda)
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Flatten
import keras.backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import metrics
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from copy import deepcopy
from conv_net import ops


def sample_gaussian_1var(z_mean, z_log_var, latent_dim, epsilon_std):
    ''' WE SAMPLE FROM A RANDOM UNIT GAUSSIAN :
        In order to estimate z (latent space) from x (input space), we need to sample z from p(z|x). Since we learn
        analytically by gradient updating. This means that the notation should be differentiable, which even means
        that the model is deterministic i.e a fized value of parameter to the network will always output the same
        output. Therefore, the only souce to add stocasticity (randomness) is via the input x.

        Denoising Autoencoder: In denoising Autoencoders we learn that adding random noise to the input image actually
        controls overfitting by trying to learn a reconstruction to map output (learned from noisy input) to the
        actual input. Variational Autoencoder: In this case we incorporate a probabilistic sampling node (
        random_normal) which makes the model stocastic. Here, epsilon id the randomness (p(epsilon)). The Encoding
        layer send to the decoder is just the addition of Mean and variance latent representation. Here,
        we just multiply the randomness to the variance
        latent representation and add it to the mean latent representation.
        And thus we bring Radomness to out model and the inference and generation become entirely differentiable and
        learnable.

        Why do we do unit Gaussian: Theoratically, the encodings can take any values, squashing the
        encodings with activation doesn't seem a right. Having a gaussian with unit variance is sort of normalizing
        and adding random noise to control overfitting. For example if we are learning mnist data, say digit 1 has a
        mean 5.2 and digit 9 has a mean 6.4. Now suppose, digit 1 comes in and the mean is found to be 6,
        then the network may say than its 9. By sampling method we learn the variance representation. So VAE method
        would say digit 1 = N(mean=5.2, sd = 1) and digit 9 = N(mean=6.4, sd=0.1). Using this phenomena the network
        would learn the right classification. IMAGE MIXTURES OF GAUSSIAN.
    '''
    
    # latent_dim = 32
    with tf.name_scope("sample_gaussian"):
        # epsilon = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=epsilon_sd, dtype=tf.float32)
        # sum_mean_std = z_mean + epsilon * tf.exp(z_log_var / 2)
        epsilon = tf.random_normal(tf.shape(z_log_var), name="epsilon")
    return z_mean + epsilon * tf.exp(z_log_var)
    # return sum_mean_std


def vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var, input_dimension):
    '''
    :param x:
    :param x_decoded_mean:
    :return:

    The loss si defined by two quantities:

    1. Reconstruction loss: Cna be binary_cross_entropy (sigmoid_cross_entropy) or mean_squared_error

    2. The regularized loss: kl_loss = The kl_loss can be thought as the relative entropy between the prior p(z) and
    the posterior distribution q(z). The posterior q(z).
    The formula seems a little weird because we strategically choose certain conjugate priors over z that let us
    analytically integrate the KL divergence, yielding a closed form equation. So,
    -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + log(z_var) - z_mean^2 -z_var), which can also be written as
    -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + z_var_log - z_mean^2 - exp(z_var_log))
    '''
    
    # Compute VAE loss
    # with tf.name_scope("binary_cross_entropy_loss"):
    #     xent_loss = input_dimension * metrics.binary_crossentropy(
    #             K.flatten(x),
    #             K.flatten(x_decoded_mean_squash))
    
    offset = 1e-7
    
    with tf.name_scope("cross_entropy"):
        # bound by clipping to avoid nan
        observed = tf.clip_by_value(x_decoded_mean_squash, offset, 1 - offset)
        xent_loss = -tf.reduce_sum(x * tf.log(observed) + (1 - x) * tf.log(1 - observed), 1)
        # b_xent_loss = input_dimension * ops.get_loss(y_true=x, y_logits=x_decoded_mean,
        # which_loss='sigmoid_cross_entropy', lamda=None)  # Entropy loss of input and
        # # b_xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
        # print('12121212121212', b_xent_loss.shape)
        #
        # b_xent_loss = input_dimension * metrics.binary_crossentropy(
        #         K.flatten(x),
        #         K.flatten(x_decoded_mean))
    with tf.name_scope("KL_divergence_loss"):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = -0.5 * tf.reduce_sum(1 + 2 * z_log_var - z_mean ** 2 - tf.exp(2 * z_log_var), 1)
    
    vae_loss = tf.reduce_mean(xent_loss + kl_loss, name='VAE_cost')
    
    # generated_flat = tf.reshape(self.generated_images, [self.batchsize, 224 * 224 * 3])
    # kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_log_var) - tf.log(tf.square(z_log_var)) - 1, 1)
    # vae_loss = K.mean(xent_loss + kl_loss)
    
    # The KL-divergence loss tries to bring the latent variables closer to a unit gaussian distribution. and the
    # binary_cross_entropy minimizes the reconstruction error. Former is trying to make things seems a bit random,
    # whereas the later is trying to learn the input properly
    # vae_loss = tf.reduce_mean(b_xent_loss + kl_loss, name="VAE_total_cost")
    
    return vae_loss, xent_loss, kl_loss


def encoder(input_img_arr, latent_dim):
    with tf.variable_scope("encoding"):
        c1 = Convolution2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(input_img_arr)
        print('c1.shape: ', c1.shape)
        c2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c1)
        print('c2.shape: ', c2.shape)
        c3 = Convolution2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c2)
        print('c3.shape: ', c3.shape)
        c4 = Convolution2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c3)
        print('c4.shape: ', c4.shape)
        f1 = Flatten()(c4)
        print('f1.shape: ', f1.shape)
        fc1 = Dense(8 * 8 * 32, activation='relu')(f1)
        print('fc1.shape: ', fc1.shape)
        fc2 = Dense(512, activation='relu')(fc1)
        print('fc2.shape: ', fc2.shape)
        z_mean = Dense(latent_dim)(fc2)
        print('z_mean.shape: ', z_mean.shape)
        z_log_var = Dense(latent_dim)(fc2)
        print('z_log_var.shape: ', z_log_var.shape)
    
    return z_mean, z_log_var


def decoder(sampled_z):
    reshape_dim = [8, 8, 32]
    with tf.variable_scope("decoding"):
        fc1 = Dense(512, activation='relu')(sampled_z)
        print('Decoder: fc1.shape: ', fc1.shape)
        fc2 = Dense(reshape_dim[0] * reshape_dim[1] * reshape_dim[2], activation='relu')(fc1)
        print('Decoder: fc2.shape: ', fc2.shape)
        rsp_imp = tf.reshape(fc2, [K.shape(sampled_z)[0], reshape_dim[0], reshape_dim[1], reshape_dim[2]])
        print('Decoder: rsp_imp.shape: ', rsp_imp.shape)
        dc1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(rsp_imp)
        print('Decoder: dc1.shape: ', dc1.shape)
        dc2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(dc1)
        print('Decoder: dc2.shape: ', dc2.shape)
        dc3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(dc2)
        print('Decoder: dc3.shape: ', dc3.shape)
        dc4 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, padding='same', activation='sigmoid')(dc3)
        print('Decoder: dc4.shape: ', dc4.shape)
    return dc4



def conv_autoencoder(img_shape, device_type=None):
    inpX = tf.placeholder(tf.float32, [None, img_shape[0], img_shape[1], img_shape[2]])
    # gradients = tf.placeholder(tf.float32, [None, 224,224,3])
    z_encoded_mean, z_encoded_log_var = encoder(inpX, latent_dim=64)
    guessed_z = sample_gaussian_1var(z_encoded_mean, z_encoded_log_var, latent_dim=64, epsilon_std=1.0)
    x_reconstructed = decoder(guessed_z)
    
    tot_ls, xent_ls, kl_ls = vae_loss(x=inpX, x_decoded_mean_squash=x_reconstructed, z_mean=z_encoded_mean,
                                      z_log_var=z_encoded_log_var,
                                      input_dimension=img_shape[0] * img_shape[1])
    opt, l_rate = ops.optimize(tot_ls, learning_rate_decay=False, add_smry=False)
    return dict(inpX=inpX, latent_features=z_encoded_mean, reconstruction=x_reconstructed, loss=tot_ls,
                xent_loss=xent_ls, kl_loss=kl_ls, optimizer=opt, learning_rate=l_rate)

