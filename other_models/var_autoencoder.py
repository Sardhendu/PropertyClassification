# coding=utf-8
# from keras import metrics
# import tensorflow as tf
# from conv_net import ops
#

# batch_size = 128
# feature_dim = 4096
# # layer_1 = 1024
# layer1_dim = 256
# latent_dim = 32
# epochs = 50
#
# from keras import backend as K
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
#         epsilon = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=epsilon_sd)
#         sum_mean_std = z_mean + epsilon * tf.exp(z_log_var/2)
#     return sum_mean_std

    # z_mean, z_log_var = args
    # print (K.shape(z_log_var), K.shape(z_mean))
    # epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1.0)
    # print (K.shape(epsilon))
    # stacked_mean_std = z_mean + epsilon* K.exp(z_log_var/2)
    # return stacked_mean_std

# #
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
#     # with tf.name_scope("binary_cross_entropy_loss"):
#     #     b_xent_loss = input_dimension * ops.get_loss(y_true=x, y_logits=x_decoded_mean, which_loss='sigmoid_cross_entropy', lamda=None)  # Entropy loss of input and
#     #     # # b_xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
#     #     # print('12121212121212', b_xent_loss.shape)
#     #     #
#     #     # b_xent_loss = input_dimension * metrics.binary_crossentropy(
#     #     #         K.flatten(x),
#     #     #         K.flatten(x_decoded_mean))
#     # with tf.name_scope("KL_divergence_loss"):
#     #     """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
#     #     # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
#     #     kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#
#     self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
#     xent_loss = input_dimension * metrics.binary_crossentropy(
#             K.flatten(x),
#             K.flatten(x_decoded_mean_squash))
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     vae_loss = K.mean(xent_loss + kl_loss)
#
#     # The KL-divergence loss tries to bring the latent variables closer to a unit gaussian distribution. and the
#     # binary_cross_entropy minimizes the reconstruction error. Former is trying to make things seems a bit random,
#     # whereas the later is trying to learn the input properly
#     # vae_loss = tf.reduce_mean(b_xent_loss + kl_loss, name="VAE_total_cost")
#
#     return vae_loss#, b_xent_loss, kl_loss

#
# #
# # def encoder(X, encoding_filters):
# #     # Case 1:
# #     X = ops.fc_layers(X, k_shape=[encoding_filters[0], encoding_filters[1]],
# #                       w_init='gu', scope_name='enc_fc_layer1', add_smry=False)
# #     X = ops.batch_norm(X, [0], 'enc_bn_1')
# #     print ('X1: ', X.shape)
# #     X = ops.activation(X, 'relu', 'enc_relu1')
# #
# #     # Case 2:
# #     X = ops.fc_layers(X, k_shape=[encoding_filters[1], encoding_filters[2]],
# #                       w_init='gu', scope_name='enc_fc_layer2', add_smry=False)
# #     X = ops.batch_norm(X, [0], 'enc_bn_2')
# #     print('X1: ', X.shape)
# #     X = ops.activation(X, 'relu', 'enc_relu2')
# #
# #
# #     z_mean = ops.fc_layers(X, k_shape=[encoding_filters[2], encoding_filters[3]],
# #                            w_init='gu', scope_name='z_mean_given_x', add_smry=False)
# #     print ('z_mean: ', z_mean.shape)
# #     z_log_var = ops.fc_layers(X, k_shape=[encoding_filters[2], encoding_filters[3]], w_init='gu', scope_name='z_log_var_given_x', add_smry=False)
# #     print ('z_log_var: ', z_log_var.shape)
# #
# #     z_encoder = sample_gaussian_1var(z_mean, z_log_var, 1.0)
# #     print ('latent encoder: ', z_encoder.shape)
# #     return z_mean, z_log_var, z_encoder
# #
# #
# #
# # def decoder(X, decoding_filters):
# #     X = ops.fc_layers(X, k_shape=[decoding_filters[0], decoding_filters[1]],
# #                       w_init='gu', scope_name='dec_fc_layer1', add_smry=False)
# #     X = ops.batch_norm(X, [0], 'dec_bn_1')
# #     X = ops.activation(X, 'relu', 'dec_relu1')
# #     print ('latent decoder: ', X.shape)
# #
# #     X = ops.fc_layers(X, k_shape=[decoding_filters[1], decoding_filters[2]],
# #                       w_init='gu', scope_name='dec_fc_layer2', add_smry=False)
# #     X = ops.batch_norm(X, [0], 'dec_bn_2')
# #     X = ops.activation(X, 'relu', 'dec_relu2')
# #     print('latent decoder: ', X.shape)
# #
# #     X_decoded_mean = ops.fc_layers(X, k_shape=[decoding_filters[2], decoding_filters[3]], w_init='gu', scope_name='x_mean_given_z', add_smry=False)
# #     X_decoded_mean = ops.activation(X_decoded_mean, 'sigmoid', 'sigmoid_1')
# #     print ('X_decoded_mean: ', X_decoded_mean.shape)
# #     return X_decoded_mean
# #
# #
# #
# # def variational_encoder():
# #
# #     inpXX = tf.placeholder(dtype=tf.float32, shape=[None, 64,64,3])#encoding_filters[0]])
# #     inpX = tf.layers.flatten(inpXX)
# #     features_in = inpX.get_shape().as_list()[-1]
# #     encoding_filters = [features_in, 256, 128, 32]  # 32 is the latent dimension
# #     decoding_filters = [32, 128, 256, features_in]
# #
# #
# #     z_encoded_mean, z_encoder_log_var, z_encoder = encoder(inpX, encoding_filters)
# #
# #     X_decoded_mean = decoder(z_encoder, decoding_filters)
# #     print('input_shape', inpX.shape)
# #
# #     loss = vae_loss(inpX, X_decoded_mean, z_mean=z_encoded_mean, z_log_var=z_encoder_log_var, input_dimension=encoding_filters[0])
# #     opt, l_rate = ops.optimize(loss, learning_rate_decay=False, add_smry=False)
# #
# #     return dict(inpX=inpXX, latent_features=z_encoded_mean, loss=loss, optimizer=opt, learning_rate=l_rate)
#
#
# def encoder(X, encoding_filters):
#     # Convolution 1:
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[0], encoding_filters[1]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='enc_conv_1', add_smry=False)
#     # X = ops.batch_norm(X, [0,1,2], 'enc_bn_1')
#     print('Encoder: X conv1: ', X.shape)
#     X = ops.activation(X, 'relu', 'enc_relu_1')
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enc_pool_1')
#     print('Encoder: X pool1: ', X.shape)
#
#     # Convolution 2: Note stride = 2 (downsamples by half)
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[1], encoding_filters[2]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='enc_conv_2', add_smry=False)
#     # X = ops.batch_norm(X, [0,1,2], 'enc_bn_2')
#     print('Encoder: X conv2: ', X.shape)
#     X = ops.activation(X, 'relu', 'enc_relu2')
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enc_pool_2')
#     print('Encoder: X pool2: ', X.shape)
#
#     # Convolution 3:
#     X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[2], encoding_filters[3]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='enc_conv_3', add_smry=False)
#     # X = ops.batch_norm(X, [0, 1, 2], 'enc_bn_3')
#     print('Encoder: X conv3: ', X.shape)
#     X = ops.activation(X, 'relu', 'enc_relu3')
#     X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enc_pool_3')
#     print('Encoder: X pool1: ', X.shape)
#
#     # # Convolution 4:
#     # X = ops.conv_layer(X, k_shape=[3, 3, encoding_filters[3], encoding_filters[4]], stride=1, padding='SAME',
#     #                    w_init='gu', w_decay=None, scope_name='enc_conv_4', add_smry=False)
#     # # X = ops.batch_norm(X, [0, 1, 2], 'enc_bn_4')
#     # print('Encoder: X conv4: ', X.shape)
#     # X = ops.activation(X, 'relu', 'enc_relu4')
#     # X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enc_pool_4')
#     # print('Encoder: X pool2: ', X.shape)
#     #
#
#     X = tf.layers.flatten(X)
#     print ('Encoder: Xflattened.shape: ', X.shape)
#
#     X = ops.fc_layers(X, k_shape=[X.get_shape().as_list()[-1], encoding_filters[5]],
#                            w_init='gu', scope_name='enc_hidden_unit', add_smry=False)
#     print('Encoder: hidden.shape: ', X.shape)
#
#     z_mean = ops.fc_layers(X, k_shape=[encoding_filters[5], encoding_filters[6]],
#                            w_init='gu', scope_name='z_mean_given_x', add_smry=False)
#     print('Encoder: z_mean: ', z_mean.shape)
#     z_log_var = ops.fc_layers(X, k_shape=[encoding_filters[5], encoding_filters[6]], w_init='gu',
#                               scope_name='z_log_var_given_x', add_smry=False)
#     print('Encoder: z_log_var: ', z_log_var.shape)
#
#     z_encoder = sample_gaussian_1var(z_mean, z_log_var, 1.0)
#     print('latent encoder: ', z_encoder.shape)
#     return z_mean, z_log_var, z_encoder
#
#
# def decoder(X, decoding_filters):
#     X = ops.fc_layers(X, k_shape=[decoding_filters[0], decoding_filters[1]], w_init='gu',
#                       scope_name='dec_hidden', add_smry=False)
#
#     print('Decoder: hidden.shape: ', X.shape)
#
#     X = ops.fc_layers(X, k_shape=[decoding_filters[1], decoding_filters[2]*8*8], w_init='gu',
#                       scope_name='dec_deflatten', add_smry=False)
#     print('Decoder: De-flatten.shape: ', X.shape)
#     X = tf.reshape(X, tf.stack([tf.shape(X)[0], 8, 8, 64]))
#     # multiplied by 8 because we want to de-flatten the input into image shape, also in encoder the last layer
#     # converted the input to [batch_size, 8, 8, 64]
#     print('Decoder: De-flatten.shape: ', X.shape)
#
#     # Fractionally Strided Convolution 1
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[2], decoding_filters[3]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dec_dconv_1',
#                                       add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     print('Decoder: Deconv 1: ', X.shape)
#     X = ops.activation(X, 'relu', 'dec_relu_1')
#
#     # Fractionally Strided Convolution 2
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[3], decoding_filters[4]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None, scope_name='dec_dconv_2',
#                                       add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     print('Decoder: Deconv 2: ', X.shape)
#     X = ops.activation(X, 'relu', 'dec_relu_2')
#
#     # Fractionally Strided Convolution 3
#     X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[4], decoding_filters[5]], stride=2,
#                                       padding='SAME', w_init='gu', out_shape=None,
#                                       scope_name='dev_dconv_3',
#                                       add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     print('Decoder: Deconv 3: ', X.shape)
#     X = ops.activation(X, 'relu', 'dec_relu_3')
#
#     # # Fractionally Strided Convolution 4
#     # X = ops.conv2D_transposed_strided(X, k_shape=[3, 3, decoding_filters[4], decoding_filters[5]], stride=2,
#     #                                   padding='SAME', w_init='gu', out_shape=None,
#     #                                   scope_name='dev_dconv_4',
#     #                                   add_smry=False)
#     # # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     # print('Decoder: Deconv 4: ', X.shape)
#     # X = ops.activation(X, 'relu', 'dec_relu_4')
#
#     # Fractionally Strided Convolution 3
#     X = ops.conv_layer(X, k_shape=[3, 3, decoding_filters[5], decoding_filters[6]], stride=1, padding='SAME',
#                        w_init='gu', w_decay=None, scope_name='dec_conv_4', add_smry=False)
#     # X = ops.batch_norm(X, axis=[0, 1, 2], scope_name='bn_7')
#     print('Decoder: Conv 1: ', X.shape)
#
#     # Squash the conv
#     X = ops.activation(X, 'sigmoid', 'dec_sigmoid_1')
#
#     return X
#
# def variational_encoder():
#     encoding_filters = [3, 64, 64, 64, 64, 128, 32]  # 32 is the latent dimension
#     decoding_filters = [32, 128, 64, 64, 64, 64, 3]
#
#     inpX = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])  # encoding_filters[0]])
#     # inpX = tf.layers.flatten(inpXX)
#     # features_in = inpX.get_shape().as_list()[-1]
#
#
#     z_encoded_mean, z_encoder_log_var, z_encoder = encoder(inpX, encoding_filters)
#     print ('z_encoded_mean.shape  ',  z_encoded_mean.shape)
#     print ('z_encoder_log_var.shape ', z_encoder_log_var.shape)
#
#
#     X_decoded_mean, _ = decoder(z_encoder, decoding_filters)
#     print('X_reconstructed_mean', X_decoded_mean.shape)
#
#     print (inpX.shape, X_decoded_mean.shape)
#     loss, b_xent_loss, kl_loss = vae_loss(inpX, X_decoded_mean, z_mean=z_encoded_mean, z_log_var=z_encoder_log_var,
#                     input_dimension=X_decoded_mean.get_shape().as_list()[1]*X_decoded_mean.get_shape().as_list()[2])
#     opt, l_rate = ops.optimize(loss, learning_rate_decay=False, add_smry=False)
#
#     return dict(inpX=inpX, latent_features=z_encoded_mean, loss=loss, b_xent_loss=b_xent_loss, kl_loss=kl_loss,
#                 optimizer=opt,
#                 learning_rate=l_rate)
#
# import numpy as np
# X = np.random.random((128,64,64,3))
# variational_encoder()




import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from conv_net.utils import Score
import matplotlib.pyplot as plt

from keras.layers import (Activation, Convolution2D,  Dense, Flatten, Input,
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









def sample_gaussian_1var(z_mean, z_log_var, epsilon_sd):
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
        epsilon = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=epsilon_sd, dtype=tf.float32)
        sum_mean_std = z_mean + epsilon * tf.exp(z_log_var/2)
    return sum_mean_std


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

    #Compute VAE loss
    with tf.name_scope("binary_cross_entropy_loss"):
        xent_loss = input_dimension * metrics.binary_crossentropy(
                K.flatten(x),
                K.flatten(x_decoded_mean_squash))
        # b_xent_loss = input_dimension * ops.get_loss(y_true=x, y_logits=x_decoded_mean, which_loss='sigmoid_cross_entropy', lamda=None)  # Entropy loss of input and
        # # b_xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
        # print('12121212121212', b_xent_loss.shape)
        #
        # b_xent_loss = input_dimension * metrics.binary_crossentropy(
        #         K.flatten(x),
        #         K.flatten(x_decoded_mean))
    with tf.name_scope("KL_divergence_loss"):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

    vae_loss = K.mean(xent_loss + kl_loss)
    

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
        c1 = Convolution2D(filters=16, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(input_img_arr)
        print ('c1.shape: ', c1.shape)
        c2 = Convolution2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c1)
        print('c2.shape: ', c2.shape)
        c3 = Convolution2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c2)
        print('c3.shape: ', c3.shape)
        c4 = Convolution2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(c3)
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
    reshape_dim = [8,8,32]
    with tf.variable_scope("decoding"):
        fc1=Dense(512, activation='relu')(sampled_z)
        print('Decoder: fc1.shape: ', fc1.shape)
        fc2=Dense(reshape_dim[0]*reshape_dim[1]*reshape_dim[2], activation='relu')(fc1)
        print('Decoder: fc2.shape: ', fc2.shape)
        rsp_imp = tf.reshape(fc2, [K.shape(sampled_z)[0], reshape_dim[0], reshape_dim[1], reshape_dim[2]])
        print('Decoder: rsp_imp.shape: ', rsp_imp.shape)
        dc1 = Conv2DTranspose(filters=64,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(rsp_imp)
        print('Decoder: dc1.shape: ', dc1.shape)
        dc2 = Conv2DTranspose(filters=32 ,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(dc1)
        print('Decoder: dc2.shape: ', dc2.shape)
        dc3 = Conv2DTranspose(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(dc2)
        print('Decoder: dc3.shape: ', dc3.shape)
        dc4 = Conv2DTranspose(filters=3,kernel_size=(3, 3),strides=2, padding = 'same',activation='sigmoid')(dc3)
        print('Decoder: dc4.shape: ', dc4.shape)
    return dc4


def variational_encoder():
    inpX = tf.placeholder(tf.float32, [None, 128,128,3])
    # gradients = tf.placeholder(tf.float32, [None, 224,224,3])
    z_encoded_mean, z_encoded_log_var = encoder(inpX, latent_dim=64)
    sampled_z = sample_gaussian_1var(z_encoded_mean, z_encoded_log_var, epsilon_sd=1.0)
    x_reconstructed = decoder(sampled_z)
    tot_ls, xent_ls, kl_ls = vae_loss(x=inpX, x_decoded_mean_squash=x_reconstructed, z_mean=z_encoded_mean,
                         z_log_var=z_encoded_log_var,
                        input_dimension=128*128)
    opt, l_rate = ops.optimize(tot_ls, learning_rate_decay=False, add_smry=False)
    return dict(inpX=inpX, latent_features=z_encoded_mean, reconstruction=x_reconstructed, loss=tot_ls, xent_loss=xent_ls, kl_loss=kl_ls, optimizer=opt, learning_rate=l_rate)


##########################################
##########################################
##########################################



from config import pathDict
from data_transformation.data_io import getH5File
import seaborn as sns
from data_transformation.preprocessing import Preprocessing

sns.set_style("whitegrid")

out_img_shape = [128, 128, 3]


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

def plot(X_true, X_reconstructed):
    n = 20
    fig, ax = plt.subplots(4, n//2, figsize=(40, 8))
    ax = ax.ravel()

    imaga_indexlist = np.arange(54, 74)
    for i in range(0, n):
        ax[i].imshow(X_true[imaga_indexlist[i]])

    for i in range(n, n + n):
        ax[i].imshow(X_reconstructed[imaga_indexlist[i - n]])

    fig.show()
    plt.pause(3)
    plt.close()
    
    
computation_graph = variational_encoder()
preprocess_graph = Preprocessing(inp_img_shape=[250,300,3],
                                 crop_shape=[196,128,3],
                                 out_img_shape=[128,128,3]).preprocessImageGraph()

trainX = []
trainY = []
batch_size = 15
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    cvalidX, cvalidY = load_batch_data(image_type='assessor_code', which_data='cvalid')
    print('cvalid_shape: ', cvalidX.shape)
    cvpreprocessed_data = run_preprocessor(sess, cvalidX, preprocess_graph, is_training=False)
    epochs = 30
    for epoch in range(0,epochs):
        for batch_num in range(0, batch_size):
            print('running batch  number: ', batch_num)
            batchX, batchY = load_batch_data(image_type='assessor_code', which_data='train_%s' % (batch_num))
            print ('batchX.shape, batchY.shape ', batchX.shape, batchY.shape)
            preprocessed_data = run_preprocessor(sess, batchX, preprocess_graph, is_training=True)
            print('preprocessed_data.shape ', preprocessed_data.shape)
            
        #     if batch_num == 0:
        #         trainX = preprocessed_data
        #         trainY = batchY
        #     else:
        #         trainX = np.vstack((trainX, preprocessed_data))
        #         trainY = np.append(trainY, batchY)
        #
        # print(trainX.shape, trainY.shape, cvpreprocessed_data.shape)
    
    
    
        # for batch_num in range(0, 20):
        #     from_idx = batch_num*100
        #     to_idx = batch_num*100 + (batch_num+1)*100
        #
        #     dataX = trainX[from_idx:to_idx,:]
        #     dataY = trainY[from_idx:to_idx]
            feed_dict = {computation_graph['inpX']: preprocessed_data}
            ls, opt, rec_x = sess.run([computation_graph['loss'], computation_graph['optimizer'], computation_graph[
                'reconstruction']], feed_dict=feed_dict)
            
            print('Batch_loss : ', ls)
            plot(X_true=preprocessed_data, X_reconstructed=rec_x)
    
        feed_dict = {computation_graph['inpX']: cvpreprocessed_data}
        ls, out, cvrec = sess.run([computation_graph['loss'], computation_graph['latent_features'], computation_graph[
                'reconstruction']], feed_dict=feed_dict)
        print ('Cross validation loss = %s, out.shape = '%str(ls), out.shape)
        
        for i in range(0,2):
            kmeans = KMeans(n_clusters=2, n_init=100, random_state=443)
            kmeans = kmeans.fit(out)
            labels = kmeans.predict(out)
            centroids = kmeans.cluster_centers_
            #                 print (labels)
            #                 labels[labels == 0] = 6
            #                 labels[labels == 1] = 8
        
            
        
            scr = Score.accuracy(cvalidY, labels)
            print ('i = %s Score = '%str(i), scr)
        
    # feed_dict = {computation_graph['inpX']: cvpreprocessed_data}
    # cv_dec, sig_logits, cv_ls = sess.run([computation_graph['decoded'],
    #                               computation_graph['sigmoid_logits'],
    #                               computation_graph['loss']], feed_dict=feed_dict)









# return dict(inpX=inpX, latent_features=z_encoded_mean, loss=tot_ls, xent_loss=xent_ls, kl_loss=kl_ls,
#                 optimizer=opt, learning_rate=l_rate)


        
# #
# vae.fit(trainX,
#         shuffle=True,
#         epochs=20,
#         batch_size=batch_size,
#         validation_data=(cvpreprocessed_data, None))
#
# encoder = Model(x, z_mean)
# x_test_encoded = encoder.predict(cvpreprocessed_data, batch_size=batch_size)
#
