from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf
from scipy import misc

import config
from config import myNet


def zero_pad(inp, crop_shape, out_shape):
    '''
    :param inp:
    :param out_shape:
    :return:

    One image at a time
    '''
    m, n, c = crop_shape#inp.get_shape().as_list()
    out_m, out_n, out_c = out_shape
    
    to_pad_m = max(out_m - m, 0)
    to_pad_n = max(out_n - n, 0)
    to_pad_c = max(out_c - c, 0)
    
    pad_m1 = to_pad_m // 2
    pad_m2 = to_pad_m - pad_m1
    
    pad_n1 = to_pad_n // 2
    pad_n2 = to_pad_n - pad_n1
    
    pad_c1 = to_pad_c // 2
    pad_c2 = to_pad_c - pad_c1
    
    paddings = tf.constant([[pad_m1, pad_m2], [pad_n1, pad_n2], [pad_c1, pad_c2]])
    inp = tf.pad(inp, paddings, 'CONSTANT')
    logging.info('Image shape after Zero Padding crop: %s', str(inp.shape))
    return inp


def random_rotate_image(image):
    if config.preprocess_seed_idx == len(config.seed_arr) - 1:
        config.preprocess_seed_idx = 0
    
    np.random.seed(config.preprocess_seed_idx)
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


class Preprocessing():
    '''
        Preprocessing in images are done per image, hence it is a good idea to create a separate computation graph
        for Preprocessing such that the graph is iteratively fed the input image one after another pertaining to a
        batch.
    '''
    
    def __init__(self, pprocessor_inp_img_shape, pprocessor_inp_crop_shape, model_inp_img_shape):
        '''
        :param model_inp_img_shape: If the output cropped shape is smaller than the required image shape, then we pad the
        cropped image shape with 0's to make it equall to the output image shape
        '''
        # self.pprocessor_inp_img_shape =  pprocessor_inp_img_shape
        # self.model_inp_img_shape = model_inp_img_shape
        # self.crop_shape = crop_shape
        #
        self.pprocessor_inp_img_shape = pprocessor_inp_img_shape
        self.model_inp_img_shape = model_inp_img_shape

        if pprocessor_inp_crop_shape is not None:
            self.pprocessor_inp_crop_shape = pprocessor_inp_crop_shape
        else:
            self.pprocessor_inp_crop_shape = []

        
        
    
    def randomCrop(self, imageIN):
        logging.info('Performing random crop')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        return tf.random_crop(imageIN, self.pprocessor_inp_crop_shape,
                              seed=config.seed_arr[config.preprocess_seed_idx])
    
    def centralCrop(self, imageIN):
        logging.info('Performing Central crop')
        return tf.image.central_crop(imageIN, self.pprocessor_inp_crop_shape[0]/self.pprocessor_inp_img_shape[0])#tf.random_crop(imageIN, self.crop_shape)
    
    def randomHFlip(self, imageIN):
        logging.info('Performing random horizontal flip')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        # Given an image this operation may or may not flip the image
        return tf.image.random_flip_left_right(imageIN, seed=config.seed_arr[config.preprocess_seed_idx])
    
    def randomVFlip(self, imageIN):
        logging.info('Performing random Vertical flip')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        # Given an image this operation may or may not flip the image
        return tf.image.random_flip_up_down(imageIN, seed=config.seed_arr[config.preprocess_seed_idx])
    
    def randomRotate(self, imageIN):
        logging.info('Performing Random Rotation')
        # Given an image this operation may or may not flip the image
        return tf.py_func(random_rotate_image, [imageIN], tf.uint8)
    
    def addRandBrightness(self, imageIN):
        logging.info('Adding random brightness')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        # Add random brightness
        return tf.image.random_brightness(imageIN, max_delta=63,
                                          seed=config.seed_arr[config.preprocess_seed_idx])
    
    def addRandContrast(self, imageIN):
        logging.info('Adding random Contrast')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        return tf.image.random_contrast(imageIN, lower=0.2, upper=1.8,
                                        seed=config.seed_arr[config.preprocess_seed_idx])
    
    def standardize(self, imageIN):
        logging.info('Standarizing the image')
        return tf.divide(imageIN,  255.0)
        # return tf.image.per_image_standardization(imageIN)
    
    
    def preprocess_for_train(self, img):
        '''
        :param img: The image as an input
        :return:
        '''
        logging.info('PREPROCESSING config: With the training Data Set')
        imageOUT = img
        if config.pp_vars['rand_brightness']:
            imageOUT = self.addRandBrightness(img)
    
        if config.pp_vars['rand_contrast']:
            imageOUT = self.addRandContrast(imageOUT)
    
        if config.pp_vars['rand_rotate']:
            imageOUT = self.randomRotate(imageOUT)
    
        if config.pp_vars['rand_Hflip']:
            imageOUT = self.randomHFlip(imageOUT)
            
        if config.pp_vars['rand_Vflip']:
            imageOUT = self.randomVFlip(imageOUT)
            
        if config.pp_vars['rand_crop']:
            imageOUT = self.randomCrop(imageOUT)
            logging.info('Image shape after random crop: %s', str(imageOUT.shape))
            
        if config.pp_vars['central_crop']:
            imageOUT = self.centralCrop(imageOUT)
            logging.info('Image shape after central crop: %s', str(imageOUT.shape))
       
            
        if config.pp_vars['standardise']:
            imageOUT = self.standardize(imageOUT)
            
        return imageOUT

    def preprocess_for_test(self, img):
        logging.info('PREPROCESSING config: With the Test Data Set')
        imageOUT = img
        if config.pp_vars['central_crop']:
            imageOUT = self.centralCrop(imageOUT)
            logging.info('Image shape after central crop: %s', str(imageOUT.shape))
    
        if config.pp_vars['standardise']:
            imageOUT = self.standardize(imageOUT)
            
        return imageOUT
    
    
    def preprocessImageGraph(self, is_training):
        """
        :param imageSize:   The size of image
        :param numChannels: The number of channels
        :return:  The distorted image
        """
        '''
            Normally the inputs have dtype unit8 (0-255), We however take the input as float32 because we perform
            operations like brightness, contrast and whitening that doest arithmatic operation which many make the
            pixels value as floating point.
        '''
        # print (self.pprocessor_inp_img_shape[0], self.pprocessor_inp_img_shape[1], self.pprocessor_inp_img_shape[2])
        
        logging.info('PREPROCESSING THE DATASET of shape %s..........', str(self.pprocessor_inp_img_shape))
        imageIN = tf.placeholder(dtype=tf.float32,
                                 shape=[self.pprocessor_inp_img_shape[0], self.pprocessor_inp_img_shape[1], self.pprocessor_inp_img_shape[2]],
                                 name="Preprocessor-variableHolder")
        # is_training = tf.placeholder(tf.bool)
        
        # Add random contrast
        imageOUT = imageIN
        
        # imageOUT = tf.cond(is_training, lambda: self.preprocess_for_train(imageOUT),
        #                    lambda : self.preprocess_for_test(imageOUT))
        
        if is_training:
            imageOUT = self.preprocess_for_train(imageOUT)
        else:
            imageOUT = self.preprocess_for_test(imageOUT)
        
        # If the out_image_size is larger than the crop_image_size, then we pad the image with zeros to make it of out_image shape,
        # If the out_image_size is smaller than the crop_image_sze, then we resize the image to the out_image_size
        if len(self.pprocessor_inp_crop_shape)>0:
            if self.pprocessor_inp_crop_shape[0] - self.model_inp_img_shape[0] < 0:
                imageOUT = zero_pad(inp=imageOUT, crop_shape=self.pprocessor_inp_crop_shape, out_shape=self.model_inp_img_shape)
                
            else:
                imageOUT = tf.image.resize_images(imageOUT, size=tf.stack([self.model_inp_img_shape[0], self.model_inp_img_shape[1]]))
        elif self.model_inp_img_shape[0] != self.pprocessor_inp_img_shape[0]:
            imageOUT = tf.image.resize_images(imageOUT, size=tf.stack([self.model_inp_img_shape[0],self.model_inp_img_shape[1]]))
        else:
            pass

        return dict(imageIN=imageIN, imageOUT=imageOUT, is_training=is_training)
    
  
  
debugg = False
if debugg:
    # X = np.random.random((100,100))
    # X = tf.random_normal(shape=(100,100,3))
    # X = zero_pad(X, out_shape = [200,200,3])
    # print (X.shape)
    # X = np.random.random((100,100))
    obj = Preprocessing(pprocessor_inp_img_shape=[400,400,3], pprocessor_inp_crop_shape=[100,100,3], model_inp_img_shape=[200,200,3])
    Xout_graph = obj.preprocessImageGraph()
    print (Xout_graph['imageOUT'].get_shape().as_list())

