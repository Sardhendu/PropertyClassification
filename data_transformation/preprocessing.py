from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf
from scipy import misc

import config
from config import myNet


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
    
    def __init__(self):
        pass
    
    def randomCrop(self, imageIN):
        logging.info('Performing random crop')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        return tf.random_crop(imageIN, config.myNet['crop_shape'],
                              seed=config.seed_arr[config.preprocess_seed_idx])
    
    def centralCrop(self, imageIN):
        logging.info('Performing Central crop')
        return tf.random_crop(imageIN, config.myNet['crop_shape'])
    
    def randomFlip(self, imageIN):
        logging.info('Performing random horizontal flip')
        if config.preprocess_seed_idx == len(config.seed_arr) - 1:
            config.preprocess_seed_idx = 0
        # Given an image this operation may or may not flip the image
        return tf.image.random_flip_left_right(imageIN,
                                               seed=config.seed_arr[config.preprocess_seed_idx])
    
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
        return tf.image.per_image_standardization(imageIN)
    
    def preprocessImageGraph(self, imageShape):
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
        logging.info('PREPROCESSING THE DATASET ..........')
        imageIN = tf.placeholder(dtype=tf.float32,
                                 shape=[imageShape[0], imageShape[1], imageShape[2]],
                                 name="Preprocessor-variableHolder")
        
        # Add random contrast
        imageOUT = imageIN
        if config.pp_vars['rand_brightness']:
            imageOUT = self.addRandBrightness(imageOUT)
        
        if config.pp_vars['rand_contrast']:
            imageOUT = self.addRandContrast(imageOUT)
        
        if config.pp_vars['rand_rotate']:
            imageOUT = self.randomRotate(imageOUT)
        
        if config.pp_vars['rand_crop']:
            imageOUT = self.randomCrop(imageOUT)
        elif config.pp_vars['central_crop']:
            imageOUT = self.centralCrop(imageOUT)
        else:
            imageOUT = tf.image.resize_image_with_crop_or_pad(
                    imageOUT, myNet['crop_shape'][0],
                    myNet['crop_shape'][1]
            )
        
        if config.pp_vars['rand_flip']:
            imageOUT = self.randomFlip(imageOUT)
        
        if config.pp_vars['standardise']:
            imageOUT = self.standardize(imageOUT)
        
        return dict(imageIN=imageIN, imageOUT=imageOUT)