from __future__ import division, print_function, absolute_import
import logging

import os
import numpy as np
from config import pathDict
from conv_net.run import Train, Test
from data_transformation.data_prep import genStratifiedBatches


logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_valid_land_house_ids(aerial_img_type=None, streetside_img_type=None, overlayed_img_type=None,
                             images_per_label=None):
    cmn_land_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'land')) if
             img != ".DS_Store"], dtype=str)

    cmn_house_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'house')) if
             img != ".DS_Store"], dtype=str)
 
    
    if aerial_img_type:
        print(os.path.join(pathDict['%s_image_path' % (aerial_img_type)], 'land'))
        aerial_land_pins = np.array(
                [img.split('.')[0]
                for img in os.listdir(os.path.join(pathDict['%s_image_path'%(aerial_img_type)], 'land'))
                 if img !=".DS_Store"], dtype=str)
    
        aerial_house_pins = np.array(
                [img.split('.')[0]
                for img in os.listdir(os.path.join(pathDict['%s_image_path'%(aerial_img_type)], 'house'))
                 if img !=".DS_Store"], dtype=str)
        print ('aerial pins: ', len(aerial_land_pins), len(aerial_house_pins))
        cmn_land_pins = np.intersect1d(cmn_land_pins, aerial_land_pins)
        cmn_house_pins = np.intersect1d(cmn_house_pins, aerial_house_pins)

        print('common aerial pins: ', len(cmn_land_pins), len(cmn_house_pins))
        
    if streetside_img_type:
        print(os.path.join(pathDict['%s_image_path' % (streetside_img_type)], 'land'))
        streetside_land_pins = np.array([img.split('.')[0] for img in os.listdir(
                os.path.join(pathDict['%s_image_path'%(streetside_img_type)], 'land')) if img !=".DS_Store"], dtype=str)
    
        streetside_house_pins = np.array([img.split('.')[0] for img in os.listdir(
                os.path.join(pathDict['%s_image_path'%(streetside_img_type)], 'house')) if img != ".DS_Store"], dtype=str)

        print('streetside pins: ', len(streetside_land_pins), len(streetside_house_pins))
        cmn_land_pins = np.intersect1d(cmn_land_pins, streetside_land_pins)
        cmn_house_pins = np.intersect1d(cmn_house_pins, streetside_house_pins)

        print('common streetside pins: ', len(cmn_land_pins), len(cmn_house_pins))
    
    if overlayed_img_type:
        print (os.path.join(pathDict['%s_image_path'%(overlayed_img_type)], 'land'))
        overlayed_land_pins = np.array([img.split('.')[0] for img in os.listdir(
                os.path.join(pathDict['%s_image_path'%(overlayed_img_type)], 'land')) if img != ".DS_Store"], dtype=str)
    
        overlayed_house_pins = np.array([img.split('.')[0] for img in os.listdir(
                os.path.join(pathDict['%s_image_path'%(overlayed_img_type)], 'house')) if img != ".DS_Store"], dtype=str)

        print('overlayed pins: ', len(overlayed_land_pins), len(overlayed_house_pins))
        cmn_land_pins = np.intersect1d(cmn_land_pins, overlayed_land_pins)
        cmn_house_pins = np.intersect1d(cmn_house_pins, overlayed_house_pins)

        print('common overlayed pins: ', len(cmn_land_pins), len(cmn_house_pins))
        

    np.random.seed(184)
    np.random.shuffle(cmn_land_pins)
    np.random.shuffle(cmn_house_pins)

    if not images_per_label:
        images_per_label = min(len(cmn_land_pins), len(cmn_house_pins))

    return cmn_land_pins[0:images_per_label], cmn_house_pins[0:images_per_label]





images_per_label = None # normally 5000 each label is good
assessor_img_type = 'assessor'
aerial_img_type = 'google_aerial' # 'bing_aerial'
overlayed_img_type = 'google_overlayed'
streetside_img_type = None
# image_shape = [224,224,3]
# inp_image_shape = [260, 260, 3]
inp_image_shape = [400, 400, 3]

image_type = overlayed_img_type#aerial_img_type#assessor_img_type



batch_prepare = False
train = False
test = True


if batch_prepare:
    valid_land_pins, valid_house_pins = get_valid_land_house_ids(
            aerial_img_type=aerial_img_type,
            streetside_img_type=streetside_img_type,
            overlayed_img_type=overlayed_img_type,
            images_per_label=images_per_label)
    print (len(valid_land_pins), len(valid_house_pins))


    if image_type == 'assessor':
        genStratifiedBatches([260, 260, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='assessor', dump=True, shuffle_seed=874)

    elif image_type == 'google_aerial':
        genStratifiedBatches([400, 400, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='google_aerial', dump=True, shuffle_seed=874)

    elif image_type == 'google_overlayed':
        genStratifiedBatches([400, 400, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64,image_type='google_overlayed', dump=True, shuffle_seed=874)
    elif image_type == 'google_streetside':
        genStratifiedBatches([260, 260, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='streetside', dump=True, shuffle_seed=874)
    
    else:
        raise ValueError('Not a valid image type provided')




if train:
    Train(dict(use_checkpoint=True,
               save_checkpoint=True,
               write_tensorboard_summary=True
               ),
          which_net='resnet',  # vgg
          image_type=image_type,
          inp_image_shape = inp_image_shape).run(num_epochs=3,
                                     num_batches=160)
if test:
    Test(params={}, which_net='resnet',
         image_type=image_type,
         inp_image_shape = inp_image_shape).run(dump_stats=True)