from __future__ import division, print_function, absolute_import
import logging

import os
import numpy as np
from config import pathDict
from data_transformation.data_prep import genStratifiedBatches

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")



batch_prepare = False
run_model = False


def get_valid_land_house_ids(images_per_label=None):
    aerial_land_pins = np.array([img.split('.')[0]
                                 for img in os.listdir(os.path.join(pathDict['bing_aerial_image_path'], 'land')) if img !=".DS_Store"], dtype=str)

    aerial_house_pins = np.array([img.split('.')[0]
                                for img in os.listdir(os.path.join(pathDict['bing_aerial_image_path'], 'house')) if
                                  img !=".DS_Store"], dtype=str)

    assessor_land_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'land')) if img !=".DS_Store"], dtype=str)

    assessor_house_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'house')) if img !=".DS_Store"], dtype=str)

    streetside_land_pins = np.array([img.split('.')[0] for img in os.listdir(
            os.path.join(pathDict['streetside_image_path'], 'land')) if img !=".DS_Store"], dtype=str)

    streetside_house_pins = np.array([img.split('.')[0] for img in os.listdir(
            os.path.join(pathDict['streetside_image_path'], 'house')) if img != ".DS_Store"], dtype=str)
    
    cmn_land_pins = np.intersect1d(aerial_land_pins, assessor_land_pins)
    cmn_land_pins = np.intersect1d(cmn_land_pins, streetside_land_pins)
    
    cmn_house_pins = np.intersect1d(aerial_house_pins, assessor_house_pins)
    cmn_house_pins = np.intersect1d(cmn_house_pins, streetside_house_pins)
    
    np.random.seed(184)
    np.random.shuffle(cmn_land_pins)
    np.random.shuffle(cmn_house_pins)
    
    if not images_per_label:
        images_per_label = min(len(cmn_land_pins), len(cmn_house_pins))
    
    return cmn_land_pins[0:images_per_label], cmn_house_pins[0:images_per_label]



    
if batch_prepare:
    valid_land_pins, valid_house_pins = get_valid_land_house_ids(images_per_label=None)
    print (len(valid_land_pins), len(valid_house_pins))

    # genStratifiedBatches([224, 224, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='aerial', dump=True, shuffle_seed=874)
    #
    # genStratifiedBatches([260, 260, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='assessor', dump=True, shuffle_seed=874)
    #
    # genStratifiedBatches([260, 260, 3], valid_land_pins, valid_house_pins, cv_batch_size=500, tr_batch_size=64, image_type='streetside', dump=True, shuffle_seed=874)

# if run_model:

