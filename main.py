
from __future__ import division, print_function, absolute_import
import logging

import os
import numpy as np
from config import pathDict
from data_transformation.data_prep import genStratifiedBatches

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")



batch_prepare = True
run_model = True


def get_valid_land_house_ids():
    aerial_land_pins = np.array([img.split('.')[0] for img in os.listdir(os.path.join(pathDict['aerial_image_path'], 'land')) if img !=".DS_Store"], dtype=str)

    aerial_house_pins = np.array([img.split('.')[0] for img in os.listdir(os.path.join(pathDict['aerial_image_path'], 'house')) if img !=".DS_Store"], dtype=str)

    assessor_land_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'land')) if img !=".DS_Store"], dtype=str)

    assessor_house_pins = np.array(
            [img.split('.')[0] for img in os.listdir(os.path.join(pathDict['assessor_image_path'], 'house')) if img !=".DS_Store"], dtype=str)
    
    cms_land_pins = np.intersect1d(aerial_land_pins, assessor_land_pins)
    cmn_house_pins = np.intersect1d(aerial_house_pins, assessor_house_pins)

    return cms_land_pins, cmn_house_pins



    
if batch_prepare:
    valid_land_pins, valid_house_ids = get_valid_land_house_ids()
    # print(valid_land_pins, '\n', valid_house_ids)

    genStratifiedBatches([224, 224, 3], valid_land_pins, valid_house_ids, cv_batch_size=200, tr_batch_size=64, image_type='aerial', dump=True)
    genStratifiedBatches([260, 260, 3], valid_land_pins, valid_house_ids, cv_batch_size=200, tr_batch_size=64, image_type='assessor', dump=True)

# if run_model:

