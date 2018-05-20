import logging
import shutil
from src.config import get_config
from src.data_transformation.data_prep import get_intersecting_images_pin, DumpBatches



def prepare_batches(**kwargs):
    inp_params = kwargs['params']
    max_batches = None

    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    if 'batch_size' not in inp_params.keys():
        raise ValueError('You should mention batch_size')
    
    if 'proportion_cv_data' not in inp_params.keys():
        raise ValueError('You should provide the proportion of validation data')
    
    if 'proportion_test_data' not in inp_params.keys():
        raise ValueError('You should provide the proportion of test data')
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    which_run = inp_params['which_run']
    is_cvalid_test = inp_params['is_cvalid_test']
    
    is_aerial = False
    is_overlaid = False
    is_aerial_cropped = False
    
    if inp_params['img_type'] == 'aerial':
        is_aerial = True
    if inp_params['img_type'] == 'aerial_cropped':
        is_aerial_cropped = True
    elif inp_params['img_type'] == 'overlaid':
        is_overlaid = True
    else:
        pass

    conf = get_config(which_run=which_run, img_type=inp_params['img_type'])
    
    cmn_land_pins, cmn_house_pins = get_intersecting_images_pin(
            conf, is_assessor=False, is_aerial=is_aerial, is_streetside=False,
            is_overlaid=is_overlaid, is_aerial_cropped=is_aerial_cropped,
            equal_proportion=True
    )
    
    logging.info('Shape: cmn_land_pins = %s, cmn_house_pins = %s ', str(len(cmn_land_pins)), str(len(cmn_house_pins)))
    
    tr_batch_size = inp_params['batch_size']
    ts_batch_size = (len(cmn_land_pins) + len(cmn_house_pins) ) * inp_params['proportion_test_data']
    cv_batch_size = (len(cmn_land_pins) + len(cmn_house_pins) ) * inp_params['proportion_cv_data']
    
    params = dict(
            image_type=inp_params['img_type'],
            img_in_shape=[400, 400, 3],
            img_out_shape=[224, 224, 3],
            img_resize_shape=[128, 128, 3],
            img_crop_shape=[128, 128, 3],
            tr_batch_size = tr_batch_size,
            cv_batch_size = cv_batch_size,
            ts_batch_size = ts_batch_size,
            enable_rotation=True,
            shuffle_seed=881,
            get_stats=True,
            max_batches=max_batches)
    
    obj_cb = DumpBatches(conf, params)
    obj_cb.dumpStratifiedBatches_balanced_class(cmn_land_pins, cmn_house_pins, is_cvalid_test=is_cvalid_test)
    return 'COMPLETE'


def remove_batches(**kwargs):
    inp_params = kwargs['params']
    logging.info('Removing Batches for RUN: %s, img_type: %s',
                 str(inp_params['which_run']), str(inp_params['img_type']))
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])

    batch_path = conf['pathDict']['batch_path']

    shutil.rmtree(batch_path)
    logging.info('Removed Batches ...................')
    return 'COMPLETE'