import logging
import os

import numpy as np
import pandas as pd
from scipy import  ndimage

from config import pathDict, myNet, fileNames, vars
from data_transformation.data_io import dumpPickleFile, getPickleFile


#### deprecated
def get_resized_image_arr(image_type, nw_shape, get_stats=False, dump=True):
    if image_type not in ['assessor', 'aerial']:
        raise ValueError('Variable image_type not understood')
    
    image_folder_path = pathDict['assessor_image_path']
    dump_folder_path = pathDict[ '%s_rsized_path' %(image_type)]
    stats_path = pathDict[ '%s_dp_stats_path' %(image_type)]
    logging.info('Fetching images from: %s', str(image_folder_path))
    logging.info('Dumping reshaped images to: %s', str(dump_folder_path))
    
    if not os.path.exists(image_folder_path):
        raise ValueError('The images are not in place')
    
    image_types = [image_type for image_type in os.listdir(image_folder_path)
                   if image_type != ".DS_Store"]
    
    # print (image_types)
    
    dataX = []
    dataY = []
    label_dict = {}
    pic_filename_arr = []
    label_name_arr = []
    label_idx_arr = []
    for label, image_type in enumerate(image_types):
        print (label, image_type)
        image_path = os.path.join(image_folder_path, image_type)
        print(image_path)
        
        image_paths = [os.path.join(image_path, pics)
                       for pics in os.listdir(image_path)
                       if pics.split('.')[1] == "jpg" or pics.split('.')[1] == "jpeg" or pics.split('.')[1] == "png"]
        print (image_paths)
        # break
        label_dict[str(label)] = image_type
        image_ndarr = np.ndarray((len(image_paths), myNet['image_shape'][0],
                                  myNet['image_shape'][1], myNet['image_shape'][2]))
        label_ndarr = np.tile(label, len(image_paths)).reshape(-1, 1)
        
        for img_num, pic_path in enumerate(image_paths):
            pic_filename_arr += [os.path.basename(pic_path).split('.')[0]]
            image = ndimage.imread(pic_path, mode='RGB')
            rsized_image = resize_image(image, resize_shape=nw_shape)
            image_ndarr[img_num, :] = rsized_image
        
        label_name_arr += [image_type] * len(image_paths)
        label_idx_arr += [str(label)] * len(image_paths)
        
        if label == 0:
            dataX = image_ndarr
            dataY = label_ndarr
        else:
            dataX = np.vstack((dataX, image_ndarr))
            dataY = np.vstack((dataY, label_ndarr))
    
    print (dataX.shape, dataY.shape)
    
    image_type_image_num_info = []
    if get_stats:
        image_type_image_num_info = np.column_stack((
            np.array(label_idx_arr).reshape(-1, 1),
            np.array(label_name_arr).reshape(-1, 1),
            np.array(pic_filename_arr).reshape(-1, 1)
        ))
        image_type_image_num_info = pd.DataFrame(
                image_type_image_num_info,
                columns=['image_label',
                         'person_name',
                         'file_name'])
        image_type_image_num_info = image_type_image_num_info.reset_index()
    
    if dump:
        dumpPickleFile(dataX=dataX, dataY=dataY, labelDict=label_dict, folderPath=dump_folder_path,
                       picklefileName=fileNames['rsized_img_file'])
        if get_stats:
            image_type_image_num_info.to_csv(stats_path)
    
    return dataX, dataY, label_dict, image_type_image_num_info


#### deprecated
def genRandomStratifiedBatchesOLD(dataX=[], dataY=[], label_dict={}, dump=True):
    if len(dataX) == 0:
        dataX, dataY, label_dict = getPickleFile(pathDict['input_rsized_image_path'], fileNames['rsized_img_file'])
    
    if not isinstance(dataX, np.ndarray):
        raise ValueError('Unhandled type dataX input')
    
    if isinstance(dataY, np.ndarray):
        dataY = dataY.flatten()
    
    img_per_lbl_per_btch = int(np.round(vars['num_img_per_label'] / vars['num_batches']))
    
    num_labels = len(np.unique(dataY))
    batch_size = img_per_lbl_per_btch * num_labels
    
    dataBatchX = np.ndarray(shape=(vars['num_batches'], batch_size,
                                   dataX.shape[1], dataX.shape[2], dataX.shape[3]),
                            dtype='float32')
    dataBatchY = np.ndarray(shape=(vars['num_batches'], batch_size),
                            dtype='float32')
    
    for batch_num in np.arange(vars['num_batches']):
        logging.info('Running for batch %s ', str(batch_num))
        batchX = np.ndarray(shape=(batch_size, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
        batchY = np.zeros(batch_size)
        for iter, labels in enumerate(np.unique(dataY)):
            logging.info('Running for label %s ', str(labels))
            label_idx = np.where(dataY == labels)[0]
            np.random.shuffle(label_idx)
            i = iter * img_per_lbl_per_btch
            j = (iter + 1) * img_per_lbl_per_btch
            batchX[i:j, :] = dataX[label_idx[0:img_per_lbl_per_btch]]
            batchY[i:j] = dataY[label_idx[0:img_per_lbl_per_btch]]
        dataBatchX[batch_num, :] = batchX
        dataBatchY[batch_num, :] = batchY
    
    if dump:
        if not os.path.exists(pathDict['batch_path']):
            os.makedirs(pathDict['batch_path'])
        logging.info('The Data batches dumped has shape: %s', str(dataBatchX.shape))
        logging.info('The Label batch dumped has shape: %s', str(dataBatchY.shape))
        dumpPickleFile(dataX=dataBatchX,
                       dataY=dataBatchY,
                       labelDict=label_dict,
                       folderPath=pathDict['batch_path'],
                       picklefileName=fileNames['batch_img_file'])
    return dataX, dataY, label_dict
#
