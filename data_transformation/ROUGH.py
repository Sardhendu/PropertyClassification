import logging
import os

import numpy as np
import pandas as pd
from scipy import  ndimage

from config import pathDict, myNet, fileNames, vars
from data_transformation.data_io import dumpPickleFile, getPickleFile

#
# #### deprecated
# def get_resized_image_arr(image_type, nw_shape, get_stats=False, dump=True):
#     if image_type not in ['assessor', 'aerial']:
#         raise ValueError('Variable image_type not understood')
#
#     image_folder_path = pathDict['assessor_image_path']
#     dump_folder_path = pathDict[ '%s_rsized_path' %(image_type)]
#     stats_path = pathDict[ '%s_dp_stats_path' %(image_type)]
#     logging.info('Fetching images from: %s', str(image_folder_path))
#     logging.info('Dumping reshaped images to: %s', str(dump_folder_path))
#
#     if not os.path.exists(image_folder_path):
#         raise ValueError('The images are not in place')
#
#     image_types = [image_type for image_type in os.listdir(image_folder_path)
#                    if image_type != ".DS_Store"]
#
#     # print (image_types)
#
#     dataX = []
#     dataY = []
#     label_dict = {}
#     pic_filename_arr = []
#     label_name_arr = []
#     label_idx_arr = []
#     for label, image_type in enumerate(image_types):
#         print (label, image_type)
#         image_path = os.path.join(image_folder_path, image_type)
#         print(image_path)
#
#         image_paths = [os.path.join(image_path, pics)
#                        for pics in os.listdir(image_path)
#                        if pics.split('.')[1] == "jpg" or pics.split('.')[1] == "jpeg" or pics.split('.')[1] == "png"]
#         print (image_paths)
#         # break
#         label_dict[str(label)] = image_type
#         image_ndarr = np.ndarray((len(image_paths), myNet['image_shape'][0],
#                                   myNet['image_shape'][1], myNet['image_shape'][2]))
#         label_ndarr = np.tile(label, len(image_paths)).reshape(-1, 1)
#
#         for img_num, pic_path in enumerate(image_paths):
#             pic_filename_arr += [os.path.basename(pic_path).split('.')[0]]
#             image = ndimage.imread(pic_path, mode='RGB')
#             rsized_image = resize_image(image, resize_shape=nw_shape)
#             image_ndarr[img_num, :] = rsized_image
#
#         label_name_arr += [image_type] * len(image_paths)
#         label_idx_arr += [str(label)] * len(image_paths)
#
#         if label == 0:
#             dataX = image_ndarr
#             dataY = label_ndarr
#         else:
#             dataX = np.vstack((dataX, image_ndarr))
#             dataY = np.vstack((dataY, label_ndarr))
#
#     print (dataX.shape, dataY.shape)
#
#     image_type_image_num_info = []
#     if get_stats:
#         image_type_image_num_info = np.column_stack((
#             np.array(label_idx_arr).reshape(-1, 1),
#             np.array(label_name_arr).reshape(-1, 1),
#             np.array(pic_filename_arr).reshape(-1, 1)
#         ))
#         image_type_image_num_info = pd.DataFrame(
#                 image_type_image_num_info,
#                 columns=['image_label',
#                          'person_name',
#                          'file_name'])
#         image_type_image_num_info = image_type_image_num_info.reset_index()
#
#     if dump:
#         dumpPickleFile(dataX=dataX, dataY=dataY, labelDict=label_dict, folderPath=dump_folder_path,
#                        picklefileName=fileNames['rsized_img_file'])
#         if get_stats:
#             image_type_image_num_info.to_csv(stats_path)
#
#     return dataX, dataY, label_dict, image_type_image_num_info
#
#
# #### deprecated
# def genRandomStratifiedBatchesOLD(dataX=[], dataY=[], label_dict={}, dump=True):
#     if len(dataX) == 0:
#         dataX, dataY, label_dict = getPickleFile(pathDict['input_rsized_image_path'], fileNames['rsized_img_file'])
#
#     if not isinstance(dataX, np.ndarray):
#         raise ValueError('Unhandled type dataX input')
#
#     if isinstance(dataY, np.ndarray):
#         dataY = dataY.flatten()
#
#     img_per_lbl_per_btch = int(np.round(vars['num_img_per_label'] / vars['num_batches']))
#
#     num_labels = len(np.unique(dataY))
#     batch_size = img_per_lbl_per_btch * num_labels
#
#     dataBatchX = np.ndarray(shape=(vars['num_batches'], batch_size,
#                                    dataX.shape[1], dataX.shape[2], dataX.shape[3]),
#                             dtype='float32')
#     dataBatchY = np.ndarray(shape=(vars['num_batches'], batch_size),
#                             dtype='float32')
#
#     for batch_num in np.arange(vars['num_batches']):
#         logging.info('Running for batch %s ', str(batch_num))
#         batchX = np.ndarray(shape=(batch_size, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
#         batchY = np.zeros(batch_size)
#         for iter, labels in enumerate(np.unique(dataY)):
#             logging.info('Running for label %s ', str(labels))
#             label_idx = np.where(dataY == labels)[0]
#             np.random.shuffle(label_idx)
#             i = iter * img_per_lbl_per_btch
#             j = (iter + 1) * img_per_lbl_per_btch
#             batchX[i:j, :] = dataX[label_idx[0:img_per_lbl_per_btch]]
#             batchY[i:j] = dataY[label_idx[0:img_per_lbl_per_btch]]
#         dataBatchX[batch_num, :] = batchX
#         dataBatchY[batch_num, :] = batchY
#
#     if dump:
#         if not os.path.exists(pathDict['batch_path']):
#             os.makedirs(pathDict['batch_path'])
#         logging.info('The Data batches dumped has shape: %s', str(dataBatchX.shape))
#         logging.info('The Label batch dumped has shape: %s', str(dataBatchY.shape))
#         dumpPickleFile(dataX=dataBatchX,
#                        dataY=dataBatchY,
#                        labelDict=label_dict,
#                        folderPath=pathDict['batch_path'],
#                        picklefileName=fileNames['batch_img_file'])
#     return dataX, dataY, label_dict
# #





# def dumpStratifiedBatches_unbalanced_class(img_resize_shape, cvalid_land_pins, cvalid_house_pins, cv_batch_size,
#                                      tr_batch_size,
#                        image_type='aerial', dump=True, shuffle_seed=873):
#
#     if image_type not in ['assessor', 'google_aerial', 'bing_aerial', 'bing_streetside', 'google_overlayed']:
#         raise ValueError('Variable image_type not understood')
#
#     land_image_path = os.path.join(pathDict['%s_image_path' % (image_type)], 'land')
#     house_image_path = os.path.join(pathDict['%s_image_path' % (image_type)], 'house')
#     output_batch_path = pathDict['%s_batch_path' % (image_type)]
#     # print (output_batch_path)
#     logging.info('Input Land Images from %s: ', str(land_image_path))
#     logging.info('Input House Images from %s: ', str(house_image_path))
#     logging.info('Output batch array to %s: ', str(output_batch_path))
#
#     land_label = 0
#     house_label = 1
#     label_dict = {'0': 'land', '1': 'house'}
#     land_pins = np.sort(np.intersect1d(
#         np.array([img.split('.')[0] for img in os.listdir(land_image_path) if img != '.DS_Store'], dtype=str),
#         cvalid_land_pins))
#
#     house_pins = np.sort(np.intersect1d(
#             np.array([img.split('.')[0] for img in os.listdir(house_image_path) if img != '.DS_Store'], dtype=str),
#             cvalid_house_pins))
#
#     my_seed = shuffle_seed
#     np.random.seed(my_seed)
#     np.random.shuffle(land_pins)
#     np.random.shuffle(house_pins)
#
#     logging.info('Input Data: Total Land: %s, Total House: %s', str(len(land_pins)), str(len(house_pins)))
#
#     ####################################################################################
#     # Create the first batch as Validation and then remove the validation data from the actual dataset. This is
# done to ensure that the none of the validation data falls under the training batches
#     cv_batch_size_per_class = cv_batch_size // 2
#     cvalid_land_pins = land_pins[0: cv_batch_size_per_class]
#     cvalid_house_pins = house_pins[0: cv_batch_size_per_class]
#
#     # New land pins, and house_pins
#     train_land_pins = land_pins[cv_batch_size_per_class: ]
#     train_house_pins = house_pins[cv_batch_size_per_class:]
#
#     paths = [os.path.join(land_image_path, pin + '.jpg') for pin in cvalid_land_pins] + \
#             [os.path.join(house_image_path, pin + '.jpg') for pin in cvalid_house_pins]
#
#     # LOAD THE VALIDATION SET TO THE DISK
#     if dump:
#         get_dump_image_given_path(paths, img_resize_shape,
#                                   cv_batch_size_per_class, label_dict,
#                                   labels=[land_label,house_label],
#                                   outpath=output_batch_path,
#                                   filename='cv.pickle')
#
#         # In-order to validate the cross validation results we need to manually analyze what images are classified
#         # correctly are which images arent. Here we build a statistic file with all the pins sequentially stashed in
#         # the Cross-validation data set.
#         # print (train_land_pins, type(train_land_pins))
#         cv_pins_path = os.path.join(pathDict['%s_pred_stats' % str(image_type)], 'cv_pins.csv')
#         _pins_ = np.append(cvalid_land_pins, cvalid_house_pins)
#         _pins_ = pd.DataFrame(_pins_, columns=['property_pins'])
#         _pins_.to_csv(cv_pins_path, index=None)
#
#     logging.info('Validation Land Size: %s, Validation House Size: %s, Training Land Size: %s, Training House Size:
#  %s', str(len(cvalid_land_pins)), str(len(cvalid_house_pins)), str(len(train_land_pins)), str(len(train_house_pins)))
#
#     ####################################################################################
#
#     tr_img_per_label_per_batch = tr_batch_size // 2
#
#     if len(train_land_pins) > len(train_house_pins):
#         bigger, b_label = train_land_pins, land_label
#         smaller, s_label = train_house_pins, house_label
#         b_land, b_house = True, False
#     else:
#         bigger, b_label = train_house_pins, house_label
#         smaller, s_label = train_land_pins, land_label
#         b_house, b_land = True, False
#
#
#     num_batches = int(np.ceil(len(bigger) / tr_img_per_label_per_batch))
#
#     count = 1
#     pin_batch_row_meta = None
#     for batch_num in range(0, num_batches):
#         logging.info('Creating BATCH %s .......... ', str(batch_num))
#         idx1 = batch_num * tr_img_per_label_per_batch
#         idx2 = batch_num * tr_img_per_label_per_batch + tr_img_per_label_per_batch
#
#         if len(bigger) >= idx2:
#             logging.info('Bigger: len(bigger) >= idx2 ')
#             b_batch_pins = bigger[idx1: idx2]
#         elif len(bigger) > idx1 and len(bigger) < idx2:
#             logging.info('Bigger: idx1 < len(bigger) < idx2 ')
#             b_batch_pins = bigger[idx1:len(bigger)]
#             how_many = idx2 - len(bigger)
#             random.seed(my_seed + count)
#             b_idx = random.sample(range(0, len(bigger)), how_many)
#             b_batch_pins = np.append(b_batch_pins, bigger[b_idx])
#             count += 1
#         else:
#             logging.info('Bigger: len(bigger) < idx1 < idx2 ')
#             how_many = tr_img_per_label_per_batch
#             random.seed(my_seed + count)
#             b_idx = random.sample(range(0, len(bigger)), how_many)
#             b_batch_pins = smaller[b_idx]
#             count += 1
#
#         if len(smaller) >= idx2:
#             logging.info('Smaller: len(Smaller) >= idx2 ')
#             s_batch_pins = smaller[idx1: idx2]
#         elif len(smaller) > idx1 and len(smaller) < idx2:
#             logging.info('Smaller: idx1 < len(Smaller) < idx2 ')
#             s_batch_pins = smaller[idx1:len(smaller)]
#             how_many = idx2 - len(smaller)
#             random.seed(my_seed + count)
#             s_idx = random.sample(range(0, len(smaller)), how_many)
#             s_batch_pins = np.append(s_batch_pins, smaller[s_idx])
#             count += 1
#         else:
#             logging.info('Smaller: len(Smaller) < idx1 < idx2 ')
#             how_many = tr_img_per_label_per_batch
#             random.seed(my_seed + count)
#             s_idx = random.sample(range(0, len(smaller)), how_many)
#             s_batch_pins = smaller[s_idx]
#             count += 1
#
#         # GET IMAGE PATH
#         if b_land:
#             paths = [os.path.join(land_image_path, pin + '.jpg') for pin in b_batch_pins] + [
#                 os.path.join(house_image_path, pin + '.jpg') for pin in s_batch_pins]
#             label = np.array(['land'] * len(b_batch_pins) + ['house'] * len(s_batch_pins))
#         else:
#             paths = [os.path.join(house_image_path, pin + '.jpg') for pin in b_batch_pins] + [
#                 os.path.join(land_image_path, pin + '.jpg') for pin in s_batch_pins]
#             label = np.array(['house'] * len(b_batch_pins) + ['land'] * len(s_batch_pins))
#
#         # We would like to create a metadata table that hods the list of pins and their corresponding batch number
#         # and record number, so that we can perform some testing and validation.
#         pins = np.vstack((b_batch_pins.reshape(-1, 1), s_batch_pins.reshape(-1, 1)))
#         batch_no = np.tile(batch_num, len(pins))
#         rownum = np.arange(len(batch_no))
#
#         if batch_num == 0:
#             pin_batch_row_meta = np.column_stack(
#                     (pins, batch_no.reshape(-1, 1), rownum.reshape(-1, 1), label.reshape(-1, 1)))
#         else:
#             pin_batch_row_meta = np.vstack((pin_batch_row_meta,
#                                             np.column_stack((pins, batch_no.reshape(-1, 1), rownum.reshape(-1, 1),
#                                                              label.reshape(-1, 1)))))
#
#
#         if dump:
#             get_dump_image_given_path(paths, img_resize_shape,
#                                       tr_img_per_label_per_batch, label_dict,
#                                       labels=[b_label, s_label],
#                                       outpath=output_batch_path,
#                                       filename='tr%s.pickle'%str(batch_num))
#
#     if dump:
#         # DUMP THE PIN_BATCH_ROW_META
#         pin_batch_row_meta = pd.DataFrame(pin_batch_row_meta, columns=['pin', 'batch_no', 'rownum', 'indicator'])
#         pin_batch_row_meta.to_csv(
#             os.path.join(pathDict['pin_batch_row_meta_path'], '%s_pin_batch_row_meta.csv' % (image_type)),
#             index=None)
#
