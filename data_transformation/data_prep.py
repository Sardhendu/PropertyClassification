import logging
import os

import numpy as np
import pandas as pd
from scipy import misc, ndimage
from config import pathDict
from data_transformation.data_io import dumpH5File, getH5File  # , dumpPickleFile, getPickleFile

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_intersecting_images_pin(is_assessor=True, is_aerial=True, is_streetside=True, is_overlayed=True,
                                is_aerial_cropped=True, is_training=True):
    '''
    :param aerial_img_type:
    :param streetside_img_type:
    :param overlayed_img_type:
    :param images_per_label:
    :return:

    Here we basically take the intersection of all the different types of images i.e Assessor images, aerial images
    and streetside images. Bu doing this we ensure that when we fit a neural network, it is actually fit on the same
    images but different types. This helps us to evaluate which image type is better. More this function return
    equal (balanced) number of land and house pins.
    '''
    cmn_land_pins = []
    cmn_house_pins = []
    
    if is_assessor:
        land_path = os.path.join(pathDict['input_image_run_dir'], 'assessor', 'land')
        house_path = os.path.join(pathDict['input_image_run_dir'], 'assessor', 'house')
        
        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the assessor image for your current run")
        
        cmn_land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        
        cmn_house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"],
                                  dtype=str)
    
    if is_aerial:
        land_path = os.path.join(pathDict['input_image_run_dir'], 'aerial', 'land')
        house_path = os.path.join(pathDict['input_image_run_dir'], 'aerial', 'house')
        
        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the aerial images for your current run")
        
        land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"], dtype=str)
        # print('aerial pins: ', len(land_pins), len(house_pins))
        
        if len(cmn_land_pins) > 0:
            cmn_land_pins = np.intersect1d(cmn_land_pins, land_pins)
            cmn_house_pins = np.intersect1d(cmn_house_pins, house_pins)
        else:
            cmn_land_pins, cmn_house_pins = np.array(land_pins), np.array(house_pins)
    
    if is_streetside:
        land_path = os.path.join(pathDict['input_image_run_dir'], 'streetside', 'land')
        house_path = os.path.join(pathDict['input_image_run_dir'], 'streetside', 'house')
        
        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the streetside images for your current run")
        
        land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"], dtype=str)
        # print('aerial pins: ', len(land_pins), len(house_pins))
        
        if len(cmn_land_pins) > 0:
            cmn_land_pins = np.intersect1d(cmn_land_pins, land_pins)
            cmn_house_pins = np.intersect1d(cmn_house_pins, house_pins)
        else:
            cmn_land_pins, cmn_house_pins = np.array(land_pins), np.array(house_pins)
            # print('common streetside pins: ', len(cmn_land_pins), len(cmn_house_pins))
    
    if is_overlayed:
        land_path = os.path.join(pathDict['input_image_run_dir'], 'overlayed', 'land')
        house_path = os.path.join(pathDict['input_image_run_dir'], 'overlayed', 'house')
        
        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the overlayed images for your current run")
        
        land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"], dtype=str)
        # print('aerial pins: ', len(land_pins), len(house_pins))
        
        if len(cmn_land_pins) > 0:
            cmn_land_pins = np.intersect1d(cmn_land_pins, land_pins)
            cmn_house_pins = np.intersect1d(cmn_house_pins, house_pins)
        else:
            cmn_land_pins, cmn_house_pins = np.array(land_pins), np.array(house_pins)
    
    if is_aerial_cropped:
        land_path = os.path.join(pathDict['input_image_run_dir'], 'aerial_cropped', 'land')
        house_path = os.path.join(pathDict['input_image_run_dir'], 'aerial_cropped', 'house')
        
        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the is_aerial_cropped images for your current run")
        
        land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"], dtype=str)
        # print('aerial pins: ', len(land_pins), len(house_pins))
        
        if len(cmn_land_pins) > 0:
            cmn_land_pins = np.intersect1d(cmn_land_pins, land_pins)
            cmn_house_pins = np.intersect1d(cmn_house_pins, house_pins)
        else:
            cmn_land_pins, cmn_house_pins = np.array(land_pins), np.array(house_pins)
    
    np.random.seed(184)
    np.random.shuffle(cmn_land_pins)
    np.random.shuffle(cmn_house_pins)
    
    if is_training:
        # For Training, in the return we ensure that the output PIN counts are balanced (equal for each class)
        images_per_label = min(len(cmn_land_pins), len(cmn_house_pins))
        return cmn_land_pins[0:images_per_label], cmn_house_pins[0:images_per_label]
    else:
        return cmn_land_pins, cmn_house_pins


def central_crop(image, height, width):
    "Function to crop center of an image file"
    if image.shape[0] > height and image.shape[1] > width:
        ysize, xsize, chan = image.shape
        xoff = (xsize - height) // 2
        yoff = (ysize - width) // 2
        img = image[yoff:-yoff, xoff:-xoff]
        # print (img.shape)
    else:
        img = image
    return img


# def zero_pad(inp, out_shape):
#     m, n, c = inp.shape
#     out_m, out_n, out_c = out_shape
#
#     to_pad_m = max(out_m - m, 0)
#     to_pad_n = max(out_n - n, 0)
#     to_pad_c = max(out_c - c, 0)
#
#     pad_m1 = to_pad_m // 2
#     pad_m2 = to_pad_m - pad_m1
#
#     pad_n1 = to_pad_n // 2
#     pad_n2 = to_pad_n - pad_n1
#
#     pad_c1 = to_pad_c // 2
#     pad_c2 = to_pad_c - pad_c1
#
#     inp = np.pad(inp, ((pad_m1, pad_m2), (pad_n1, pad_n2), (pad_c1, pad_c2)), 'constant')
#     return inp


def zero_pad(inp, crop_shape, out_shape):
    '''
    :param inp:
    :param out_shape:
    :return:

    One image at a time
    '''
    m, n, c = crop_shape  # inp.get_shape().as_list()
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
    
    inp = np.pad(inp, ((pad_m1, pad_m2), (pad_n1, pad_n2), (pad_c1, pad_c2)), 'constant')
    logging.info('Image shape after Zero Padding crop: %s', str(inp.shape))
    return inp


class DumpBatches():
    def __init__(self, outpath, img_in_shape, img_out_shape, img_crop_shape, img_resize_shape, image_type):
        # print(img_in_shape, img_out_shape, img_crop_shape, img_resize_shape)
        self.outpath = outpath
        self.img_in_shape = img_in_shape
        self.img_out_shape = img_out_shape

        if img_crop_shape is not None:
            self.img_crop_shape = img_crop_shape
        else:
            self.img_crop_shape = []
        
        if img_resize_shape is not None:
            self.img_resize_shape = img_resize_shape
        else:
            self.img_resize_shape = []
        self.image_type = image_type
    
    def process_images_given_path(self, pic_path):
        image = ndimage.imread(pic_path, mode='RGB')
        
        # if self.image_type == 'aerial_cropped':
        if image.shape[0] == self.img_in_shape[0] and image.shape[1] == self.img_in_shape[1]:
            # The above condition takes care of the different sizes. If a bounding box was already cropped
            # then we don't crop further. But if no bounding box was found then we perform a central crop.
            if len(self.img_crop_shape) > 0:
                image = central_crop(image, height=self.img_crop_shape[0], width=self.img_crop_shape[1])
        
        if len(self.img_resize_shape) > 0:
            image = misc.imresize(image, self.img_resize_shape)
            

        
        if image.shape[0] - self.img_out_shape[0] < 0:
            image = zero_pad(inp=image, crop_shape=self.img_crop_shape, out_shape=self.img_out_shape)
        elif len(self.img_out_shape) > 0:
            image = misc.imresize(image, self.img_out_shape)
        
        # if image.shape[0] < self.img_out_shape[0] or image.shape[1] < self.img_out_shape[1]:
        #     # IF the image does not fit the out shape then we pad the image with zeros
        #     image = zero_pad(inp=image, out_shape=self.img_out_shape)
        # else:
        #     # IF the image is larger than the out shape then we resize it fit the out_shape
        #     image = misc.imresize(image, self.img_out_shape)
        return image
    
    def dump_train_validate_test_batches(self, land_paths, house_paths, labels, filename):
        dataBatchX = np.ndarray(shape=(len(land_paths) + len(house_paths),
                                       self.img_out_shape[0],
                                       self.img_out_shape[1], 3), dtype='int32')
        for num, pic_path in enumerate(land_paths + house_paths):
            image = self.process_images_given_path(pic_path)
            dataBatchX[num, :] = image
        
        dataBatchY = np.append(
                np.tile(float(labels[0]), len(land_paths)),
                np.tile(float(labels[1]), len(house_paths)))
        
        dumpH5File(dataX=dataBatchX,
                   dataY=dataBatchY,
                   # labelDict=label_dict,
                   folderPath=self.outpath,
                   fileName=filename)
    
    def dump_new_data_batches(self, paths, filename):
        dataBatchX = np.ndarray(shape=(len(paths),
                                       self.img_out_shape[0],
                                       self.img_out_shape[1], 3), dtype='int32')
        for num, pic_path in enumerate(paths):
            image = self.process_images_given_path(pic_path)
            dataBatchX[num, :] = image
        
        dumpH5File(dataX=dataBatchX,
                   dataY=[],
                   # labelDict=label_dict,
                   folderPath=self.outpath,
                   fileName=filename)


def dumpStratifiedBatches_balanced_class(cmn_land_pins, cmn_house_pins, ts_batch_size, cv_batch_size, tr_batch_size,
                                         image_type, img_in_shape, img_out_shape, img_resize_shape=None,
                                         img_crop_shape=None,
                                         shuffle_seed=873, get_stats=True, max_batches=None,
                                         is_training=True):
    land_image_path = os.path.join(pathDict['image_path'], 'land')
    house_image_path = os.path.join(pathDict['image_path'], 'house')
    output_data_path = pathDict['batch_path']
    
    logging.info('Input Land Images from %s: ', str(land_image_path))
    logging.info('Input House Images from %s: ', str(house_image_path))
    logging.info('Output batch array to %s: ', str(output_data_path))
    
    land_label = 0
    house_label = 1
    label_dict = {'0': 'land', '1': 'house'}
    land_pins = np.sort(np.intersect1d(
            np.array([img.split('.')[0] for img in os.listdir(land_image_path) if img != '.DS_Store'], dtype=str),
            cmn_land_pins))
    
    house_pins = np.sort(np.intersect1d(
            np.array([img.split('.')[0] for img in os.listdir(house_image_path) if img != '.DS_Store'], dtype=str),
            cmn_house_pins))
    
    # RANDOM SHUFFLE LAND AND HOUSE PINS
    np.random.seed(shuffle_seed)
    np.random.shuffle(land_pins)
    np.random.seed(shuffle_seed + 62)
    np.random.shuffle(house_pins)
    
    logging.info('Input Data: Total Land: %s, Total House: %s', str(len(land_pins)), str(len(house_pins)))
    
    obj_dump = DumpBatches(outpath=output_data_path, img_in_shape=img_in_shape, img_out_shape=img_out_shape,
                           img_crop_shape=img_crop_shape, img_resize_shape=img_resize_shape, image_type=image_type)
    
    tr_cv_ts_pins_ = []
    tr_cv_ts_land_house = []
    tr_cv_ts_type_info = []
    if is_training:
        ####################################################################################
        # Create the first batch as Validation and then remove the validation data from the actual dataset. This is done
        # to ensure that the none of the validation data falls under the training batches
        ts_batch_size_per_class = ts_batch_size // 2
        cv_batch_size_per_class = cv_batch_size // 2
        
        test_land_pins = land_pins[0: ts_batch_size_per_class]
        test_house_pins = house_pins[0: ts_batch_size_per_class]
        
        cvalid_land_pins = land_pins[ts_batch_size_per_class: ts_batch_size_per_class + cv_batch_size_per_class]
        cvalid_house_pins = house_pins[ts_batch_size_per_class: ts_batch_size_per_class + cv_batch_size_per_class]
        
        # New land pins, and house_pins
        train_land_pins = land_pins[ts_batch_size_per_class + cv_batch_size_per_class:]
        train_house_pins = house_pins[ts_batch_size_per_class + cv_batch_size_per_class:]
        
        # print (len(train_land_pins), len(train_house_pins), len(cvalid_land_pins), len(cvalid_house_pins))
        ##### DUMP THE TEST DATASET
        obj_dump.dump_train_validate_test_batches(
                land_paths=[os.path.join(land_image_path, pin + '.jpg') for pin in test_land_pins],
                house_paths=[os.path.join(house_image_path, pin + '.jpg') for pin in test_house_pins],
                labels=[land_label, house_label], filename='test')
        
        # get_dump_image_given_path(land_paths, house_paths, label_dict, labels, outpath, filename,
        #                           img_out_shape, img_crop_shape, img_resize_shape, image_type)
        #
        ##### DUMP CROSS VALIDATION DATASET
        # LOAD THE VALIDATION SET TO THE DISK
        obj_dump.dump_train_validate_test_batches(
                land_paths=[os.path.join(land_image_path, pin + '.jpg') for pin in cvalid_land_pins],
                house_paths=[os.path.join(house_image_path, pin + '.jpg') for pin in cvalid_house_pins],
                labels=[land_label, house_label],
                filename='cvalid')
        
        # GATHER STATISTICS FOR CROSS VALIDATION DATASET
        if get_stats:
            tr_cv_ts_pins_ = np.append(np.append(test_land_pins, test_house_pins),
                                       np.append(cvalid_land_pins, cvalid_house_pins))
            tr_cv_ts_land_house = np.append(
                    np.append(np.tile('land', len(test_land_pins)), np.tile('house', len(test_house_pins))),
                    np.append(np.tile('land', len(cvalid_land_pins)), np.tile('house', len(cvalid_house_pins)))
            )
            tr_cv_ts_type_info = np.append(np.tile('test', ts_batch_size), np.tile('cvalid', cv_batch_size))
        
        ##### DUMP TRAINING DATA IN BATCHES
        num_batches = int(np.ceil(len(train_land_pins) + len(train_house_pins)) / tr_batch_size)
        tr_batch_size_per_class = tr_batch_size // 2
        
        for batch_num in range(0, num_batches):
            if batch_num != (num_batches - 1):
                from_idx = batch_num * tr_batch_size_per_class
                to_idx = (batch_num * tr_batch_size_per_class) + tr_batch_size_per_class
            else:
                from_idx = batch_num * tr_batch_size_per_class
                to_idx = (batch_num * tr_batch_size_per_class) + (
                    len(train_land_pins) - (batch_num * tr_batch_size_per_class))
            
            element_count = to_idx - from_idx
            
            batch_land_pins = train_land_pins[from_idx:to_idx]
            batch_house_pins = train_house_pins[from_idx:to_idx]
            
            obj_dump.dump_train_validate_test_batches(
                    land_paths=[os.path.join(land_image_path, pin + '.jpg') for pin in batch_land_pins],
                    house_paths=[os.path.join(house_image_path, pin + '.jpg') for pin in batch_house_pins],
                    labels=[land_label, house_label],
                    filename='train_%s' % str(batch_num))
            
            # GATHER STATISTICS ABOUT PINS AND THEIR BATCH NUMBER
            if get_stats:
                tr_cv_ts_pins_ = np.append(tr_cv_ts_pins_, np.append(batch_land_pins, batch_house_pins))
                tr_cv_ts_land_house = np.append(tr_cv_ts_land_house,
                                                np.append(np.tile('land', len(batch_land_pins)),
                                                          np.tile('house', len(batch_house_pins))))
                tr_cv_ts_type_info = np.append(tr_cv_ts_type_info,
                                               np.tile('batch_%s' % str(batch_num), element_count * 2))
            
            if max_batches:
                if max_batches == batch_num + 1:
                    break
        
        ##### DUMP TRAIN CV STATISTICS INFO
        if get_stats:
            folder_path = os.path.join(pathDict['statistics_path'], 'prediction_stats')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            dump_pins_path = os.path.join(folder_path, 'tr_cv_ts_pins_info.csv')
            dataOUT = pd.DataFrame(np.column_stack((tr_cv_ts_pins_, tr_cv_ts_land_house, tr_cv_ts_type_info)),
                                   columns=['property_pins', 'property_type', 'dataset_type'])
            dataOUT.to_csv(dump_pins_path, index=None)
            
            logging.info(
                    'Validation Land Size: %s, Validation House Size: %s, Training Land Size: %s, Training House '
                    'Size: %s',
                    str(len(cvalid_land_pins)), str(len(cvalid_house_pins)), str(len(train_land_pins)),
                    str(len(train_house_pins)))
    else:
        img_paths = np.append([os.path.join(land_image_path, pin + '.jpg') for pin in land_pins],
                              [os.path.join(land_image_path, pin + '.jpg') for pin in house_pins])
        
        # for paths in img_paths:
        num_batches = int(np.ceil(len(img_paths) / ts_batch_size))
        for batch_num in range(0, num_batches):
            if batch_num != (num_batches - 1):
                from_idx = batch_num * ts_batch_size
                to_idx = (batch_num * ts_batch_size) + ts_batch_size
            else:
                from_idx = batch_num * ts_batch_size
                to_idx = (batch_num * ts_batch_size) + (len(img_paths) - (batch_num * ts_batch_size))
            
            element_count = to_idx - from_idx
            
            batch_paths = img_paths[from_idx:to_idx]
            
            obj_dump.dump_new_data_batches(batch_paths, filename='test_%s' % str(batch_num))


debugg = False
if debugg:
    import time
    
    start_time = time.time()
    
    cmn_land_pins, cmn_house_pins = get_intersecting_images_pin(is_assessor=False, is_aerial=True,
                                                                is_streetside=False, is_overlayed=True,
                                                                is_aerial_cropped=True, is_training=True)
    print(len(cmn_land_pins), len(cmn_house_pins))
    
    tr_batch_size = 128
    ts_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
    cv_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
    
    dumpStratifiedBatches_balanced_class(cmn_land_pins, cmn_house_pins, ts_batch_size=ts_batch_size,
                                         cv_batch_size=cv_batch_size, tr_batch_size=tr_batch_size,
                                         image_type='aerial_cropped', img_in_shape=[400, 400, 3],
                                         img_out_shape=[224, 224, 3],
                                         img_crop_shape=[128, 128, 3], img_resize_shape=[128, 128, 3],
                                         shuffle_seed=873, get_stats=True, max_batches=None, is_training=True)
    print('--------------- %s seconds ------------------' % (time.time() - start_time))



#
# debugg = False
# if debugg:
#     cvalid_land_pins = np.array(['30-07-132-018-0000', '29-36-102-064-0000', '24-33-100-102-0000',
#                                  '31-34-404-016-0000', '24-04-427-016-0000', '18-04-236-010-0000',
#                                  '28-22-315-044-0000', '30-07-102-004-0000', '16-32-405-054-0000',
#                                  '29-36-101-023-0000', '29-36-101-022-0000', '01-13-101-026',
#                                  '30-07-108-009-0000', '31-34-402-001-0000', '30-30-107-003-0000',
#                                  '19-01-311-005-0000', '17-31-104-037-0000', '28-22-315-032-0000',
#                                  '26-30-404-045-0000', '26-30-404-044-0000', '29-16-118-084-0000',
#                                  '31-34-403-029-0000', '05-10-411-007', '06-36-307-008-0000',
#                                  '31-35-331-009-0000', '29-25-400-117-0000', '17-31-302-013-0000'], dtype=str)
#
#     valid_house_ids = np.array(['14-33-108-015', '29-04-203-032-0000',
#                                 '11-20-218-024', '16-23-304-011',
#                                 '16-34-212-014', '16-10-411-005-0000', '16-23-304-005', '15-10-100-043',
#                                 '20-18-221-030-0000', '26-07-302-005-0000', '09-26-101-015', '08-29-207-029',
#                                 '16-18-102-001', '20-20-311-006-0000', '07-01-210-021', '16-11-102-002',
#                                 '11-31-116-012', '07-19-402-001', '16-08-402-009', '09-13-208-007',
#                                 '09-35-103-010', '13-34-201-039', '21-31-400-005-0000', '29-24-100-018-1035',
#                                 '17-09-124-020-1222', '07-15-202-002', '06-27-406-020', '08-29-220-010',
#                                 '14-05-301-001', '13-26-101-001', '08-32-429-025', '09-36-109-002',
#                                 '06-17-421-013', '08-19-106-006' '07-07-407-006' '06-18-303-008',
#                                 '25-27-126-013-0000', '25-22-306-060-0000', '30-17-107-043-0000',
#                                 '09-24-206-042', '05-23-200-043', '07-15-206-036', '04-28-216-008',
#                                 '01-33-100-603', '13-01-100-013', '32-25-307-021-0000', '10-25-204-015',
#                                 '10-24-412-005', '07-26-419-018', '31-04-404-011-0000', '31-02-204-170-0000',
#                                 '08-04-111-001', '29-32-406-043-1219', '09-25-102-004', '29-19-413-054-0000',
#                                 '08-16-335-003', '20-24-430-011-1113', '22-33-400-011-0000', '07-02-102-028'
#                                                                                              '08-21-324-003',
#                                 '13-28-400-005', '14-33-205-001', '18-13-223-020-0000',
#                                 '11-29-401-020', '11-21-309-013', '30-20-304-061-0000', '20-02-303-040-0000',
#                                 '05-23-200-042', '11-29-306-010', '04-16-412-015', '30-17-103-004-0000',
#                                 '07-06-303-011'])
#
#     images_per_label = None  # normally 5000 each label is good
#     assessor_img_type = 'assessor'
#     aerial_img_type = 'google_aerial'  # 'bing_aerial'
#     overlayed_img_type = 'google_overlayed'
#     streetside_img_type = None
#     # image_shape = [224,224,3]
#     # inp_image_shape = [260, 260, 3]
#     inp_image_shape = [400, 400, 3]
#
#     image_type = overlayed_img_type  # aerial_img_type#assessor_img_type
#
#     cmn_land_pins, cmn_house_pins = get_valid_land_house_ids(
#             aerial_img_type=aerial_img_type,
#             streetside_img_type=streetside_img_type,
#             overlayed_img_type=overlayed_img_type,
#             images_per_label=images_per_label)
#     print(len(cmn_land_pins), len(cmn_house_pins))
#
#     dumpStratifiedBatches_balanced_class(image_type, inp_image_shape, cmn_land_pins, cmn_house_pins,
#                                          cv_batch_size=(len(cmn_land_pins) + len(cmn_house_pins)) // 10,
#                                          tr_batch_size=128, shuffle_seed=873)
#     # genStratifiedBatches([224, 224], cvalid_land_pins, valid_house_ids, cv_batch_size=100, tr_batch_size=20,
# image_type='aerial',
#     #                            dump=True)
