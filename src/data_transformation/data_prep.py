import logging
import os

import imutils
import numpy as np
import pandas as pd
from scipy import misc, ndimage

from src.data_transformation.data_io import dumpH5File  # , dumpPickleFile, getPickleFile

# logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
#                     format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_intersecting_images_pin(conf, is_assessor=True, is_aerial=True, is_streetside=True, is_overlaid=True, is_aerial_cropped=True, equal_proportion=True):
    '''
    :param aerial_img_type:
    :param streetside_img_type:
    :param overlaid_img_type:
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
        land_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'assessor', 'land')
        house_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'assessor', 'house')

        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the assessor image for your current run")

        cmn_land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)

        cmn_house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"],
                                  dtype=str)

    if is_aerial:
        land_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial', 'land')
        house_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial', 'house')

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
        land_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'streetside', 'land')
        house_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'streetside', 'house')

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

    if is_overlaid:
        land_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'overlaid', 'land')
        house_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'overlaid', 'house')

        if not os.path.exists(land_path) or not os.path.exists(house_path):
            raise ValueError("It seems you haven't parsed and dumped the overlaid images for your current run")

        land_pins = np.array([img.split('.')[0] for img in os.listdir(land_path) if img != ".DS_Store"], dtype=str)
        house_pins = np.array([img.split('.')[0] for img in os.listdir(house_path) if img != ".DS_Store"], dtype=str)
        # print('aerial pins: ', len(land_pins), len(house_pins))

        if len(cmn_land_pins) > 0:
            cmn_land_pins = np.intersect1d(cmn_land_pins, land_pins)
            cmn_house_pins = np.intersect1d(cmn_house_pins, house_pins)
        else:
            cmn_land_pins, cmn_house_pins = np.array(land_pins), np.array(house_pins)

    if is_aerial_cropped:
        land_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial_cropped', 'land')
        house_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial_cropped', 'house')

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

    # For Training, in the return we ensure that the output PIN counts are balanced (equal for each class)
    if equal_proportion:
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
    # logging.info('Image shape after Zero Padding crop: %s', str(inp.shape))
    return inp


def process_images_given_path(pic_path, img_in_shape, img_crop_shape, img_resize_shape, img_out_shape,
                              enable_rotation, angle):
    bbox_cropped = None
    image = ndimage.imread(pic_path, mode='RGB')

    # if self.image_type == 'aerial_cropped':
    if image.shape[0] == img_in_shape[0] and image.shape[1] == img_in_shape[1]:
        bbox_cropped = 0
        # The above condition takes care of the different sizes. If a bounding box was already cropped
        # then we don't crop further. But if no bounding box was found then we perform a central crop.
        if len(img_crop_shape) > 0:
            image = central_crop(image, height=img_crop_shape[0], width=img_crop_shape[1])
    else:
        # If the image shape is not 400x400, then it means that the image was cropped using building polygons
        bbox_cropped = 1
        # If the height is greater than width then we rotate the image by 90%
    if enable_rotation:
        if image.shape[0] > image.shape[1]:
            image = imutils.rotate_bound(image, angle)

    if len(img_resize_shape) > 0:
        image = misc.imresize(image, img_resize_shape)

    if image.shape[0] - img_out_shape[0] < 0:
        image = zero_pad(inp=image, crop_shape=img_crop_shape, out_shape=img_out_shape)
    elif len(img_out_shape) > 0:
        image = misc.imresize(image, img_out_shape)

    # if image.shape[0] < self.img_out_shape[0] or image.shape[1] < self.img_out_shape[1]:
    #     # IF the image does not fit the out shape then we pad the image with zeros
    #     image = zero_pad(inp=image, out_shape=self.img_out_shape)
    # else:
    #     # IF the image is larger than the out shape then we resize it fit the out_shape
    #     image = misc.imresize(image, self.img_out_shape)
    return image, bbox_cropped


class DumpBatches():
    def __init__(self, conf, params):
        # print(img_in_shape, img_out_shape, img_crop_shape, img_resize_shape)
        self.stats_path = conf['pathDict']['statistics_path']
        self.land_image_path = os.path.join(conf['pathDict']['image_path'], 'land')
        self.house_image_path = os.path.join(conf['pathDict']['image_path'], 'house')
        self.output_data_path = conf['pathDict']['batch_path']

        logging.info('Input Land Images from %s: ', str(self.land_image_path))
        logging.info('Input House Images from %s: ', str(self.house_image_path))
        logging.info('Output batch array to %s: ', str(self.output_data_path))

        param_keys = params.keys()
        self.img_in_shape = params["img_in_shape"]
        self.img_out_shape = params["img_out_shape"]

        logging.info('Image Shape: %s', str(self.img_in_shape))

        if "img_crop_shape" in param_keys:
            self.img_crop_shape = params["img_crop_shape"]
            logging.info('Enabling Crop: %s', str(self.img_crop_shape))
        else:
            self.img_crop_shape = []

        if "img_resize_shape" in param_keys:
            self.img_resize_shape = params["img_resize_shape"]
            logging.info('Enabling Resize: %s', str(self.img_resize_shape))
        else:
            self.img_resize_shape = []

        if "enable_rotation" in param_keys:
            logging.info('Enabling Rotation angle = 90:')
            self.angle = 90
            self.enable_rotation = params["enable_rotation"]
        else:
            self.enable_rotation = False

        self.image_type = params["image_type"]
        self.shuffle_seed = params['shuffle_seed']
        self.get_stats = params['get_stats']
        self.max_batches = params['max_batches']

        self.tr_batch_size = params["tr_batch_size"]
        self.cv_batch_size = params["cv_batch_size"]
        self.ts_batch_size = params["ts_batch_size"]

        if self.ts_batch_size % 2 != 0:
            self.ts_batch_size -= 1

        if self.cv_batch_size % 2 != 0:
            self.cv_batch_size -= 1
        
        

        logging.info('Running Seed for batch creation: %s', str(self.shuffle_seed))

    def process_images_given_path_wrapper(self, pic_path):
        return process_images_given_path(pic_path, self.img_in_shape, self.img_crop_shape, self.img_resize_shape,
                                         self.img_out_shape,
                                         self.enable_rotation, self.angle)

        # if image.shape[0] < self.img_out_shape[0] or image.shape[1] < self.img_out_shape[1]:
        #     # IF the image does not fit the out shape then we pad the image with zeros
        #     image = zero_pad(inp=image, out_shape=self.img_out_shape)
        # else:
        #     # IF the image is larger than the out shape then we resize it fit the out_shape
        #     image = misc.imresize(image, self.img_out_shape)
        # return image, bbox_cropped

    def dump_train_validate_test_batches(self, land_paths, house_paths, labels, filename):
        dataBatchX = np.ndarray(shape=(len(land_paths) + len(house_paths),
                                       self.img_out_shape[0],
                                       self.img_out_shape[1], 3), dtype='int32')
        bbox_cropped_arr = []
        for num, pic_path in enumerate(land_paths + house_paths):
            image, bbox_cropped = self.process_images_given_path_wrapper(pic_path)
            bbox_cropped_arr.append(bbox_cropped)
            dataBatchX[num, :] = image

        dataBatchY = np.append(
            np.tile(float(labels[0]), len(land_paths)),
            np.tile(float(labels[1]), len(house_paths)))

        dumpH5File(dataX=dataBatchX,
                   dataY=dataBatchY,
                   # labelDict=label_dict,
                   folderPath=self.output_data_path,
                   fileName=filename)
        return bbox_cropped_arr

    def dump_new_data_batches(self, paths, filename):
        dataBatchX = np.ndarray(shape=(len(paths),
                                       self.img_out_shape[0],
                                       self.img_out_shape[1], 3), dtype='int32')
        bbox_cropped_arr = []
        for num, pic_path in enumerate(paths):
            image, bbox_cropped = self.process_images_given_path_wrapper(pic_path)
            bbox_cropped_arr.append(bbox_cropped)
            dataBatchX[num, :] = image

        dumpH5File(dataX=dataBatchX,
                   dataY=[],
                   # labelDict=label_dict,
                   folderPath=self.output_data_path,
                   fileName=filename)
        return bbox_cropped_arr
    #
    def dump_cv_test_data(self, land_pins, house_pins, land_label, house_label):
        ####################################################################################
        # Create the first batch as Validation and then remove the validation data from the actual dataset. This is done
        # to ensure that the none of the validation data falls under the training batches
        self.ts_batch_size_per_class = self.ts_batch_size // 2
        self.cv_batch_size_per_class = self.cv_batch_size // 2

        test_land_pins = land_pins[0: self.ts_batch_size_per_class]
        test_house_pins = house_pins[0: self.ts_batch_size_per_class]

        cvalid_land_pins = land_pins[
                                self.ts_batch_size_per_class: self.ts_batch_size_per_class +
                                                              self.cv_batch_size_per_class]
        cvalid_house_pins = house_pins[
                                 self.ts_batch_size_per_class: self.ts_batch_size_per_class +
                                                               self.cv_batch_size_per_class]

        # print (len(train_land_pins), len(train_house_pins), len(cvalid_land_pins), len(cvalid_house_pins))
        ##### DUMP THE TEST DATASET
        ts_bbox_cropped_arr = self.dump_train_validate_test_batches(
            land_paths=[os.path.join(self.land_image_path, pin + '.jpg') for pin in test_land_pins],
            house_paths=[os.path.join(self.house_image_path, pin + '.jpg') for pin in test_house_pins],
            labels=[land_label, house_label], filename='test')

        ##### DUMP CROSS VALIDATION DATASET
        # LOAD THE VALIDATION SET TO THE DISK
        cv_bbox_cropped_arr = self.dump_train_validate_test_batches(
            land_paths=[os.path.join(self.land_image_path, pin + '.jpg') for pin in cvalid_land_pins],
            house_paths=[os.path.join(self.house_image_path, pin + '.jpg') for pin in cvalid_house_pins],
            labels=[land_label, house_label],
            filename='cvalid')

        # GATHER STATISTICS FOR CROSS VALIDATION DATASET
        if self.get_stats:
            self.keep_rownum = np.append(np.arange(len(test_land_pins)+ len(test_house_pins)),
                                        np.arange(len(cvalid_land_pins)+ len(cvalid_house_pins)))
            
            self.tr_cv_ts_pins_ = np.append(np.append(test_land_pins, test_house_pins),
                                       np.append(cvalid_land_pins, cvalid_house_pins))
            self.tr_cv_ts_bbox_crpd = np.append(np.array(ts_bbox_cropped_arr), np.array(cv_bbox_cropped_arr))
            self.tr_cv_ts_land_house = np.append(
                np.append(np.tile('land', len(test_land_pins)), np.tile('house', len(test_house_pins))),
                np.append(np.tile('land', len(cvalid_land_pins)),
                          np.tile('house', len(cvalid_house_pins)))
            )
            self.tr_cv_ts_type_info = np.append(np.tile('test', self.ts_batch_size), np.tile('cvalid', self.cv_batch_size))

        logging.info(
                'Validation Land Size: %s, Validation House Size: %s, Test Land Size: %s, Test House '
                'Size: %s',
                str(len(cvalid_land_pins)), str(len(cvalid_house_pins)),
                str(len(test_land_pins)),
                str(len(test_house_pins)))
            
    
    def dump_training_batches(self, land_pins, house_pins, land_label, house_label):
        # New land pins, and house_pins
        train_land_pins = land_pins[self.ts_batch_size_per_class + self.cv_batch_size_per_class:]
        train_house_pins = house_pins[self.ts_batch_size_per_class + self.cv_batch_size_per_class:]
        
        num_batches = int(np.ceil(len(train_land_pins) + len(train_house_pins)) / self.tr_batch_size)
        tr_batch_size_per_class = self.tr_batch_size // 2
    
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
        
            tr_bbox_cropped_arr = self.dump_train_validate_test_batches(
                    land_paths=[os.path.join(self.land_image_path, pin + '.jpg') for pin in batch_land_pins],
                    house_paths=[os.path.join(self.house_image_path, pin + '.jpg') for pin in batch_house_pins],
                    labels=[land_label, house_label],
                    filename='batch_%s' % str(batch_num))
        
            # GATHER STATISTICS ABOUT PINS AND THEIR BATCH NUMBER
            if self.get_stats:
                if len(self.keep_rownum) > 0:
                    self.keep_rownum = np.append(self.keep_rownum, np.arange(len(batch_land_pins)+ len(batch_house_pins)))
                else:
                    self.keep_rownum = np.arange(len(batch_land_pins)+ len(batch_house_pins))
                
                if len(self.tr_cv_ts_pins_) > 0:
                    self.tr_cv_ts_pins_ = np.append(self.tr_cv_ts_pins_, np.append(batch_land_pins, batch_house_pins))
                else:
                    self.tr_cv_ts_pins_ = np.append(batch_land_pins, batch_house_pins)
                
                if len(self.tr_cv_ts_bbox_crpd) > 0:
                    self.tr_cv_ts_bbox_crpd = np.append(self.tr_cv_ts_bbox_crpd, np.array(tr_bbox_cropped_arr))
                else:
                    self.tr_cv_ts_bbox_crpd = np.array(tr_bbox_cropped_arr)
                
                if len(self.tr_cv_ts_land_house) > 0:
                    self.tr_cv_ts_land_house = np.append(self.tr_cv_ts_land_house,
                                                    np.append(np.tile('land',len(batch_land_pins)),np.tile('house', len(batch_house_pins))))
                else:
                    self.tr_cv_ts_land_house = np.append(np.tile('land',len(batch_land_pins)),
                                                         np.tile('house',len(batch_house_pins)))
                    
                if len(self.tr_cv_ts_type_info)>0:
                    self.tr_cv_ts_type_info = np.append(self.tr_cv_ts_type_info,
                                                        np.tile('batch_%s' % str(batch_num), element_count * 2))
                else:
                    self.tr_cv_ts_type_info = np.tile('batch_%s' % str(batch_num), element_count * 2)
        
            b = "TOTAL BATCH DONE:  ======== %s"
            print(b % (batch_num), end="\r")
        
            if self.max_batches:
                if self.max_batches == batch_num + 1:
                    break
                    
        logging.info(
                'Training Land Size: %s, Training House Size: %s', str(len(train_land_pins)),
                str(len(train_house_pins)))

    def dumpStratifiedBatches_balanced_class(self, cmn_land_pins, cmn_house_pins, is_cvalid_test=True):

        land_label = 0
        house_label = 1
        label_dict = {'0': 'land', '1': 'house'}
        land_pins = np.sort(np.intersect1d(
            np.array([img.split('.')[0] for img in os.listdir(self.land_image_path) if img != '.DS_Store'],
                     dtype=str),
            cmn_land_pins))

        house_pins = np.sort(np.intersect1d(
            np.array([img.split('.')[0] for img in os.listdir(self.house_image_path) if img != '.DS_Store'],
                     dtype=str),
            cmn_house_pins))

        # RANDOM SHUFFLE LAND AND HOUSE PINS
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(land_pins)
        np.random.seed(self.shuffle_seed + 62)
        np.random.shuffle(house_pins)

        logging.info('Input Data: Total Land: %s, Total House: %s', str(len(land_pins)), str(len(house_pins)))

        self.keep_rownum = []
        self.tr_cv_ts_pins_ = []
        self.tr_cv_ts_land_house = []
        self.tr_cv_ts_type_info = []
        self.tr_cv_ts_bbox_crpd = []
        self.ts_batch_size_per_class = 0
        self.cv_batch_size_per_class = 0
        
        stats_file_name = 'test_new_pins_info.csv'
        if is_cvalid_test:
            logging.info("Training Batch Size = %s", str(self.tr_batch_size))
            logging.info("CrossValidation Batch Size = %s", str(self.cv_batch_size))
            logging.info("Test Batch Size = %s", str(self.ts_batch_size))
            stats_file_name = 'tr_cv_ts_pins_info.csv'

            ##### DUMP TEST AND CROSS-VALIDATION DATA
            self.dump_cv_test_data(land_pins, house_pins, land_label, house_label)

        ##### DUMP TRAINING DATA IN BATCHES
        self.dump_training_batches(land_pins, house_pins, land_label, house_label)

            ##### DUMP TRAIN CV STATISTICS INFO
        if self.get_stats:
            folder_path = os.path.join(self.stats_path, 'prediction_stats')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            dump_pins_path = os.path.join(folder_path, stats_file_name)
            
            print (len(self.keep_rownum), len(self.tr_cv_ts_pins_), len(self.tr_cv_ts_land_house), len(self.tr_cv_ts_type_info),
                                 len(self.tr_cv_ts_bbox_crpd))
            dataOUT = pd.DataFrame(
                np.column_stack((self.keep_rownum, self.tr_cv_ts_pins_, self.tr_cv_ts_land_house, self.tr_cv_ts_type_info,
                                 self.tr_cv_ts_bbox_crpd)),
                columns=['rownum','property_pins', 'property_type', 'dataset_type', 'bbox_cropped'])
            dataOUT.to_csv(dump_pins_path, index=None)

                
        # else:
        #     land_image_paths = [os.path.join(self.land_image_path, pin + '.jpg') for pin in land_pins]
        #     house_image_paths = [os.path.join(self.land_image_path, pin + '.jpg') for pin in house_pins]
        #     logging.info('# Land Images: %s', str(len(land_image_paths)))
        #     logging.info('# House Images: %s', str(len(house_image_paths)))
        #     img_paths = np.append(land_image_paths,house_image_paths)
        #
        #     # for paths in img_paths:
        #     num_batches = int(np.ceil(len(img_paths) / self.ts_batch_size))
        #     for batch_num in range(0, num_batches):
        #         if batch_num != (num_batches - 1):
        #             from_idx = batch_num * self.ts_batch_size
        #             to_idx = (batch_num * self.ts_batch_size) + self.ts_batch_size
        #         else:
        #             from_idx = batch_num * self.ts_batch_size
        #             to_idx = (batch_num * self.ts_batch_size) + (len(img_paths) - (batch_num * self.ts_batch_size))
        #
        #         element_count = to_idx - from_idx
        #
        #         batch_paths = img_paths[from_idx:to_idx]
        #
        #         self.dump_new_data_batches(batch_paths, filename='test_%s' % str(batch_num))

#
# debugg = False
# if debugg:
#     import time
#
#     start_time = time.time()
#
#     cmn_land_pins, cmn_house_pins = get_intersecting_images_pin(is_assessor=False, is_aerial=True, is_streetside=False,is_overlaid=True, is_aerial_cropped=True,equal_proportion=True)
#     print(len(cmn_land_pins), len(cmn_house_pins))
#
#     tr_batch_size = 128
#     ts_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
#     cv_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
#
#     params = dict(
#         image_type='aerial_cropped',
#         img_in_shape=[400, 400, 3],
#         img_out_shape=[224, 224, 3],
#         img_resize_shape=[128, 128, 3],
#         img_crop_shape=[128, 128, 3],
#         tr_batch_size=tr_batch_size,
#         cv_batch_size=cv_batch_size,
#         ts_batch_size=ts_batch_size,
#         enable_rotation=True,
#         shuffle_seed=873,
#         get_stats=True,
#         max_batches=None)
#
#     obj_cb = DumpBatches(params)
#     obj_cb.dumpStratifiedBatches_balanced_class(cmn_land_pins, cmn_house_pins, is_cvalid_test=True)
#
#     print('--------------- %s seconds ------------------' % (time.time() - start_time))
