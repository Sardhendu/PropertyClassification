import os
dir_path  =  os.path.abspath(os.path.join(__file__ ,"../..")) # Moves one level up in the directory

import sys
sys.path.append(dir_path)

import logging
import time
import json
import numpy as np
import pandas as pd
import shutil
import random
from sklearn.utils import shuffle

from src.config import get_config
from src.external_data.google_aerial import fetch_dump_google_aerial_images
from src.semantic_segmentation.overlay_building_bbox import overlay_parcel_on_images, crop_parcel_from_images


def collateData(filesPath, fileType="csv"):
    fileDirList = [os.path.join(filesPath, files) for files in
                   os.listdir(filesPath) if files.endswith('.%s' % fileType)]
    if fileType == "csv":
        rowList = []
        # fileDirList = [files for files in listdir if files.endswith('.csv')]
        for files in fileDirList:
            df = pd.read_csv(files, header=0)
            rowList.append(df)
        return pd.concat(rowList)
    elif fileType == "json":
        dataOUT = {}
        for files in fileDirList:
            jsonFilepath = os.path.join(filesPath, files)
            with open(jsonFilepath, 'r') as fileIN:
                dataOUT = merge(dataOUT, json.load(fileIN))
        return dataOUT


class GetAerial():
    def __init__(self, conf, zoom, state, map_size, batch_size):
        self.conf = conf
        self.zoom = zoom
        self.state = state
        self.map_size = map_size
        self.batch_size = batch_size

        self.inp_img_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial')
        self.stats_path = os.path.join(conf['pathDict']['general_stats_path'], 'aerial', 'aerial_collected_data_stats')
        
        logging.info('Initiating Aerial Dump ...............................')
        logging.info('Input data path: %s', str(self.inp_img_path))
        logging.info('Stats dump path: %s', str(self.stats_path))

    def parse_dump_aerial(self, land_house_metadata):
        
        if os.path.exists(self.stats_path):
            if len(os.listdir(self.stats_path)) > 0:
                aerial_stats = collateData(filesPath=self.stats_path, fileType="csv")
                processed_images = aerial_stats.dropna()
                processed_images = processed_images[processed_images['meta_url'] != 'EXCEED']
                logging.info('Number of images already parsed and dumped %s', str(len(aerial_stats)))

                # Delete the seperate batch files batch files
                shutil.rmtree(self.stats_path + '/')
                os.makedirs(self.stats_path)

                # Combine the batchfiles into 1 and save
                processed_images = processed_images[
                    ['pin', 'address', 'city', 'lat', 'lon', 'loc_type', 'indicator', 'meta_url', 'img_url']]
                processed_images.to_csv(os.path.join(self.stats_path, '%s_%s_%s.csv' % (
                    str(time.time()).split('.')[0], str(0), str(len(processed_images) - 1))), index=None)
                land_house_metadata = land_house_metadata[
                    ~land_house_metadata['pin'].isin(np.array(processed_images['pin'], dtype=str))]
        else:
            os.makedirs(self.stats_path)

        land_house_metadata = shuffle(land_house_metadata)
        logging.info('Number of new images to be parsed and dumped: %s', str(len(land_house_metadata)))
        fetch_dump_google_aerial_images(self.conf, land_house_metadata,
                                        self.inp_img_path, self.stats_path,
                                        zoom=self.zoom, state=self.state,
                                        map_size=self.map_size,
                                        batch_size=self.batch_size, get_stats=True)

        return 'COMPLETE'


class GetOverlaid():
    '''
        Note : In the parse_dump_overlaid function we hard code the selection process i.e the image should be
        roof-top, ans should be from Chicago because we have property shape files only for Chicago region
        Since overlaid are created using Aerial Images we take reference of that directory

        * Now that we have Google Aerial Images metadata, we perform the below operation to do some supposedly "bad
        data" removal
            * A image should have a valid address_line1 and city. The address should not start with '0 0 GROVE AVE',
            '0 SUMMIT ST', '0 VACANT PROPERTY'
            * The location_type should not be "RANGE_INTERPOLATED"
            * The Lat and Lon should have a 4 or more than 4 decimal precision.
    '''
    
    def __init__(self, conf):
        self.conf = conf
        self.overlaid_image_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'overlaid')
        self.aerial_image_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial')
        self.stats_path = os.path.join(conf['pathDict']['general_stats_path'], 'aerial', 'aerial_collected_data_stats')
        
        if not os.path.exists(self.stats_path) or not os.path.exists(self.aerial_image_path):
            raise ValueError("Is seems you haven't, dumped the Aerial Images, fetching Aerial image is a prerequisite "
                             "to creating overlaid images ", )

        logging.info('Initiating Aerial Overlaid ...............................')
        logging.info('Input data path: %s', str(self.aerial_image_path))
        logging.info('Output data path: %s', str(self.overlaid_image_path))
    
    def parse_dump_overlaid(self):
        aerial_stats_data = collateData(self.stats_path)
        # print('Number of Images to be processed = ', len(aerial_stats_data))
        
        data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
            aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
            'address'].str.lower().str.strip().str.match('0'))]
        # print('Shape: Images for Overlaying: ', data_to_model.shape)
        overlay_parcel_on_images(self.conf, data_to_model, self.aerial_image_path, self.overlaid_image_path,
                                 zoom=20, map_size=[400, 400])

        return 'COMPLETE'


class GetAerialCropped():
    '''
        Note : In the parse_dump_overlaid function we hard code the selection process i.e the image should be
        roof-top, ans should be from Chicago because we have property shape files only for Chicago region
        Since overlaid are created using Aerial Images we take reference of that directory

        * Now that we have Google Aerial Images metadata, we perform the below operation to do some supposedly "bad
        data" removal
            * A image should have a valid address_line1 and city. The address should not start with '0 0 GROVE AVE',
            '0 SUMMIT ST', '0 VACANT PROPERTY'
            * The location_type should not be "RANGE_INTERPOLATED"
            * The Lat and Lon should have a 4 or more than 4 decimal precision.
    '''
    
    def __init__(self, conf):
        self.conf = conf
        self.aerial_cropped_img_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial_cropped')
        self.aerial_image_path = os.path.join(conf['pathDict']['input_image_run_dir'], 'aerial')
        
        self.stats_path = os.path.join(conf['pathDict']['general_stats_path'], 'aerial', 'aerial_collected_data_stats')
        
        if not os.path.exists(self.stats_path) or not os.path.exists(self.aerial_image_path):
            raise ValueError("Is seems you haven't, dumped the Aerial Images, fetching Aerial image is a prerequite "
                             "to creating overlaid images ", )

        logging.info('Initiating Aerial Crop ...............................')
        logging.info('Input data path: %s', str(self.aerial_image_path))
        logging.info('Output data path: %s', str(self.aerial_cropped_img_path))
        
    
    def parse_dump_aerial_cropped(self, condition_apply=False):
        aerial_stats_data = collateData(self.stats_path)
        # print('Number of Images to be processed = ', len(aerial_stats_data))
        
        if condition_apply:
            logging.info('Apply Condition: RANGE_INTERPOLATED and CHICAGO')
            data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
                aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
                'address'].str.lower().str.strip().str.match('0'))]
        else:
            data_to_model = aerial_stats_data[~aerial_stats_data['address'].str.lower().str.strip().str.match('0')]
        # print('Shape: Images for Overlaying: ', data_to_model.shape)
        crop_parcel_from_images(self.conf, data_to_model, self.aerial_image_path, self.aerial_cropped_img_path,
                                zoom=20, map_size=[400, 400])
        return 'COMPLETE'


def data_selection_cond(conf, cond_dict, your_csv_file_path):
    metadata = pd.read_csv(your_csv_file_path)
    metadata.columns = ['row_id', 'removed', 'property_id', 'state', 'county_name', 'pin',
                        'address_line1', 'address_line2', 'address_city', 'address_zip',
                        'zoning', 'improvement_level', 'type', 'exterior',
                        'last_reviewed_timestamp', 'gone_timestamp', 'indicator',
                        'assessor_photo']
    metadata['state'] = metadata['state'].astype('str')
    metadata['county_name'] = metadata['county_name'].astype('str')
    metadata['pin'] = metadata['pin'].astype('str')
    metadata['address_line1'] = metadata['address_line1'].astype('str')
    metadata['address_line2'] = metadata['address_line2'].astype('str')
    metadata['address_city'] = metadata['address_city'].astype('str')
    metadata['address_zip'] = metadata['address_zip'].astype('str')
    metadata['zoning'] = metadata['zoning'].astype('str')
    metadata['improvement_level'] = metadata['improvement_level'].astype('str')
    metadata['type'] = metadata['type'].astype('str')
    metadata['exterior'] = metadata['exterior'].astype('str')
    metadata['last_reviewed_timestamp'] = pd.to_datetime(metadata['last_reviewed_timestamp'])  # .astype('str')
    metadata['gone_timestamp'] = pd.to_datetime(metadata['gone_timestamp'])  # .astype('str')
    metadata['indicator'] = metadata['indicator'].astype('str')
    metadata['assessor_photo'] = metadata['assessor_photo'].astype('str')
    
    # Remove the Test Dat from Metadata
    metadata = metadata[metadata['removed'] == 0]
    print ('Shape: Metadata table', str(metadata.shape))

    property_metadata = metadata
    if cond_dict['last_reviewed_ts']:
        property_metadata = property_metadata[property_metadata['last_reviewed_timestamp'] != '']
        property_metadata['last_reviewed_timestamp'] = pd.to_datetime(
                property_metadata['last_reviewed_timestamp'])
        print('Data shape  = %s' % (str(property_metadata.shape)))
        if cond_dict['from_year'] and cond_dict['to_year']:
            print ('Applying conditions: from_year %s - to_year %s'%(str(cond_dict['from_year']), str(cond_dict['from_year'])))
            property_metadata = property_metadata[
                ((property_metadata['last_reviewed_timestamp'].dt.year >= cond_dict['from_year']) &
                 (property_metadata['last_reviewed_timestamp'].dt.year < cond_dict['to_year']))]
            print('Data shape  = %s' % (str(property_metadata.shape)))
        
        if cond_dict['from_month'] and cond_dict['to_month']:
            print('Applying conditions: from_month %s - to_month %s' % (
            str(cond_dict['from_month']), str(cond_dict['from_month'])))
            property_metadata = property_metadata[
            ((property_metadata['last_reviewed_timestamp'].dt.month >= cond_dict['from_month']) &
            (property_metadata['last_reviewed_timestamp'].dt.month < cond_dict['to_month']))]
            print('Data shape  = %s' % (str(property_metadata.shape)))
    
    if cond_dict['which_city']:
        print('Applying conditions: city: %s' % (str(cond_dict['which_city'])))
        property_metadata = property_metadata[
            (property_metadata['address_city'].str.lower().str.strip().str.match(cond_dict['which_city']))]
        print('Data shape  = %s' % (str(property_metadata.shape)))
    
    property_metadata = property_metadata[
        (property_metadata['address_line1'] != 'nan') &
        (property_metadata['indicator'].isin(["Likely House", "Likely Land"]))]
    
    if cond_dict['use_improvement_lvl']:
        print('Applying conditions: improvement_lvl: True')
        land_data = property_metadata[(property_metadata['indicator'] == 'Likely Land') & (property_metadata['improvement_level'] =='Land')]
        
        house_data = property_metadata[(property_metadata['indicator'] == 'Likely House') & (property_metadata['improvement_level'] != 'Land')]
        print('Data shape land = %s, house = %s' % (str(land_data.shape), str(house_data.shape)))
    else:
        land_data = property_metadata[(property_metadata['indicator'] == 'Likely Land')]
    
        house_data = property_metadata[(property_metadata['indicator'] == 'Likely House')]
        

    if cond_dict['max_num_records']:
        random.seed(481)
        n = cond_dict['max_num_records']
        print('Applying conditions: max_num_records: %s'%(n))
        land_data = land_data.reset_index().drop('index', axis=1)
        house_data = house_data.reset_index().drop('index', axis=1)
        
        if len(land_data) >= n//2:
            lindex = np.array(random.sample(list(range(len(land_data))), n // 2))
            land_data = land_data.ix[lindex]
        if len(house_data) >= n//2:
            rindex = np.array(random.sample(list(range(len(house_data))), n // 2))
            house_data = house_data.ix[rindex]
        
    print('Shape: Land Property table %s'%(str(land_data.shape)))
    print('Shape: House Property table %s'%(str(house_data.shape)))
    
    filtered_data = pd.concat([land_data, house_data]).reset_index().drop('index', axis=1)

    filtered_data.to_csv(os.path.join(conf['pathDict']['csv_path'], 'input_data.csv'), index=None)
    
    return 'COMPLETE'

# cond_dict = dict(last_reviewed_ts=True,
#                  from_year=2015, to_year=2015,
#                  from_month=5, to_month=6,
#                  which_city='chicago',
#                  use_improvement_lvl = True,
#                  max_num_records=200)
#
# conf = get_config(which_run='new_new', img_type='aerial_cropped')
# a  = data_selection_cond(conf, cond_dict)
# print (a)


def create_csv_file_for_input(**kwargs):
    inp_params = kwargs['params']
    
    if 'input_csv_path' not in inp_params.keys():
        raise ValueError('You should provide the path to input csv file')
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')
    
    if 'cond_dict' not in inp_params.keys():
        raise ValueError('You should provide the conditions to filter data for testing')
    
    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])
    
    status = data_selection_cond(conf, inp_params['cond_dict'], inp_params['input_csv_path'])
    
    return status


def dump_aerial(**kwargs):
    inp_params = kwargs['params']
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])
    
    land_house_data = pd.read_csv(os.path.join(conf['pathDict']['csv_path'], 'input_data.csv'))
    
    status = GetAerial(conf, zoom=20, state='IL', map_size = '400x400', batch_size=100).parse_dump_aerial(
            land_house_data)
    return status


def dump_aerial_cropped(**kwargs):
    inp_params = kwargs['params']
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])
    obj_cropped = GetAerialCropped(conf)
    status = obj_cropped.parse_dump_aerial_cropped(condition_apply=True)
    return status

def dump_overlaid(**kwargs):
    inp_params = kwargs['params']
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')
    
    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])
    obj_overlaid = GetOverlaid(conf)
    status = obj_overlaid.parse_dump_overlaid()
    return status



#
#
#
# def data_selection_cond1(conf):
#     parent_path = conf['pathDict']['parent_path']
#     meta_data_path = os.path.join(parent_path, "house_metadata_nw.csv")
#     metadata = pd.read_csv(meta_data_path)
#     metadata.columns = ['row_id', 'removed', 'property_id', 'state', 'county_name', 'pin',
#                         'address_line1', 'address_line2', 'address_city', 'address_zip',
#                         'zoning', 'improvement_level', 'type', 'exterior',
#                         'last_reviewed_timestamp', 'gone_timestamp', 'indicator',
#                         'assessor_photo']
#     metadata['state'] = metadata['state'].astype('str')
#     metadata['county_name'] = metadata['county_name'].astype('str')
#     metadata['pin'] = metadata['pin'].astype('str')
#     metadata['address_line1'] = metadata['address_line1'].astype('str')
#     metadata['address_line2'] = metadata['address_line2'].astype('str')
#     metadata['address_city'] = metadata['address_city'].astype('str')
#     metadata['address_zip'] = metadata['address_zip'].astype('str')
#     metadata['zoning'] = metadata['zoning'].astype('str')
#     metadata['improvement_level'] = metadata['improvement_level'].astype('str')
#     metadata['type'] = metadata['type'].astype('str')
#     metadata['exterior'] = metadata['exterior'].astype('str')
#     metadata['last_reviewed_timestamp'] = pd.to_datetime(metadata['last_reviewed_timestamp'])  # .astype('str')
#     metadata['gone_timestamp'] = pd.to_datetime(metadata['gone_timestamp'])  # .astype('str')
#     metadata['indicator'] = metadata['indicator'].astype('str')
#     metadata['assessor_photo'] = metadata['assessor_photo'].astype('str')
#
#     # Remove the Test Dat from Metadata
#     metadata = metadata[metadata['removed'] == 0]
#     logging.info('Shape: Metadata table %s', str(metadata.shape))
#
#     property_metadata = metadata[metadata['last_reviewed_timestamp'] != '']
#     property_metadata['last_reviewed_timestamp'] = pd.to_datetime(
#             property_metadata['last_reviewed_timestamp'])
#     property_metadata = property_metadata[
#         ((property_metadata['last_reviewed_timestamp'].dt.year == 2015) &
#          (property_metadata['last_reviewed_timestamp'].dt.month >= 5) &
#          (property_metadata['last_reviewed_timestamp'].dt.month < 7) &
#          (property_metadata['address_city'].str.lower().str.strip().str.match('chicago')) &
#          (property_metadata['address_line1'] != 'nan') &
#          (property_metadata['assessor_photo'] != 'nan') &
#          (property_metadata['indicator'].isin(["Likely House", "Likely Land"])
#           ))
#     ]
#
#     land_data = property_metadata[
#         (property_metadata['indicator'] == 'Likely Land') & (property_metadata['improvement_level'] ==
#                                                              'Land')]
#     logging.info('Shape: Land Property table %s', str(land_data.shape))
#     house_data = property_metadata[
#         (property_metadata['indicator'] == 'Likely House') & (property_metadata['improvement_level'] != 'Land')]
#     logging.info('Shape: House Property table %s', house_data.shape)
#
#     return pd.concat([land_data, house_data])

#
# from src.config import get_config
# conf = get_config('jordan', 'aerial_cropped')
# out = data_selection_cond1(conf)
# print (out)