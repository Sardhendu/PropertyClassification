
import os
import shutil
import json
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


from external_data.google_aerial import fetch_dump_google_aerial_images
from semantic_segmentation.overlay_building_bbox import overlay_parcel_on_images, crop_parcel_from_images


from config import pathDict

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
    def __init__(self, zoom, state, map_size, batch_size):
        self.zoom = zoom
        self.state = state
        self.map_size = map_size
        self.batch_size = batch_size
    
    def parse_dump_aerial(self, land_house_metadata):
        stats_path = os.path.join(pathDict['statistics_path'], 'aerial_collected_data_stats')
        if os.path.exists(stats_path):
            if len(os.listdir(stats_path)) > 0:
                aerial_stats = collateData(filesPath=stats_path, fileType="csv")
                processed_images = aerial_stats.dropna()
                processed_images = processed_images[processed_images['meta_url'] != 'EXCEED']
                print('Number of images already parsed and dumped ', len(aerial_stats))
                
                # Delete the seperate batch files batch files
                shutil.rmtree(stats_path + '/')
                os.makedirs(stats_path)
                
                # Combine the batchfiles into 1 and save
                processed_images = processed_images[
                    ['pin', 'address', 'city', 'lat', 'lon', 'loc_type', 'indicator', 'meta_url', 'img_url']]
                processed_images.to_csv(os.path.join(stats_path, '%s_%s_%s.csv' % (
                    str(time.time()).split('.')[0], str(0), str(len(processed_images) - 1))), index=None)
                land_house_metadata = land_house_metadata[
                    ~land_house_metadata['pin'].isin(np.array(processed_images['pin'], dtype=str))]
        else:
            os.makedirs(stats_path)
        
        land_house_metadata = shuffle(land_house_metadata)
        print('Number of new images to be parsed and dumped ', len(land_house_metadata))
        fetch_dump_google_aerial_images(land_house_metadata, stats_path, zoom=self.zoom, state=self.state,
                                                 map_size=self.map_size, batch_size=self.batch_size, get_stats=True,)



class GetOverlayed():
    '''
        Note : In the parse_dump_overlayed function we hard code the selection process i.e the image should be
        roof-top, ans should be from Chicago because we have property shape files only for Chicago region
        Since overlayed are created using Aerial Images we take reference of that directory

        * Now that we have Google Aerial Images metadata, we perform the below operation to do some supposedly "bad data" removal
            * A image should have a valid address_line1 and city. The address should not start with '0 0 GROVE AVE', '0 SUMMIT ST', '0 VACANT PROPERTY'
            * The location_type should not be "RANGE_INTERPOLATED"
            * The Lat and Lon should have a 4 or more than 4 decimal precision.
    '''
    def __init__(self):
        self.overlayed_image_path = os.path.join(pathDict['input_image_run_dir'], 'overlayed')
        self.aerial_image_path = os.path.join(pathDict['input_image_run_dir'], 'aerial')
        
        self.stats_path = os.path.join(pathDict['general_stats_path'], 'aerial', 'aerial_collected_data_stats')
        
        if not os.path.exists(self.stats_path) or not os.path.exists(self.aerial_image_path):
            raise ValueError("Is seems you haven't, dumped the Aerial Images, fetching Aerial image is a prerequisite "
                             "to creating overlayed images ", )

    def parse_dump_overlayed(self):
        aerial_stats_data = collateData(self.stats_path)
        # print('Number of Images to be processed = ', len(aerial_stats_data))
    
        data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
            aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
            'address'].str.lower().str.strip().str.match('0'))]
        # print('Shape: Images for Overlaying: ', data_to_model.shape)
        overlay_parcel_on_images(data_to_model, self.aerial_image_path, self.overlayed_image_path, zoom=20,
                                 map_size=[400, 400])


class GetAerialCropped():
    '''
        Note : In the parse_dump_overlayed function we hard code the selection process i.e the image should be
        roof-top, ans should be from Chicago because we have property shape files only for Chicago region
        Since overlayed are created using Aerial Images we take reference of that directory
        
        * Now that we have Google Aerial Images metadata, we perform the below operation to do some supposedly "bad data" removal
            * A image should have a valid address_line1 and city. The address should not start with '0 0 GROVE AVE', '0 SUMMIT ST', '0 VACANT PROPERTY'
            * The location_type should not be "RANGE_INTERPOLATED"
            * The Lat and Lon should have a 4 or more than 4 decimal precision.
    '''
    
    def __init__(self):
        self.aerial_cropped_img_path = os.path.join(pathDict['input_image_run_dir'], 'aerial_cropped')
        self.aerial_image_path = os.path.join(pathDict['input_image_run_dir'], 'aerial')

        self.stats_path = os.path.join(pathDict['general_stats_path'], 'aerial', 'aerial_collected_data_stats')
        
        if not os.path.exists(self.stats_path) or not os.path.exists(self.aerial_image_path):
            raise ValueError("Is seems you haven't, dumped the Aerial Images, fetching Aerial image is a prerequite "
                             "to creating overlayed images ", )
    
    def parse_dump_aerial_cropped(self):
        aerial_stats_data = collateData(self.stats_path)
        # print('Number of Images to be processed = ', len(aerial_stats_data))
        
        data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
            aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
            'address'].str.lower().str.strip().str.match('0'))]
        # print('Shape: Images for Overlaying: ', data_to_model.shape)
        crop_parcel_from_images(data_to_model, self.aerial_image_path, self.aerial_cropped_img_path, zoom=20,
                                 map_size=[400, 400])
