

import codecs
import json
import logging
import os
import re
import urllib

import numpy as np
import pandas as pd
import requests


from config import pathDict, api_call

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


map_size = [400,400]
zoom_lvl = 40
location_url = 'https://maps.googleapis.com/maps/api/streetview?'
street_tail = '&heading=235&pitch=10&key=%s'%(api_call['google_streetside_key'])
reader = codecs.getreader("utf-8")

class GoogleMapSearch():
    def __init__(self, params):
        if 'address_line' and 'state' in params.keys():
            self.location =  params['address_line'] + ', ' + params['state']
        
        elif 'lat' and 'lon' in params.keys():
            self.location = params['lat'] + ', ' + params['lon']
            
    def get_street_view_image(self):
        url = location_url + 'size=%sx%s'%(map_size[0], map_size[1]) + \
              '&location=%s'%self.location + \
              '&fov=%s'%(zoom_lvl) + street_tail

        image_data = requests.get(url).content
        return image_data, url
    
    
def get_image_address(dataIN, batch_size,  get_stats=False):
    
    # STATS DUMP PATH
    folder_path = os.path.join(pathDict['statistics_path'],
                               'streetside_images', 'data_loader')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    
    data_arr = np.array(dataIN[['pin', 'address_line1', 'address_city', 'address_zip', 'indicator']], dtype='str')
    # print(data_arr[0:5, :])

    params = {}
    statistics = []

    params['country'] = 'US'
    params['state'] = 'IL'

    prev = 0
    for num, (pin, add1, city, zip, indicator) in enumerate(data_arr):
        p = pin
        add = 'nan'
        img_url = 'nan'
        if str(add1) != 'nan':
            params['address_line'] = add1
            add = add1

            obj_st = GoogleMapSearch(params=params)
            image_data, img_url = obj_st.get_street_view_image()

            if indicator == "Likely House":
                with open(os.path.join(pathDict['streetside_image_path'], 'house', '%s.jpg' % str(pin)),
                          'wb') as handler:
                    handler.write(image_data)
            elif indicator == 'Likely Land':
                with open(os.path.join(pathDict['streetside_image_path'], 'land', '%s.jpg' % str(pin)),
                          'wb') as handler:
                    handler.write(image_data)
            else:
                with open(os.path.join(pathDict['streetside_image_path'], 'unknown', '%s.jpg' % str(pin)),
                          'wb') as handler:
                    handler.write(image_data)
        
        b = "TOTAL RECORDS PARSED: IMAGES DONE ======== %s"
        print(b % (num), end="\r")
        
        if get_stats:
            statistics.append([pin,city,zip,add1,indicator, img_url])
            
        if ((num+1)%batch_size) == 0 or num ==len(data_arr)-1:
            if get_stats:
                file_path = os.path.join(folder_path, '%s_%s.csv' % (prev, num))
                statistics = pd.DataFrame(statistics,
                                          columns=['pin', 'city','zip','address', 'indicator','img_url'])
                statistics.to_csv(file_path, index=None)
                prev = num + 1
    
                statistics = []
        
            
            

def metadata_prep(metadata):
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
    metadata['last_reviewed_timestamp'] = metadata['last_reviewed_timestamp'].astype('str')  # .astype('str')
    metadata['gone_timestamp'] = metadata['gone_timestamp'].astype('str')
    metadata['indicator'] = metadata['indicator'].astype('str')
    metadata['assessor_photo'] = metadata['assessor_photo'].astype('str')
    return metadata
    
debugg = False
if debugg:
    input_path = os.path.join(pathDict['parent_path'], 'house_metadata_nw.csv')

    metadata = pd.read_csv(input_path)
    logging.info('Metadata shape: %s', str(metadata.shape))

    metadata = metadata_prep(metadata)

    # Remove Test Data set
    metadata = metadata[metadata['removed'] == 0]
    logging.info('Metadata after removing test data set shape: %s', str(metadata.shape))

    # Remove data where the last_reviewed_timestamp column doesn't have a valid timestamp
    metadata = metadata[metadata['last_reviewed_timestamp'] != 'nan']
    logging.info('Metadata after retaining last_reviewed_timestamp: %s', str(metadata.shape))
    logging.info('Metadata Head: \n %s', str(metadata.head()))
    # print (metadata)

    get_image_address(metadata, batch_size=5)

 
#
# https://maps.googleapis.com/maps/api/streetview?size=400x400&location=555 E, 33rd place chicago, IL
# &fov=90&heading=235&pitch=10
# &key=AIzaSyC3X54Yp0C8xXlQSSCdkCjgVjR0ji3A8UE
