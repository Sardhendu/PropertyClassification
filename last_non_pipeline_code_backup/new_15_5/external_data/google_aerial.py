from __future__ import division, print_function, absolute_import

import codecs
import json
import logging
import os
import urllib

import numpy as np
import pandas as pd
import requests


from config import pathDict, api_call

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



# map_size = [400,400]
# zoom_lvl = 40
metaURL_head = 'https://maps.googleapis.com/maps/api/geocode/json?address='
aerialURL_head = 'https://maps.googleapis.com/maps/api/staticmap?center='
metaURL_tail = '&key=%s'%(api_call['google_meta_key'])
aerialURL_tail = '&maptype=satellite&key=%s'%(api_call['google_aerial_key'])
reader = codecs.getreader("utf-8")

# Statistic and Image Dump paths



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



class GoogleFetch_AerialMap():
    def __init__(self, params):
        pass
        
    @staticmethod
    def get_latlon_locationtype(address_line, city=None, state=None):
        '''
        :param address_line : '555E, 33rd Place'
        :param city: 'chicago'
        :param state: 'IL'
        :return:
            lat: latitude of the property
            lon: longitude of the property
            location_type = ROOFTOP
            url : The URL used to fetch the meta data information
            
        '''
        address_string = '+'.join([add for add in address_line.split(' ')])
        if city:
            address_string =  address_string + '+' + '+'.join([add for add in city.split(' ')])
        if state:
            address_string = address_string + '+' + state
        
        # print (address_string)
        url = metaURL_head + address_string + metaURL_tail
        r = urllib.request.urlopen(url)
        res_body = r.read()
        content = json.loads(res_body.decode("utf-8"))

        if content['status'] == 'OK':
        # try:
            lat = content['results'][0]['geometry']['location']['lat']
            lon = content['results'][0]['geometry']['location']['lng']
            location_type = content['results'][0]['geometry']['location_type']
            return lat, lon, location_type, url
        elif content['status'] == 'OVER_QUERY_LIMIT':
            return 'EXCEED', 'EXCEED', 'EXCEED', 'EXCEED'
        else:
        # except KeyError:
            logging.info('GET_LATLON: Content lat lon not found')
            return None, None, None, None
            
    @staticmethod
    def get_aerial_image_given_latlon(lat, lon, zoom=19, map_size='400x400'):
        '''
        :param lat: The input latitude
        :param lon: The input Longitude
        :param zoom: The input zoom level
        :param map_size: The input mapSize
        :return:
            : image_data : The image to be saved
            : location_url: The url used to fetch the image
        '''
        location_url = aerialURL_head + str(lat) + ' ' + str(lon) + '&zoom=' + str(zoom) + '&size=' + map_size + \
                     aerialURL_tail
        try:
            img_data = requests.get(location_url).content
            return img_data, location_url

        except:
            logging.info('GET_AERIAL_IMAGE: Response error')
            return None, None



def fetch_dump_google_aerial_images(dataIN, stats_path, batch_size, zoom=19, state='IL', map_size = '400x400',
                                        get_stats=False):
    
    # if which_run == 'latest':
    # new_folder_name = str(time.time()).split('.')[0]
    
    house_dump_path = os.path.join(pathDict['image_path'], 'house')
    land_dump_path = os.path.join(pathDict['image_path'], 'land')
    unknown_dump_path = os.path.join(pathDict['image_path'], 'unknown')


    for dir in [house_dump_path, land_dump_path, unknown_dump_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    data_arr = np.array(dataIN[['pin', 'address_line1', 'address_city', 'indicator']], dtype='str')

    statistics = []
    prev = 0
    # state = 'IL'
    # zoom = 19
    # map_size = '400x400'
    for num, (pin, add1, city, indicator) in enumerate(data_arr):
        # if num
        lat = 'nan'
        lon = 'nan'
        meta_url = 'nan'
        img_url = 'nan'
        location_type = 'nan'
        if str(add1) != 'nan':

            lat, lon, location_type, meta_url = GoogleFetch_AerialMap.get_latlon_locationtype(address_line=add1,
                                                                                         city=city, state=state)
            if lat ==None or lon == None or meta_url == None:
                lat = 'nan'
                lon = 'nan'
                meta_url = 'nan'
            elif lat == 'EXCEED':
                logging.info('Total extraction quota for today EXCEEDS the Free Quota LIMIT')
            else:
                image_data, img_url = GoogleFetch_AerialMap.get_aerial_image_given_latlon(lat=lat, lon=lon,
                                                                                          zoom=zoom, map_size=map_size)
            
                if indicator == "Likely House":
                    with open(os.path.join(house_dump_path, '%s.jpg' % str(pin)), 'wb') as handler:
                        handler.write(image_data)
                elif indicator == 'Likely Land':
                    with open(os.path.join(land_dump_path, '%s.jpg' % str(pin)),'wb') as handler:
                        handler.write(image_data)
                else:
                    with open(os.path.join(unknown_dump_path, '%s.jpg' % str(pin)), 'wb') as handler:
                        handler.write(image_data)
    
        b = "TOTAL RECORDS PARSED: IMAGES DONE ======== %s"
        print(b % (num), end="\r")
    
        if get_stats:
            statistics.append([pin, add1, city, lat, lon, location_type, indicator, meta_url, img_url, ])
    
        if ((num + 1) % batch_size) == 0 or num == len(data_arr) - 1:
            if get_stats:
                file_path = os.path.join(stats_path, '%s_%s.csv' % (prev, num))
                statistics = pd.DataFrame(statistics,
                                          columns=['pin', 'address',  'city', 'lat','lon','loc_type',
                                                   'indicator','meta_url','img_url'])
                statistics.to_csv(file_path, index=None)
                prev = num + 1
            
                statistics = []






        
        
        
        
        
        
###########################  ROUGH RUN


debugg1 = False
if debugg1:
    lat, lon, location_type, url = GoogleFetch_AerialMap.get_latlon_locationtype(address_line='555E 33rd place',
                                                                                 city='chicago', state='IL')
    img, location_url = GoogleFetch_AerialMap.get_aerial_image_given_latlon(lat=lat, lon=lon, zoom=19,
                                                                            map_size='400x400')

debugg = False
if debugg:
    input_path = os.path.join(pathDict['parent_path'], 'house_metadata_nw.csv')
    print (input_path)
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

    metadata = pd.concat([metadata[metadata['indicator'] == 'Likely Land'].head(50),
                          metadata[metadata['indicator'] == 'Likely House'].head(50)])

    fetch_google_aerial_images(dataIN=metadata, zoom=20, state='IL', map_size='400x400', batch_size=10, get_stats=True)
