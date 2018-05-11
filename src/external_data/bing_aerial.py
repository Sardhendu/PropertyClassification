

import codecs
import json
import logging
import os
import re
import urllib

import numpy as np
import pandas as pd
import requests

from src.config import pathDict, api_call

reader = codecs.getreader("utf-8")

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

map_size = [256,256]
location_url = 'https://dev.virtualearth.net/REST/v1/Locations/'
loc_tail = '?o=json&key=' + api_call['bing_key']

aerial_url = 'https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Aerial/'
aer_tail = '&o=json&key=' + api_call['bing_key']



# Statistic and Image Dump paths
bing_aerial_stats_path = pathDict['bing_aerial_stats_path']
bing_house_dump_path = os.path.join(pathDict['bing_aerial_image_path'], 'house')
bing_land_dump_path = os.path.join(pathDict['bing_aerial_image_path'], 'land')
bing_unknown_dump_path = os.path.join(pathDict['bing_aerial_image_path'], 'unknown')

for dir in [bing_aerial_stats_path, bing_house_dump_path, bing_land_dump_path, bing_unknown_dump_path]:
    if not os.path.exists(dir):
        os.makedirs(dir)


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



class BingSearch():
    def __init__(self, params):
        self.address_stack = []
        if 'country' in params.keys():
            self.address_stack.append(re.sub(' ', '%20', params['country'].strip()))
        
        if 'state' in params.keys():
            self.address_stack.append(re.sub(' ', '%20', params['state'].strip()))
        
        if 'locality' in params.keys():
            self.address_stack.append(re.sub(' ', '%20', params['locality'].strip()))
        
        if 'postal' in params.keys():
            self.address_stack.append(re.sub(' ', '%20', params['postal'].strip()))
        
        if 'address_line' in params.keys():
            self.address_stack.append(re.sub(' ', '%20', params['address_line'].strip()))
    
    def get_latlon(self):
        address_string = '/'.join([add for add in self.address_stack])

        fetch_url = location_url + address_string + loc_tail
        # logging.info('lat_lon fetch URL: %s', str(fetch_url))
        try:
            r = urllib.request.urlopen(fetch_url)
            res_body = r.read()
            content = json.loads(res_body.decode("utf-8"))
            
            try:
                lat,lon = content['resourceSets'][0]['resources'][0]['point']['coordinates']
                return lat, lon, fetch_url
            except KeyError:
                logging.info('GET_LATLON: Content lat lon not found')
                return None, None, None
        except:
            logging.info('GET_LATLON: Response error')
            return None, None, None

    # url = aerial_url + \
    #       '%s,%s' % (lat, lon) + \
    #       '?zl=%s' % str(zoom_lvl) + \
    #       '?mapSize=%s,%s' % (str(map_size[0]), str(map_size[1])) + aer_tail
    def get_aerial_image(self, lat, lon):
        
        zoom_lvl = 20   # The maximum zoom level provided Bing Map

        while zoom_lvl >= 18:
            # logging.info('Running with Zoom: %s', str(zoom_lvl))
            url = aerial_url + \
                  '%s,%s' % (lat, lon) + \
                  '/%s' % str(zoom_lvl) + \
                  '?mapSize=%s,%s' %(str(map_size[0]), str(map_size[1])) + aer_tail
            # logging.info('aerial fetch URL: %s', str(url))
            try:
                r = urllib.request.urlopen(url)
                res_body = r.read()
                content = json.loads(res_body.decode("utf-8"))

                # The zoom level is not sme for all the places, when a particular zoom level is not available then the big API returns garbage values. In order to overcome this we check i vintageEnd r vintageStart keys are present in the dictionary if so then the image is nt a garbage, if not then we reduce the zoom level further and try out.
                if content['resourceSets'][0]['resources'][0]['vintageEnd']:
                    image_url = content['resourceSets'][0]['resources'][0]['imageUrl']
                    img_data = requests.get(image_url).content
                    return img_data, zoom_lvl, image_url
                else:
                    zoom_lvl -= 1

            except:
                logging.info('GET_AERIAL_IMAGE: Response error')
                return None, None, None
        return None, None, None
        
        

        


def get_image_address(dataIN, batch_size, get_stats=False):
    data_arr = np.array(dataIN[['pin', 'address_line1', 'address_city', 'address_zip', 'indicator']], dtype='str')
    print (data_arr[0:5,:])
    
    params = {}
    statistics = []

    params['country'] = 'US'
    params['state'] = 'IL'
    
    prev = 0
    for num, (pin, add1, city, zip, indicator) in enumerate(data_arr):
        # if num < 500:
        #     continue
        logging.info('Running Record number: %s', str(num))
        try:
            zip = str(int(float(zip))) if str(zip)!='nan' else 'aa'
        except ValueError:
            zip = '0'
        
        p = pin
        cty = 'nan'
        zp = 'nan'
        add = 'nan'
        lt = 'nan'
        ln = 'nan'
        zl = 'nan'
        img = 'nan'
        meta_url = 'nan'
        img_url = 'nan'
        
        if str(city) != 'nan':
            params['locality'] = city
            cty = city
          
        if len(zip) == 5:
            params['postal'] = zip
            zp = zip

        if str(add1) != 'nan':
            params['address_line'] = add1
            add = add1
            
            obj_BS = BingSearch(params)
            lat, lon, meta_url = obj_BS.get_latlon()

            lt = lat if lat else 'nan'
            ln = lon if lon else 'nan'
            
            if lt != 'nan':
                image_data, zoom_lvl, img_url = obj_BS.get_aerial_image(lat, lon)
                zl = zoom_lvl
    
                img = 'yes' if zoom_lvl else 'nan'

                if img == 'yes':
                    if indicator == "Likely House":
                        with open(os.path.join(bing_house_dump_path, '%s.jpg' % str(pin)), 'wb') as handler:
                            handler.write(image_data)
                    elif indicator == 'Likely Land':
                        with open(os.path.join(bing_land_dump_path, '%s.jpg' % str(pin)), 'wb') as handler:
                            handler.write(image_data)
                    else:
                        with open(os.path.join(bing_unknown_dump_path, '%s.jpg' % str(pin)), 'wb') as handler:
                            handler.write(image_data)
        
        if get_stats:
            statistics.append([p,cty,add,lt,ln,zl,img,indicator, meta_url, img_url])

        b = "TOTAL RECORDS PARSED: IMAGES DONE ======== %s"
        print(b % (num), end="\r")
        
        if ((num+1)%batch_size) == 0 or num ==len(data_arr)-1:
            
            if get_stats:
                file_path = os.path.join(bing_aerial_stats_path, '%s_%s.csv'%(prev, num))
                statistics = pd.DataFrame(statistics,
                                          columns=['pin', 'city', 'address',
                                                   'lat', 'lon','zoom_lvl', 'image_fetch',
                                                   'indicator', 'meta_url','img_url'])

                statistics.to_csv(file_path, index=None)
                prev = num + 1
    
                statistics = []

        # if num == 50:
        #     break

#
#
# debugg = False
# if debugg:
#     input_path = os.path.join(pathDict['parent_path'], 'house_metadata_nw.csv')
#
#     metadata = pd.read_csv(input_path)
#     logging.info('Metadata shape: %s', str(metadata.shape))
#
#     metadata = metadata_prep(metadata)
#
#     # Remove Test Data set
#     metadata = metadata[metadata['removed'] == 0]
#     logging.info('Metadata after removing test data set shape: %s', str(metadata.shape))
#
#     # Remove data where the last_reviewed_timestamp column doesn't have a valid timestamp
#     metadata = metadata[metadata['last_reviewed_timestamp'] != 'nan']
#     logging.info('Metadata after retaining last_reviewed_timestamp: %s', str(metadata.shape))
#     logging.info('Metadata Head: \n %s', str(metadata.head()))
#
#
#
#     get_image_address(metadata, batch_size=10)
#
