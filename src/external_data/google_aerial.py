from __future__ import division, print_function, absolute_import

import codecs
import json
import logging
import os
import urllib

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



class GoogleFetch_AerialMap():
    def __init__(self, conf):
        self.metaURL_head = 'https://maps.googleapis.com/maps/api/geocode/json?address='
        self.aerialURL_head = 'https://maps.googleapis.com/maps/api/staticmap?center='
        self.metaURL_tail = '&key=%s' % (conf['api_call']['google_meta_key'])
        self.aerialURL_tail = '&maptype=satellite&key=%s' % (conf['api_call']['google_aerial_key'])
        self.reader = codecs.getreader("utf-8")
        
    def get_latlon_locationtype(self, address_line, city=None, state=None):
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
        url = self.metaURL_head + address_string + self.metaURL_tail
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
            
    def get_aerial_image_given_latlon(self, lat, lon, zoom=19, map_size='400x400'):
        '''
        :param lat: The input latitude
        :param lon: The input Longitude
        :param zoom: The input zoom level
        :param map_size: The input mapSize
        :return:
            : image_data : The image to be saved
            : location_url: The url used to fetch the image
        '''
        location_url = self.aerialURL_head + str(lat) + ' ' + str(lon) + '&zoom=' + str(zoom) + '&size=' + map_size + \
                     self.aerialURL_tail
        try:
            img_data = requests.get(location_url).content
            return img_data, location_url

        except:
            logging.info('GET_AERIAL_IMAGE: Response error')
            return None, None



def fetch_dump_google_aerial_images(conf, dataIN, inp_img_path, stats_path, batch_size,
                                    zoom=19, state='IL', map_size ='400x400', get_stats=False):
    
    # if which_run == 'latest':
    # new_folder_name = str(time.time()).split('.')[0]
    
    house_dump_path = os.path.join(inp_img_path, 'house')
    land_dump_path = os.path.join(inp_img_path, 'land')
    unknown_dump_path = os.path.join(inp_img_path, 'unknown')


    for dir in [house_dump_path, land_dump_path, unknown_dump_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    data_arr = np.array(dataIN[['pin', 'address_line1', 'address_city', 'indicator']], dtype='str')

    statistics = []
    prev = 0
    # state = 'IL'
    # zoom = 19
    # map_size = '400x400'
    obj_GFAM = GoogleFetch_AerialMap(conf)
    for num, (pin, add1, city, indicator) in enumerate(data_arr):
        # if num
        lat = 'nan'
        lon = 'nan'
        meta_url = 'nan'
        img_url = 'nan'
        location_type = 'nan'
        if str(add1) != 'nan':

            lat, lon, location_type, meta_url = obj_GFAM.get_latlon_locationtype(address_line=add1,
                                                                                         city=city, state=state)
            if lat ==None or lon == None or meta_url == None:
                lat = 'nan'
                lon = 'nan'
                meta_url = 'nan'
            elif lat == 'EXCEED':
                logging.info('Total extraction quota for today EXCEEDS the Free Quota LIMIT, HENCE STOPPING')
                break
            else:
                image_data, img_url = obj_GFAM.get_aerial_image_given_latlon(lat=lat, lon=lon,
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
            logging.info('Total images Parsed/Dumped: %s', str(num))
            if get_stats:
                logging.info('Dumping statistics %s at %s', str('%s_%s.csv' % (prev, num)),str(stats_path))
                file_path = os.path.join(stats_path, '%s_%s.csv' % (prev, num))
                statistics = pd.DataFrame(statistics,
                                          columns=['pin', 'address',  'city', 'lat','lon','loc_type',
                                                   'indicator','meta_url','img_url'])
                statistics.to_csv(file_path, index=None)
                prev = num + 1
            
                statistics = []
