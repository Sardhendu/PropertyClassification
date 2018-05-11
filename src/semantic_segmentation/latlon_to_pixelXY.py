from __future__ import division, print_function, absolute_import

import math
import numpy as np

# Courtesy: https://gis.stackexchange.com/questions/46729/corner-coordinates-of-google-static-map-tile
import utm
image_shape = 400

class lonlatToPixel():
    
    def __init__(self, zoom):
        tileSize = 256
        self.initial_resolution = 2 * math.pi * 6378137 / tileSize
        self.resolution = self.initial_resolution / (2 ** zoom)
        self.origin_shift = 2 * math.pi * 6378137 / 2.0

    def lonlat_toUTM(self, lon, lat):
        '''
            Input: Latitude and Longitude
            Output: Longitude and Latitude UTM projections
        '''
        tup = utm.from_latlon(lat, lon)
        lat_cnvrt = tup[0]
        lon_cnvrt = tup[1]
        # zone_number = tup[2]
        # zone_letter = tup[3]
    
        return lon_cnvrt, lat_cnvrt


    def lonlat_to_meters(self,lon,lat):
        '''
        :param lat: Center lat value
        :param lon: Center Lon value
        :return: mx: lon meter projection ESPG:900913
                 my: lat meter projection ESPG:900913
        
        "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"
        '''

        mx = lon * self.origin_shift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

        my = my * self.origin_shift / 180.0

        # mx,my = self.lonlat_toUTM(lon, lat)
        # print (mx, my)
        return mx, my
    
    
    def meters_to_pixels(self, mx, my):
        '''
        :param mx:   Input ESPG projection of Longitude
        :param my:   Input ESPG projection of Latitude
        :return:     px, py: pixel coordinate ihe center lonlat
        
        "Converts EPSG:900913 to pyramid pixel coordinates in given zoom level"
        '''
        res = self.resolution# Resolution( zoom )
        px = (mx + self.origin_shift) / res
        py = (my + self.origin_shift) / res
        return px, py
    
    
    def pixels_to_meters(self, px, py):
        # print ('px, py: ', px, py)
        "Converts pixel coordinates in given zoom level of pyramid to EPSG:900913"
        res = self.resolution #Resolution(zoom)
        mx = px * res - self.origin_shift
        my = py * res - self.origin_shift
        # print ('mx, my: ', mx, my)
        return mx, my
    
    def meters_to_lonlat(self, mx, my):
        "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"
    
        lon = (mx / self.origin_shift) * 180.0
        lat = (my / self.origin_shift) * 180.0
    
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
        return lon, lat

    def extend_polygon(self, x, y, x_cent, y_cent, how_many_pixels, prop_of_increase_in_x,
                       map_size=[400,400]):
        if x_cent > x:
            x -= how_many_pixels * prop_of_increase_in_x
        elif x_cent < x:
            x += how_many_pixels * prop_of_increase_in_x
        else:
            pass
    
        if y_cent > y:
            y -= how_many_pixels * (1 - prop_of_increase_in_x)
        elif y_cent < y:
            y += how_many_pixels * (1 - prop_of_increase_in_x)
        else:
            pass
    
        if x >= 0 and x <= map_size[1] and y >= 0 and y <= map_size[0]:
            return x, y
        else:
            return min(map_size[1], max(int(round(x)), 0)), min(map_size[0], max(int(round(y)), 0))


    def convert_map_pxl_to_img_pxl(self, pxl_at_corner00, to_convert_pxl, map_size=[400,400]):
        '''
        :param pxl_at_corner00:  The corner pixel value of the entire map derived using meters_to_pixels function
        :param to_convert_pxl:   The map pixel value to be converted into the static map pixel
        :return: The pixel number at range (0-image_shape(400)) of the static map
        
        Note : There is one subtle thing to note. The x = px - px_ and y = py_ - py are arranged
        differently. This is because :
         
         The pixels number obtained by using ESPG projection are cornered as:
                    01     11
                    
                    00     10       # here pxl_at_00 is the left lower corner
                    
        But the way Open CV/matplotlib treats images are:
                    00     01       # here pxl_at_00 is the left upper corner
                    
                    10     11
                    
        Hence when we use top-left corner of ESPG map projection pixel (01) as the point of reference for deriving
        new pixel points: we do x = px - px_ and y = py_ - px . Where px_, py_ are reference point pixels.
                    
        '''
        # print ('fsdfsdfsdfds ', pxl_at_corner00, to_convert_pxl)
        px_, py_ = pxl_at_corner00
        px, py = to_convert_pxl
        x = px - px_
        y = py_ - py
    
        if x >= 0 and x <= map_size[1] and y >= 0 and y <= map_size[0]:
            return x, y
        else:
            # print('The new pixel value is not bounded by the images shape ')
            return min(map_size[1], max(x, 0)), min(map_size[0], max(y, 0))

    def widen_a_bounding_box(self, pxl_at_corner00, to_convert_pxl):
        '''
        :param pxl_at_corner00:  The corner pixel value of the entire map derived using meters_to_pixels function
        :param to_convert_pxl:   The map pixel value to be converted into the static map pixel
        :return: The pixel number at range (0-image_shape(400)) of the static map

        Note : There is one subtle thing to note. The x = px - px_ and y = py_ - py are different arranged
        differently. This is because :

         The pixels number obtained by using ESPG projection are cornered as:
                    01     11

                    00     10

        But the way Open CV/matplotlib treats images are:
                    00     01

                    10     11

        Hence when we use top-left corner of ESPG map projection pixel (01) as the point of reference for deriving
        new pixel points: we do x = px - px_ and y = py_ - px . Where px_, py_ are reference point pixels.

        '''
    
        px_, py_ = pxl_at_corner00
        px, py = to_convert_pxl
        x = px - px_
        y = py_ - py
    
        if x >= 0 and x <= image_shape and y >= 0 and y <= image_shape:
            return x, y
        else:
            # print('The new pixel value is not bounded by the images shape ')
            return min(400, max(x, 0)), min(400, max(y, 0))


def get_image_corner_latlon_and_pxls_vals(lat, lon, zoom):
    '''
    
    :param lat:    The center latitude of the static map
    :param lon:    The center longitude of the static map
    :param zoom:   The zoom level selected while obtaining the image
    :return:
    
    '''
    obj_ll_px = lonlatToPixel(zoom=zoom)
    mx, my = obj_ll_px.lonlat_to_meters(lon, lat)
    px, py = obj_ll_px.meters_to_pixels(mx, my)

    top_left_x = px - int(image_shape/2)
    top_left_y = py + int(image_shape/2)

    top_right_x = px + int(image_shape/2)
    top_right_y = py + int(image_shape/2)

    bottom_left_x = px - int(image_shape/2)
    bottom_left_y = py - int(image_shape/2)

    bottom_right_x = px + int(image_shape/2)
    bottom_right_y = py - int(image_shape/2)

    pxls_corner_arr = np.array([[top_left_x, top_left_y],
                         [top_right_x, top_right_y],
                         [bottom_left_x, bottom_left_y],
                         [bottom_right_x, bottom_right_y]], dtype='float')
    # print('Top Left corner pixels: ', top_left_x, top_left_y)
    # print('Top Right corner pixels: ', top_right_x, top_right_y)
    # print('Bottom Left corner pixels: ', bottom_left_x, bottom_left_y)
    # print('Bottom Right corner pixels: ', bottom_right_x, bottom_right_y)

    lonlat_corner_arr = []
    for x, y in pxls_corner_arr:
        px_m, py_m = obj_ll_px.pixels_to_meters(x, y)
        llx, lly = obj_ll_px.meters_to_lonlat(px_m, py_m)
        lonlat_corner_arr.append([llx, lly])
    
    return pxls_corner_arr, lonlat_corner_arr
    
    



# convert(41.845989, -87.717689)
#
# debug = False
#
# if debug:
#     '''
#     NOTE: THE IMAGE SIZE SHOULD BE 400x400, since we are hard coding 200
#     :return:
#     '''
#     zoom = 19
#     # lat, lon = 41.89748,	-87.67867 #40.714728, -73.998672
#     lat, lon = 41.845989, -87.717689
#     # -87.68654
#      # 41.91407
#     # mx = -8237494.4864285 #-73.998672
#     # my = 4970354.7325767 # 40.714728
#     pxls_corner_arr, lonlat_corner_arr = get_image_corner_latlon_and_pxls_vals(lat, lon, zoom)
#     print (lonlat_corner_arr)
#
#
# [[41.617743580373769, 3.953749859998868], [41.618816463979741, 3.953749859998868], [41.617743580373769, 3.9526795291286323], [41.618816463979741, 3.9526795291286323]]   4632925.739760591 440419.42407523404

#
# [[-87.718225441802957, 41.84638861611691], [-87.717152558197, 41.84638861611691], [-87.718225441802957, 41.84558938138701], [-87.717152558197, 41.84558938138701]]