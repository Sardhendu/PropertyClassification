'''
We have created the seven folder containing the shape files (using ism_bbox_jsonToshp.py module), related to Lat-Lon scoop.
We store the folder names such as:

Why we do this: Because the shape files are very big in size and performing read operation for each record would not be feasible.
Hence we would like to arrange the lat lon in such a way that we select all records belonging to a particular Lat-Lon and
read the shp file from disk and process all the records.
'''

import os

import cv2
import geopandas as gpd
import numpy as np
import shapely.geometry as geom
import logging

from src.semantic_segmentation import utils as utl
from src.semantic_segmentation.latlon_to_pixelXY import lonlatToPixel, get_image_corner_latlon_and_pxls_vals

scoop_arr = ['scoop_738_768', 'scoop_768_798', 'scoop_798_848', 'scoop_848_908', 'scoop_908_938', 'scoop_938_968', 'scoop_968_028']

lat_scoop = dict(scoop_738_768 = '41.738, -87.803, 41.768, -87.510',
                scoop_768_798 = '41.768, -87.803, 41.798, -87.510',
                scoop_798_848 = '41.798, -87.803, 41.848, -87.510',
                scoop_848_908 = '41.848, -87.803, 41.908, -87.510',
                scoop_908_938 = '41.908, -87.803, 41.938, -87.510',
                scoop_938_968 = '41.938, -87.803, 41.968, -87.510',
                scoop_968_028 = '41.968, -87.803, 42.028, -87.510')


def image_corner_polygon_wraper(lat, lon, zoom):
    '''
    :param lat:
    :param lon:
    :param zoom: 19, 20, 21   {19 is used here}
    :return:     A polygon with lat-lon of the corner at points in the polygon
    '''
    pxls_corner_arr, lonlat_corner_arr = get_image_corner_latlon_and_pxls_vals(lat, lon, zoom)
    tl_ll, tr_ll, bl_ll, br_ll = lonlat_corner_arr
    tl_pxl, tr_pxl, bl_pxl, br_pxl = pxls_corner_arr
    image_corner_polygon = geom.Polygon(np.array([tl_ll, tr_ll, br_ll, bl_ll, tl_ll]))
    return image_corner_polygon, tl_pxl


def get_search_polygons(df, scoop_latlon):
    '''
    :param df:           Is a mini dataFrame contained the parcels within the scooped region
    :param scoop_latlon: Dictionary of scooped lat-lon to search parcels in
    :return:
    '''
    query = ''
    for num, (i, j) in enumerate(scoop_latlon):
        query += '((lon_scoop=="%s") & (lat_scoop=="%s"))' % (str(i), (j))
        if num != (len(scoop_latlon) - 1):
            query += ' | '
    return df[df.eval(query)]


def get_parcels_inside_image(df, image_polygon):
    '''
    :param df:             Is a mini dataFrame contained the parcels within the scooped region
    :param image_polygon:  The polygon with lat-lon as corners of the static image
    :return:
    '''
    list_parcel_ids = []
    for parcel_id, lon_cent, lat_cent in np.array(df[['parcel_id', 'lon_center', 'lat_center']]):
        lon_cent = float(lon_cent)
        lat_cent = float(lat_cent)
        is_contains = image_polygon.contains(geom.Point([lon_cent, lat_cent]))
        if is_contains:
            #             print (parcel_id)
            list_parcel_ids.append(parcel_id)
    
    if len(list_parcel_ids) > 0:
        return df[df['parcel_id'].isin(list_parcel_ids)]
    else:
        return []


def get_parcels_given_latlon(df, lat, lon):
    '''
    :param df:             Is a mini dataFrame contained the parcels within the scooped region
    :param image_polygon:  The polygon with lat-lon as corners of the static image
    :return:
    '''
    list_parcel_ids = []
    for parcel_id, poly in np.array(df[['parcel_id', 'geometry']]):
        lon_cent = float(lon)
        lat_cent = float(lat)
        is_contains = poly.contains(geom.Point([lon_cent, lat_cent]))
        if is_contains:
            list_parcel_ids.append(parcel_id)
    
    if len(list_parcel_ids) > 0:
        return df[df['parcel_id'].isin(list_parcel_ids)]
    else:
        return []


def draw_polygons(img, xy_arr):
    alpha = 0.5
    xy_arr = np.array(xy_arr, np.int32)
    layer = img.copy()
    output = img.copy()
    layer = cv2.fillPoly(layer, [xy_arr], (0,0,255))#, offset=0.4)
    output = cv2.addWeighted(layer, alpha, output, 1 - alpha, 0, output)
    return output


def get_overlaid_image(parcels_in_image, tl_pxl, img, zoom=19, map_size=[400,400]):
    '''
    :param parcels_in_image:    The parcels (shapely polygons) contained in the image
    :param tl_pxl:              The top left pixel in the image
    :param img:                 The google static image
    :return:
    
    1. Converts the lat-lon of the parcel into pixel position in the image.
    2. Colors the bounding boxes within the pixel position.
    '''
    a = []
    obj_ll_to_pxl = lonlatToPixel(zoom=zoom)
    for parcel_id, polygon_geom in np.array(parcels_in_image[['parcel_id', 'geometry']]):
        points_tuple = polygon_geom.exterior.coords.xy
        lon_arr, lat_arr = list(points_tuple[0]), list(points_tuple[1])
        parcel_xy_arr = []
        for lon, lat in zip(lon_arr, lat_arr):
            mx, my = obj_ll_to_pxl.lonlat_to_meters(lon,lat)
            px, py = obj_ll_to_pxl.meters_to_pixels(mx, my)
            # print (tl_pxl)
            # print (px, py)
            img_x, img_y = obj_ll_to_pxl.convert_map_pxl_to_img_pxl(tl_pxl, [px, py])
            parcel_xy_arr.append([int(np.round(img_x)), int(np.round(img_y))])
        # print ('parcel_xy_arr ', parcel_xy_arr)
        # print('img ', img)
        img = draw_polygons(img, xy_arr=parcel_xy_arr)
    return img



def crop_polygons(img, xy_arr):
    xy_arr = np.array(xy_arr, np.int32)
    layer = img.copy()
    x, y, w, h = cv2.boundingRect(xy_arr)
    # print (x, y, w, h)
    region_of_interest = layer[y:y + h, x:x + w]

    return region_of_interest




def get_cropped_bbox(parcels_in_image, tl_pxl, img, zoom=19, map_size=[400,400]):
    '''
    :param parcels_in_image:    The parcel for the address in question(shapely polygons)
    :param tl_pxl:              The top left pixel in the image
    :param img:                 The google static image
    :return:

    1. Converts the lat-lon of the parcel into pixel position in the image.
    2. Colors the bounding boxes within the pixel position.
    '''
    obj_ll_to_pxl = lonlatToPixel(zoom=zoom)
    for parcel_id, polygon_geom in np.array(parcels_in_image[['parcel_id', 'geometry']]):
        
        # FOR THE CENTER POINT (CENTER OF THE BOUNDARY)
        center_of_polygon = polygon_geom.centroid.coords.xy
        lon_cent, lat_cent = list(center_of_polygon[0])[0], list(center_of_polygon[1])[0]
        mx_cent, my_cent = obj_ll_to_pxl.lonlat_to_meters(lon_cent, lat_cent)
        px_cent, py_cent = obj_ll_to_pxl.meters_to_pixels(mx_cent, my_cent)
        img_x_cent, img_y_cent = obj_ll_to_pxl.convert_map_pxl_to_img_pxl(tl_pxl, [px_cent, py_cent], map_size=map_size)
        img_x_cent, img_y_cent = int(round(img_x_cent)), int(round(img_y_cent))
        
        # FOR THE EXTERIOR POINTS (BOUNDARY POINTS OF THE POLYGON)
        points_tuple = polygon_geom.exterior.coords.xy
        lon_arr, lat_arr = list(points_tuple[0]), list(points_tuple[1])
        parcel_xy_arr = []
        # print (lon_arr, lat_arr)
        for lon, lat in zip(lon_arr, lat_arr):
            mx, my = obj_ll_to_pxl.lonlat_to_meters(lon, lat)
            px, py = obj_ll_to_pxl.meters_to_pixels(mx, my)
            img_x, img_y = obj_ll_to_pxl.convert_map_pxl_to_img_pxl(tl_pxl, [px, py], map_size=map_size)
            # print (img_x, img_y)
            parcel_xy_arr.append([int(np.round(img_x)), int(np.round(img_y))])
        
        parcel_xy_arr = np.array(parcel_xy_arr, dtype=int)
        # print(parcel_xy_arr)
        
        # GET THE LENGTH OF THE PROPERTY TO AUGMENT THE CROPPED REGION USING THE ASPECT RATIO
        min_x, min_y = min(parcel_xy_arr[:,0]), min(parcel_xy_arr[:,1])
        max_x, max_y = max(parcel_xy_arr[:, 0]), max(parcel_xy_arr[:, 1])
        legth_property = max_x - min_x
        width_property = max_y - min_y
        prop_of_increase_in_x = legth_property/(legth_property+width_property)

        ext_parcel_xy_arr = []
        for img_x, img_y in parcel_xy_arr:
            ext_img_x, ext_img_y = obj_ll_to_pxl.extend_polygon(x=img_x, y=img_y, x_cent=img_x_cent, y_cent=img_y_cent,
                                              how_many_pixels=100, prop_of_increase_in_x=prop_of_increase_in_x,
                                                                map_size=map_size)
            ext_parcel_xy_arr.append([ext_img_x, ext_img_y])

        img = crop_polygons(img, xy_arr=ext_parcel_xy_arr)

    return img


def overlay_parcel_on_images(conf, data_to_model, aerial_img_path, overlaid_img_path, zoom, map_size):
    '''
    
    :param data_to_model:
    :return:
    
    Step 1: Load the csv file that contains the google downloaded information such as pin, lat, lon, indicator. This
    file is created while downloading google static images. For a given scoop extract all the records with lat lon within that scoop
    Step 2: Iterate over all "larger scoop"
            Step 2.1: Load all the properties in the "larger scoop"., For example: '41.738, -87.803, 41.768, -87.510'.
            Step 2.2: Fetch all the property images from teh disk that belongs to the scoop. For example: '41.738, -87.803, 41.768, -87.510'.
            Step 2.3: Iterate through each image.
                    Step 2.3.1: Load the image from disk
                    Step 2.3.2: We have the center lat-lon of the image
                    Step 2.3.3: For that image get the corner pixels values and corner lat-lon values.
                    Step 2.3.4: Initiate another scoop search based on the center lat-lon of the image
                    Step 2.3.5: Get all the building polygons from the scoop that were found to be under the image
                    corner. Basically check if the lat-lon of the building polygon center is within the lat-lon of
                    the image corner.
                    Step 2.3.6: For all building polygons present inside the image corner, convert the lat-lon into
                    meters and meters into pixels and color the polygon_pixels as red
    '''
    
    aerial_land_path =  os.path.join(aerial_img_path, 'land')
    aerial_house_path = os.path.join(aerial_img_path, 'house')
    overlaid_land_path = os.path.join(overlaid_img_path, 'land')
    overlaid_house_path = os.path.join(overlaid_img_path, 'house')

    processed_pins = []
    for dir in [overlaid_land_path, overlaid_house_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            processed_pins += [img.split('.')[0] for img in os.listdir(dir) if img != '.DS_Store']
    
    done = 0
    for sc_cnt,scoop in enumerate(scoop_arr):

        min_lat, max_lon, max_lat, min_lon = [float(i.strip()) for i in lat_scoop[scoop].split(',')]
        # print(min_lat, max_lon, max_lat, min_lon)

        ################ Step 1.
        data_ = data_to_model[
            (data_to_model['lat'] >= min_lat) & (data_to_model['lat'] < max_lat) & (data_to_model['lon'] >= max_lon) & (
            data_to_model['lon'] < min_lon)]

        ################ Step 2.
        property_parcel = gpd.read_file(os.path.join(conf['pathDict']['chicago_bbox_shp_files'], lat_scoop[scoop]))
        print('Initiating Run for scoop %s : DATA SHAPE %s'%(scoop, str(property_parcel.shape)))

        ################ Step 3.
        if len(processed_pins) > 0:
            data_ = data_[~data_['pin'].isin(processed_pins)]
            print('Total images to be processed are ', data_.shape)
        
        dataIN = np.array(data_[['pin', 'lat', 'lon', 'indicator']])
        for rcnt, (pin, lat, lon, label) in enumerate(dataIN):
            # if rcnt<700:
            #     continue

            ################ Step 3.1.
            # Get the Google map static images using Pin from the disk
            if label == 'Likely Land':
                aerial_land_image_path = os.path.join(aerial_land_path, '%s.jpg' % str(pin))
                overlaid_dump_path = os.path.join(overlaid_land_path, '%s.jpg'%str(pin))
                img = cv2.imread(aerial_land_image_path)
            elif label == 'Likely House':
                aerial_house_image_path = os.path.join(aerial_house_path, '%s.jpg' % str(pin))
                overlaid_dump_path = os.path.join(overlaid_house_path, '%s.jpg' % str(pin))
                img = cv2.imread(aerial_house_image_path)
            else:
                raise ValueError('The image is not indicated as "Likely Land" or "Likely House"')
            
            
            # If the image is not present and None is returned handle such cases here
            if type(img).__module__ != np.__name__:
                continue
            
            lat = float(lat)
            lon = float(lon)

            ################ Step 3.2.
            image_polygon, tl_pxl = image_corner_polygon_wraper(lat, lon, zoom)
            # print(image_polygon)
            
            scoop_lon, scoop_lat = utl.getscoopLonLat(lonIN=lon, latIN=lat, decimalPlaces=1000)
            scoop_latlon = utl.getscoopSearchItems(scoopLon=scoop_lon, scoopLat=scoop_lat, decimalPlaces=1000)
            
            search_polygons = get_search_polygons(property_parcel, scoop_latlon)
            # print(search_polygons.shape)
            
            parcels_in_image = get_parcels_inside_image(search_polygons, image_polygon)
            # print(len(parcels_in_image))

            if len(parcels_in_image) >0:
                overlaid_image = get_overlaid_image(parcels_in_image, tl_pxl, img, zoom, map_size)
            else:
                overlaid_image = img
            # print (os.path.join(pathDict['google_overlaid_image_path'], '%s.jpg'%str(pin)))

            cv2.imwrite(overlaid_dump_path, overlaid_image)

            done += 1
            
            if ((done+1) % 100) == 0:
                b = "TOTAL RECORDS PARSED: IMAGES DONE ======== %s"
                print(b % (rcnt), end="\r")
                # print("Total Records Parsed ======== %s"%(str(done)))


def crop_parcel_from_images(conf, data_to_model, aerial_img_path, aerial_cropped_img_path, zoom, map_size):
    '''
    Step 1: Load the csv file that contains the google downloaded information such as pin, lat, lon, indicator. This
    file is created while downloading google static images. For a given scoop extract all the records with lat lon within that scoop
    Step 2: Iterate over all "larger scoop"
            Step 2.1: Load all the properties in the "larger scoop"., For example: '41.738, -87.803, 41.768, -87.510'.
            Step 2.2: Fetch all the property images from teh disk that belongs to the scoop. For example: '41.738, -87.803, 41.768, -87.510'.
            Step 2.3: Iterate through each image.
                    Step 2.3.1: Load the image from disk
                    Step 2.3.2: We have the center lat-lon of the image
                    Step 2.3.3: For that image get the corner pixels values and corner lat-lon values.
                    Step 2.3.4: Initiate another scoop search based on the center lat-lon of the image
                    Step 2.3.5: Get all the building polygons from the scoop that were found to be under the image
                    corner. Basically check if the lat-lon of the building polygon center is within the lat-lon of
                    the image corner.
                    Step 2.3.6: For all building polygons present inside the image corner, check which building
                    polygon contains the center of the image. If found then crop and augment the center of the image
                    with the aspect ratio.
    '''
    aerial_land_path = os.path.join(aerial_img_path, 'land')
    aerial_house_path = os.path.join(aerial_img_path, 'house')
    aerial_cropped_land_path = os.path.join(aerial_cropped_img_path, 'land')
    aerial_cropped_house_path = os.path.join(aerial_cropped_img_path, 'house')
    
    processed_pins = []
    for dir in [aerial_cropped_land_path, aerial_cropped_house_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            processed_pins += [img.split('.')[0] for img in os.listdir(dir) if img!='.DS_Store']
    
    
    done = 0
    for sc_cnt, scoop in enumerate(scoop_arr):
        min_lat, max_lon, max_lat, min_lon = [float(i.strip()) for i in lat_scoop[scoop].split(',')]
        # print(min_lat, max_lon, max_lat, min_lon)

        # ################ Step 1.
        data_ = data_to_model[
            (data_to_model['lat'] >= min_lat) & (data_to_model['lat'] < max_lat) & (data_to_model['lon'] >= max_lon) & (
                data_to_model['lon'] < min_lon)]
        #
        # ################ Step 2.
        property_parcel = gpd.read_file(os.path.join(conf['pathDict']['chicago_bbox_shp_files'], lat_scoop[scoop]))
        print('Initiating Run for scoop %s : DATA SHAPE %s' % (scoop, str(property_parcel.shape)))

        ################ Step 3.
        if len(processed_pins) > 0:
            data_ = data_[~data_['pin'].isin(processed_pins)]
            print ('Total images to be processed are ', data_.shape)
            
        dataIN = np.array(data_[['pin', 'lat', 'lon', 'indicator']])
        for rcnt, (pin, lat, lon, label) in enumerate(dataIN):

        #     ################ Step 3.1.
        #     # Get the Google map static images using Pin from the disk
            if label == 'Likely Land':
                aerial_land_image_path = os.path.join(aerial_land_path, '%s.jpg' % str(pin))
                aerial_cropped_dump_path = os.path.join(aerial_cropped_land_path, '%s.jpg' % str(pin))
                img = cv2.imread(aerial_land_image_path)
            elif label == 'Likely House':
                aerial_house_image_path = os.path.join(aerial_house_path, '%s.jpg' % str(pin))
                aerial_cropped_dump_path = os.path.join(aerial_cropped_house_path, '%s.jpg' % str(pin))
                img = cv2.imread(aerial_house_image_path)
            else:
                raise ValueError('The image is not indicated as "Likely Land" or "Likely House"')
        #
        #     # If the image is not present and None is returned handle such cases here
            if type(img).__module__ != np.__name__:
                continue

            lat = float(lat)
            lon = float(lon)
        
        #     ################ Step 3.2.
            image_polygon, tl_pxl = image_corner_polygon_wraper(lat, lon, zoom)
            # print(image_polygon)
            

            scoop_lon, scoop_lat = utl.getscoopLonLat(lonIN=lon, latIN=lat, decimalPlaces=1000)
            scoop_latlon = [[scoop_lon, scoop_lat]]#utl.getscoopSearchItems(scoopLon=scoop_lon, scoopLat=scoop_lat,
                            # decimalPlaces=1000)
            # print (scoop_latlon)

            search_polygons = get_search_polygons(property_parcel, scoop_latlon)
            # print(search_polygons.shape)
            
            parcels_in_image = get_parcels_given_latlon(search_polygons, lat, lon)
            # print (parcels_in_image)

            if len(parcels_in_image) > 0:
                cropped_aerial_image = get_cropped_bbox(parcels_in_image, tl_pxl, img, zoom, map_size)
            else:
                cropped_aerial_image = img
        # assert len(img.shape) == 3
            # img_resized = scipy.misc.imresize(img, (224, 224))
            # img_answer = (img_resized / 255.0).astype('float32')
            #
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(overlaid_image))
            # plt.show()
        
            cv2.imwrite(aerial_cropped_dump_path, cropped_aerial_image)

            if ((done+1) % 100) == 0:
                logging.info("Total Records Parsed ======== %s", str(done))

        # print('')



#
#
# which_run = '1522630301'
# debugg = True
# overlaid = False
# aerial_cropped = False
#
# if debugg:
#
#     if overlaid:
#         aerial_stats_path = os.path.join(pathDict['aerial_stats_path'], which_run)
#         if not os.path.exists(aerial_stats_path):
#             raise ValueError('The path %s seems to not exist, Make sure you have dumped the Aerial Images.', )
#         aerial_stats_data = utl.collateData(aerial_stats_path)
#         print('Shape: Google Stats images ', aerial_stats_data.shape)
#
#         data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
#         aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
#             'address'].str.lower().str.strip().str.match('0'))]
#         print('Shape: Images for Overlaying: ', data_to_model.shape)
#         overlay_parcel_on_images(data_to_model, zoom=20, map_size=[400, 400], which_run=which_run)
#
#     if aerial_cropped:
#         aerial_stats_path = os.path.join(pathDict['aerial_stats_path'], which_run)
#         if not os.path.exists(aerial_stats_path):
#             raise ValueError('The path %s seems to not exist, Make sure you have dumped the Aerial Images.', )
#         aerial_stats_data = utl.collateData(aerial_stats_path)
#         print('Shape: Google Stats images ', aerial_stats_data.shape)
#
#         data_to_model = aerial_stats_data[(aerial_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
#             aerial_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (~aerial_stats_data[
#             'address'].str.lower().str.strip().str.match('0'))]
#         print('Shape: Images for Overlaying: ', data_to_model.shape)
#         crop_parcel_on_images(data_to_model, zoom=20, map_size=[400, 400], which_run=which_run)