'''
We have created the seven folder containing the shape files (using ism_bbox_jsonToshp.py module), related to Lat-Lon scoop.
We store the folder names such as:

Why we do this: Because the shape files are very big in size and performing read operation for each record would not be feasible.
Hence we would like to arrange the lat lon in such a way that we select all records belonging to a particular Lat-Lon and
read the shp file from disk and process all the records.
'''


import os
import cv2
import numpy as np
import shapely.geometry as geom
import geopandas as gpd


from config import pathDict
from semantic_segmentation import utils as utl
from semantic_segmentation.latlon_to_pixelXY import lonlatToPixel, get_image_corner_latlon_and_pxls_vals

bbox_shape_folder_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/'
scoop_arr = ['scoop_738_768', 'scoop_768_798', 'scoop_798_848', 'scoop_848_908', 'scoop_908_938', 'scoop_938_968', 'scoop_968_028']

lat_scoop = dict(scoop_738_768 = '41.738, -87.803, 41.768, -87.510',
                scoop_768_798 = '41.768, -87.803, 41.798, -87.510',
                scoop_798_848 = '41.798, -87.803, 41.848, -87.510',
                scoop_848_908 = '41.848, -87.803, 41.908, -87.510',
                scoop_908_938 = '41.908, -87.803, 41.938, -87.510',
                scoop_938_968 = '41.938, -87.803, 41.968, -87.510',
                scoop_968_028 = '41.968, -87.803, 42.028, -87.510')
zoom = 19


def image_corner_polygon_wraper(lat, lon, zoom):
    pxls_corner_arr, lonlat_corner_arr = get_image_corner_latlon_and_pxls_vals(lat, lon, zoom)
    tl_ll, tr_ll, bl_ll, br_ll = lonlat_corner_arr
    tl_pxl, tr_pxl, bl_pxl, br_pxl = pxls_corner_arr
    image_corner_polygon = geom.Polygon(np.array([tl_ll, tr_ll, br_ll, bl_ll, tl_ll]))
    return image_corner_polygon, tl_pxl


def get_search_polygons(df, scoop_latlon):
    query = ''
    for num, (i, j) in enumerate(scoop_latlon):
        query += '((lon_scoop=="%s") & (lat_scoop=="%s"))' % (str(i), (j))
        if num != (len(scoop_latlon) - 1):
            query += ' | '
    return df[df.eval(query)]


def get_parcels_inside_image(df, image_polygon):
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


def draw_polygons(img, xy_arr):
    alpha = 0.5
    xy_arr = np.array(xy_arr, np.int32)
    layer = img.copy()
    output = img.copy()
    layer = cv2.fillPoly(layer, [xy_arr], (0,0,255))#, offset=0.4)
    output = cv2.addWeighted(layer, alpha, output, 1 - alpha, 0, output)
    return output


def get_overlayed_image(parcels_in_image, tl_pxl, img):
    a = []
    obj_ll_to_pxl = lonlatToPixel(zoom=19)
    for parcel_id, polygon_geom in np.array(parcels_in_image[['parcel_id', 'geometry']]):
        points_tuple = polygon_geom.exterior.coords.xy
        lon_arr, lat_arr = list(points_tuple[0]), list(points_tuple[1])
        each_parcel_xy_arr = []
        for lon, lat in zip(lon_arr, lat_arr):
            mx, my = obj_ll_to_pxl.lonlat_to_meters(lon,lat)
            px, py = obj_ll_to_pxl.meters_to_pixels(mx, my)
            # print (tl_pxl)
            # print (px, py)
            img_x, img_y = obj_ll_to_pxl.convert_map_pxl_to_img_pxl(tl_pxl, [px, py])
            each_parcel_xy_arr.append([int(np.round(img_x)), int(np.round(img_y))])
        # print ('each_parcel_xy_arr ', each_parcel_xy_arr)
        # print('img ', img)
        img = draw_polygons(img, xy_arr=each_parcel_xy_arr)
    return img



def overlay_parcel_on_images(data_to_model):
    for sc_cnt,scoop in enumerate(scoop_arr):
        # if sc_cnt == 0:
        #     continue
        min_lat, max_lon, max_lat, min_lon = [float(i.strip()) for i in lat_scoop[scoop].split(',')]
        # print(min_lat, max_lon, max_lat, min_lon)
        data_ = data_to_model[
            (data_to_model['lat'] >= min_lat) & (data_to_model['lat'] < max_lat) & (data_to_model['lon'] >= max_lon) & (
            data_to_model['lon'] < min_lon)]
        property_parcel = gpd.read_file(os.path.join(bbox_shape_folder_path, lat_scoop[scoop]))
        print('Initiating Run for scoop %s : DATA SHAPE %s'%(scoop, str(property_parcel.shape)))
        
        for rcnt, (pin, lat, lon, label) in enumerate(np.array(data_[['pin', 'lat', 'lon', 'indicator']])):
            # if rcnt<700:
            #     continue
            # Get the Google map static images using Pin from the disk
            if label == 'Likely Land':
                output_path = os.path.join(pathDict['google_overlayed_image_path'], 'land', '%s.jpg'%str(pin))
                static_image_path = os.path.join(pathDict['google_aerial_image_path'], 'land', '%s.jpg'%str(pin))
                img = cv2.imread(static_image_path)
            elif label == 'Likely House':
                output_path = os.path.join(pathDict['google_overlayed_image_path'], 'house', '%s.jpg' % str(pin))
                static_image_path = os.path.join(pathDict['google_aerial_image_path'], 'house', '%s.jpg' % str(pin))
                img = cv2.imread(static_image_path)
            else:
                raise ValueError('The image is not indicated as "Likely Land" or "Likely House"')
            
            
            # If the image is not present and None is returned handle such cases here
            if type(img).__module__ != np.__name__:
                continue
            
            lat = float(lat)
            lon = float(lon)
            image_polygon, tl_pxl = image_corner_polygon_wraper(lat, lon, zoom)
            # print(image_polygon)
            
            scoop_lon, scooop_lat = utl.getscoopLonLat(lonIN=lon, latIN=lat, decimalPlaces=1000)
            scoop_latlon = utl.getscoopSearchItems(scoopLon=scoop_lon, scoopLat=scooop_lat, decimalPlaces=1000)
            
            search_polygons = get_search_polygons(property_parcel, scoop_latlon)
            # print(search_polygons.shape)
            
            parcels_in_image = get_parcels_inside_image(search_polygons, image_polygon)
            # print(len(parcels_in_image))

            if len(parcels_in_image) >0:
                overlayed_image = get_overlayed_image(parcels_in_image, tl_pxl, img)
            else:
                overlayed_image = img
            # print (os.path.join(pathDict['google_overlayed_image_path'], '%s.jpg'%str(pin)))

            cv2.imwrite(output_path, overlayed_image)

            if ((rcnt+1) % 100) == 0:
                print("TOTAL IMAGES PARSED ======== %s"%(str(rcnt)))

        print('')
        # break
        #
        
   
   
debugg = False
if debugg:
    google_stats_data = utl.collateData(pathDict['google_aerial_stats_path'])
    print('Shape: Google Stats images ', google_stats_data.shape)

    data_to_model = google_stats_data[(google_stats_data['loc_type'] != 'RANGE_INTERPOLATED') & (
    google_stats_data['city'].str.lower().str.strip().str.match('chicago')) & (
                                      ~google_stats_data['address'].str.lower().str.strip().str.match('0'))]
    print('Shape: Images for Overlaying: ', data_to_model.shape)
    overlay_parcel_on_images(data_to_model)