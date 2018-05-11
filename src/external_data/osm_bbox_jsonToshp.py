'''
About the module:

Prerequisite: Download building footprints of Chicago Region (or any place) from OSM using Turbo API (or how ever you
wish). It is better to have them as small geojson files, somewhere around 80-100 MB per file

Functionality: This module reads in the geojson file extracts properties such as centre of the building bounding box, address and others and load the file
as a shape file and dumps in the respective directory.
'''



import os
import json
import numpy as np
from shapely import geometry as geom
import geopandas as gpd
import pandas as pd

from src.semantic_segmentation.plot import GeoPlot
from src.semantic_segmentation import utils as utls


def plot_parcels(base_path, top_layer_path):
    county_data = gpd.read_file(base_path)
    county_data['NAME'] = county_data['NAME'].str.lower()
    cook_county = county_data[(county_data['NAME'] == 'cook') & (county_data['GEOID'] == '17031')]
    
    parcel_shape_data = gpd.read_file(top_layer_path)
    
    obj_GP = GeoPlot()
    obj_GP.set_figure(lenXaxis=30, lenYaxis=15)
    obj_GP.base_plot(color='white', dataIN=cook_county)
    obj_GP.add_plot(parcel_shape_data, shapePlot=True)
    obj_GP.show()
    
def convert_geojosn_to_shape(geojson_file_path):
    outfile_name = os.path.basename(geojson_file_path).rsplit('.', 1)[0]
    # print (outfile_name)
    dir_path = os.path.dirname(geojson_file_path)
    folder_path = os.path.join(dir_path, outfile_name)
    print (folder_path)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Read the Geojson file
    with open(geojson_file_path) as f:
        data = json.load(f)

    geometry = 'nan'
    geometry_poly = 'nan'
    shape_type = 'nan'
    parcel_id = 'nan'
    house_number = 'nan'
    street_address = 'nan'
    street_name = 'nan'
    street_prefix = 'nan'
    building = 'nan'
    building_level = 'nan'
    short_lat = 'nan'
    short_lon = 'nan'
    center_lat = 'nan'
    center_lon = 'nan'

    array_stack = []
    for features in data['features']:
        features_keys = features.keys()

        if 'geometry' in features_keys:
            geometry = features['geometry']['coordinates']
            shape_type = features['geometry']['type']
            if shape_type == 'Polygon':
                geometry_poly = geom.Polygon(geometry[0])
                center_lon, center_lat = np.array(geometry_poly.centroid.coords)[0]
                short_lon, short_lat = utls.getscoopLonLat(lonIN=center_lon, latIN=center_lat, decimalPlaces=1000)
            else:
                continue

        if 'id' in features_keys:
            parcel_id = features['id']

        if 'properties' in features_keys:
            address_keys = features['properties'].keys()
            if 'addr:housenumber' in address_keys:
                house_number = features['properties']['addr:housenumber']

            if 'addr:street' in address_keys:
                street_address = features['properties']['addr:street']

            if 'addr:street:name' in address_keys:
                street_name = features['properties']['addr:street:name']

            if 'addr:street:prefix' in address_keys:
                street_prefix = features['properties']['addr:street:prefix']

            if 'building' in address_keys:
                building = features['properties']['building']

            if 'building:levels' in address_keys:
                building_level = features['properties']['building:levels']

        if geometry != 'nan':
            hstack = [parcel_id, house_number, street_address, street_name, street_prefix, building, building_level,
                      shape_type, short_lat, short_lon, float(center_lat), float(center_lon), geometry_poly]
            array_stack.append(hstack)

    parcel_data = None
    if len(array_stack) > 0:
        columns = ['parcel_id', 'house_number', 'street_address', 'street_name', 'street_prefix', 'building',
                   'building_level', 'shape_type','lat_scoop','lon_scoop', 'lat_center','lon_center','geometry']
        parcel_data = pd.DataFrame(np.array(array_stack), columns=columns)
        parcel_data = gpd.GeoDataFrame(parcel_data, geometry='geometry')

        parcel_data.to_file(os.path.join(folder_path, outfile_name + '.shp'))

    return folder_path, parcel_data

#
# debug = False
# create_shp = False
# wanna_plot = False
#
# if debug:
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.738,
#     # -87.803, 41.768, -87.510.geojson'
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.768,
#     # -87.803, 41.798, -87.510.geojson'
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.798,
#     # -87.803, 41.848, -87.510.geojson'
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.848,
#     # -87.803, 41.908, -87.510.geojson'
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.908,
#     # -87.803, 41.938, -87.510.geojson'
#     # geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.938,
#     # -87.803, 41.968, -87.510.geojson'
#     geo_json_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files/building_bbox/41.968, ' \
#                     '-87.803, 42.028, -87.510.geojson'
#
#     parcel_folder_path = None
#     if create_shp:
#         parcel_folder_path, parcel_data = convert_geojosn_to_shape(geo_json_path)
#         print (parcel_data.head(10))
#     # print (np.array(parcel_data['lon_center']))
#     if wanna_plot:
#         parcel_folder_path = os.path.join(os.path.dirname(geo_json_path), os.path.basename(geo_json_path).rsplit('.',1)[0])
#         US_couty_shapefile_path = '/Users/sam/All-Program/App-DataSet/HouseClassification/shape_files' \
#                                    '/US_county_shape_files'
#         plot_parcels(base_path=US_couty_shapefile_path, top_layer_path=parcel_folder_path)