# Semantic Segmentation:

Deep Learning algorithms are robust to some amount of mislabeling and are very data hungry. Obtaining segmented training data is very not feasible. 

This module contains 3 steps.

## Data Creation

#### Open Street Map (Fetch building polygons (corners))
1. We fetch the geojson file for building corners of chicago region using the Open Street Map Turbo API.
2. We then parse the geojson file and collect information pertaining to each building such as "adddress", "building polygons, "center" and etc ... . see [a link] https://github.com/Sardhendu/PropertyClassification/blob/master/ExternalData/OSM_fetch.py
3. Further Data Preparation is achieved using "Link to git py file"

##### Snapshot of the parcel polygon when plotted on Chicago's map (Using GeoPandas)
![alt text](https://github.com/Sardhendu/PropertyClassification/blob/master/images/OSM_parcel.png)

#### Google static map and Bing static map
1. For a given property address we first fetch the Latitude and Longitude of the property and then fetch the static map using a zoom level of 19. Note the image center is the Lat-Lon. This is achieved from both Google Map and Bing Map. see [a link] https://github.com/Sardhendu/PropertyClassification/blob/master/ExternalData/bing_search.py and [a link] https://github.com/Sardhendu/PropertyClassification/blob/master/ExternalData/google_satellite_img.py

##### Extracted Images from Map
<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/staticmap0.png" width="400" height="300"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/staticmap1.png" width="400" height="300">

2. Now we overlay the parcel Polygon fetched from OSM on the static map. This is achieved by converting the LatLon of center of image to meter and then converting them to Pixel space. The code can be found here. [a link] https://github.com/Sardhendu/PropertyClassification/blob/master/semantic_segmentation/latlon_to_pizelXY.py

##### Images after overlaying:
<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/staticmap0_out.png" width="400" height="300"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/staticmap1_out.png" width="400" height="300">

Now if you compare the **OSM building polygons** with the **Overlayed Images**, you can easily figure out the part of the image that is overlayed. 




