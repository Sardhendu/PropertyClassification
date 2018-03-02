
import math

tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
# 156543.03392804062 for tileSize 256 pixels
originShift = 2 * math.pi * 6378137 / 2.0
# 20037508.342789244

def LatLonToMeters( lat, lon ):
        "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"

        mx = lon * originShift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

        my = my * originShift / 180.0
        return mx, my


def MetersToPixels( mx, my, zoom):
        "Converts EPSG:900913 to pyramid pixel coordinates in given zoom level"

        res = initialResolution / (2 ** zoom)
        px = (mx + originShift) / res
        py = (my + originShift) / res
        return px, py


def PixelsToMeters( px, py, zoom):
    "Converts pixel coordinates in given zoom level of pyramid to EPSG:900913"

    res = initialResolution / (2 ** zoom)
    mx = px * res - originShift
    my = py * res - originShift
    return mx, my

def MetersToLatLon( mx, my ):
    "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"

    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0

    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon




#Result

# Input lonlat:  40.714728 -73.998672
# Output projection lat, lon:  -8237494.486418471 4970354.732576708
# projection Lat Lon:  4970354.732576708 -8237494.486418471
# Pixel Center Lat Lon:  167510775.62588218 79040318.93968214
# Top Left corner pixels:  79040118.93968214 167510975.62588218
# Top Right corner pixels:  79040518.93968214 167510975.62588218
# Bottom Left corner pixels:  79040118.93968214 167510575.62588218
# Bottom Right corner pixels:  79040518.93968214 167510575.62588218
# [[-73.99894022  40.7149313 ]
#  [-73.99840378  40.7149313 ]
#  [-73.99894022  40.7145247 ]
#  [-73.99840378  40.7145247 ]]

# Dont forget you have to convert your projection to EPSG:900913
# mx = -8237494.4864285 #-73.998672
# my = 4970354.7325767 # 40.714728
lat = 40.714728
lon = -73.998672
zoom = 20

mx, my = LatLonToMeters(lat, lon)

px,py = MetersToPixels( mx, my, zoom)

x = px - 200
y = py + 200

mx2, my2 = PixelsToMeters(x, y, zoom)

llx, lly = MetersToLatLon(mx2, my2)

print (lon, lat)
print (mx,my)
print (px,py)
print (x,y)
print (mx2,my2)
print (llx, lly)






#
# import urllib.request
# from PIL import Image
# import os
# import math
#
#
# class GoogleMapDownloader:
#     """
#         A class which generates high resolution google maps images given
#         a longitude, latitude and zoom level
#     """
#
#     def __init__(self, lat, lng, zoom=12):
#         """
#             GoogleMapDownloader Constructor
#             Args:
#                 lat:    The latitude of the location required
#                 lng:    The longitude of the location required
#                 zoom:   The zoom level of the location required, ranges from 0 - 23
#                         defaults to 12
#         """
#         self._lat = lat
#         self._lng = lng
#         self._zoom = zoom
#
#     def getXY(self):
#         """
#             Generates an X,Y tile coordinate based on the latitude, longitude
#             and zoom level
#             Returns:    An X,Y tile coordinate
#         """
#
#         tile_size = 256
#
#         # Use a left shift to get the power of 2
#         # i.e. a zoom level of 2 will have 2^2 = 4 tiles
#         numTiles = 1 << self._zoom
#
#         # Find the x_point given the longitude
#         point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size
#
#         # Convert the latitude to radians and take the sine
#         sin_y = math.sin(self._lat * (math.pi / 180.0))
#
#         # Calulate the y coorindate
#         point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
#             tile_size / (2 * math.pi))) * numTiles // tile_size
#
#         return int(point_x), int(point_y)
#
#     def generateImage(self, **kwargs):
#         """
#             Generates an image by stitching a number of google map tiles together.
#
#             Args:
#                 start_x:        The top-left x-tile coordinate
#                 start_y:        The top-left y-tile coordinate
#                 tile_width:     The number of tiles wide the image should be -
#                                 defaults to 5
#                 tile_height:    The number of tiles high the image should be -
#                                 defaults to 5
#             Returns:
#                 A high-resolution Goole Map image.
#         """
#
#         start_x = kwargs.get('start_x', None)
#         start_y = kwargs.get('start_y', None)
#         tile_width = kwargs.get('tile_width', 5)
#         tile_height = kwargs.get('tile_height', 5)
#
#         # Check that we have x and y tile coordinates
#         if start_x == None or start_y == None:
#             start_x, start_y = self.getXY()
#
#         print('start_x: %s, start_y: %s, tile_width: %s, tile_height: %s '
#               % (str(start_x), str(start_y), str(tile_width), str(tile_height)))
#
#         # # Determine the size of the image
#         width, height = 400, 400  # 256 * tile_width, 256 * tile_height
#         print('Image width and height: ', width, height)
#         # # Create a new image of the size require
#         map_img = Image.new('RGB', (width, height))
#         #
#         for x in range(0, tile_width):
#             for y in range(0, tile_height):
#                 url = 'https://mt0.google.com/vt?x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
#                         self._zoom)
#
#                 current_tile = str(x) + '-' + str(y)
#                 urllib.request.urlretrieve(url, current_tile)
#
#                 im = Image.open(current_tile)
#                 map_img.paste(im, (x * 256, y * 256))
#
#                 os.remove(current_tile)
#
#         return map_img
#
#
# # 41.89748	-87.67867
# #
# #
# #
# #
# # https://maps.googleapis.com/maps/api/staticmap?center=42.0245	-87.79775&zoom=20&size=260x260&maptype=satellite&key=AIzaSyAKs5HIZPt-dCHaglfjpqTGSNYOhMj4GVU
#
#
#
# def main():
#     # Create a new instance of GoogleMap Downloader
#     gmd = GoogleMapDownloader(40.714728, -73.998672, 20)
#
#     print("The tile coorindates are {}".format(gmd.getXY()))
#
#     try:
#         # Get the high resolution image
#         img = gmd.generateImage()
#     except IOError:
#         print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
#     else:
#         # Save the image to disk
#         img.save("high_resolution_image.png")
#         print("The map has successfully been created")
#
#
# if __name__ == '__main__':  main()
