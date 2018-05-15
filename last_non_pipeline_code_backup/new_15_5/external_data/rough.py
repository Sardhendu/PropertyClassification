
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





