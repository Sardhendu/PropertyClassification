from __future__ import division, print_function, absolute_import

import os
import json
# from jsonmerge
import pandas as pd
import numpy as np
import itertools
from decimal import Decimal



def collateData(filesPath, fileType="csv"):
    fileDirList = [os.path.join(filesPath, files) for files in
                   os.listdir(filesPath) if files.endswith('.%s' % fileType)]
    if fileType == "csv":
        rowList = []
        # fileDirList = [files for files in listdir if files.endswith('.csv')]
        for files in fileDirList:
            df = pd.read_csv(files, header=0)
            rowList.append(df)
        return pd.concat(rowList)
    elif fileType == "json":
        dataOUT = {}
        for files in fileDirList:
            jsonFilepath = os.path.join(filesPath, files)
            with open(jsonFilepath, 'r') as fileIN:
                dataOUT = merge(dataOUT, json.load(fileIN))
        return dataOUT

def getscoopLonLat(lonIN, latIN, decimalPlaces):
    '''
        :param lonIN: The input latitude value
        :param latIN: The input Longitude Value
        :return: The curtailed scoop

    For example;
        if lonIN = 1.123567
        if latIN = 2.2345678
        if decimalPlaces = 1000
        Return ['1.123', '2.234']
    '''
    
    # Note: We use Decimal to avoid values such as 2.565 being converted to 2.56499999999998
    # or similar
    lon_trunc = Decimal(str(int(lonIN * decimalPlaces))) / Decimal(str(float(decimalPlaces)))
    lat_trunc = Decimal(str(int(latIN * decimalPlaces))) / Decimal(str(float(decimalPlaces)))
    
    # print (lon_trunc, lat_trunc)
    scoopLonLat = [str(lon_trunc), str(lat_trunc)]
    
    return scoopLonLat



def getscoopSearchItems(scoopLon, scoopLat, decimalPlaces):
    '''
        :param scoopLon: The input curtailed scoop of longitude:
        :param scoopLat: The input curtailed scoop for latitude
        :return: A permutation of all curtailed lat lon +- 1

    For example;
        if scoopLon = 1.123
        if scoopLon = 2.234
        if decimalPlaces = 1000
        Return [(1.122,2.233), (1.122,2.234), (1.122,2.235),
                (1.123,2.233), (1.123,2.234), (1.123,2.235),
                (1.124,2.233), (1.124,2.234), (1.124,2.235)]
    '''
    rangeLon = np.array([str(Decimal(scoopLon) - Decimal(str(1 / decimalPlaces))),
                         str(scoopLon),
                         str(Decimal(scoopLon) + Decimal(str(1 / decimalPlaces)))])
    
    rangeLat = np.array([str(Decimal(scoopLat) - Decimal(str(1 / decimalPlaces))),
                         str(scoopLat),
                         str(Decimal(scoopLat) + Decimal(str(1 / decimalPlaces)))])
    
    return np.array(list(itertools.product(rangeLon, rangeLat)))