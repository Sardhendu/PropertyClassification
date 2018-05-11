import os
import pickle
import logging
import h5py
import numpy as np


def dumpPickleFile(dataX, dataY, labelDict=None, folderPath=None, fileName=None):
    if not folderPath or not fileName:
        raise ValueError('You should provide a folder path and pickle file name to dump your file')
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    path_to_dump = os.path.join(folderPath, fileName +'.pickle')
    
    logging.info('DATA FORMATTER: Dumping the pickle file %s to disk, dataX shape = %s, dataY shape = %s',
                 str(fileName),
                 str(dataX.shape),
                 str(dataY.shape))
    
    with open(path_to_dump, 'wb') as f:
        fullData = {
            'dataX': dataX,
            'dataY': dataY,
            'labelDict': labelDict
        }
        pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)


def dumpCSVFile(dataDF, folderPath, csvfileName):
    path = os.path.join(folderPath, csvfileName)
    dataDF.to_csv(path, index=None)


def getPickleFile(folderPath, fileName):
    path_from = os.path.join(folderPath, fileName+'.pickle')

    with open(path_from, "rb") as p:
        data = pickle.load(p)
        dataX = data['dataX']
        dataY = data['dataY']
        labelDict = data['labelDict']
    
    logging.info('DATA FORMATTER: Retrieved the pickle file %s from disk, dataX shape = %s, dataY shape = %s',
                 str(fileName),
                 str(dataX.shape),
                 str(dataY.shape))

    return dataX, dataY, labelDict


def dumpH5File(dataX, dataY, folderPath=None, fileName=None):
    if not folderPath or not fileName:
        raise ValueError('You should provide a folder path and hdf5 file name to dump your file')
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    path_to_dump = os.path.join(folderPath, fileName+'.h5')
    
    logging.info('DATA FORMATTER: Dumping the hdf5 file %s to disk, dataX shape = %s, dataY shape = %s',
                 str(fileName),
                 str(dataX.shape),
                 str(dataY.shape))

    with h5py.File(path_to_dump, 'w') as hf:
        hf.create_dataset("dataX", data=dataX)
        hf.create_dataset("dataY", data=dataY)


def getH5File(folderPath, fileName):
    path_from = os.path.join(folderPath, fileName + '.h5')
    with h5py.File(path_from,'r') as hf:
        dataX = np.array(hf.get('dataX'), dtype='float32')
        dataY = np.array(hf.get('dataY'), dtype='float32')
    logging.info('DATA FORMATTER: Retrieved the hdf5 file %s from disk, dataX shape = %s, dataY shape = %s',
                 str(fileName),
                 str(dataX.shape),
                 str(dataY.shape))

    return dataX, dataY


#
# debugg = False
# if debugg:
#     pathh = '/Users/sam/All-Program/App-DataSet/HouseClassification/'
#     filename= 'plplpl'
#     import numpy as np
#     labelDict = {'0':'land'}
#     dataX = np.random.random((10,10))
#     dataY = np.append(np.ones(5), np.zeros(5))
#     print (dataX)
#     dumpH5File(dataX, dataY, folderPath=pathh, fileName=filename)
#
#     getH5File(pathh, filename)
#     print ('')
#     print (dataX)
#     print ('')
#     print (dataY)