import os
import pickle
import logging


def dumpPickleFile(dataX, dataY, labelDict=None, folderPath=None, picklefileName=None):
    if not folderPath or not picklefileName:
        raise ValueError('You should provide a folder path and pickle file name to dump your file')
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    path_to_dump = os.path.join(folderPath, picklefileName)
    
    logging.info('DATA FORMATTER: Dumping the pickle file %s to disk, dataX shape = %s, dataY shape = %s',
                 str(picklefileName),
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


def getPickleFile(folderPath, picklefileName):
    path_from = os.path.join(folderPath, picklefileName)

    with open(path_from, "rb") as p:
        data = pickle.load(p)
        dataX = data['dataX']
        dataY = data['dataY']
        labelDict = data['labelDict']
    
    logging.info('DATA FORMATTER: Retrieved the pickle file %s from disk, dataX shape = %s, dataY shape = %s',
                 str(picklefileName),
                 str(dataX.shape),
                 str(dataY.shape))

    return dataX, dataY, labelDict