
def genRandomStratifiedBatches(img_resize_shape, valid_land_pins, valid_house_pins, batch_size, image_type='aerial',
                               dump=True):
    '''
    :param img_resize_shape:
    :param valid_land_pins:
    :param valid_house_pins:
    :param batch_size:
    :return:

    The idea of the random stratified batch generation is that given a list of pins (unique identifier for a property
    - land, house), the data is distributed such that every batch has same number of each labels.

    The complexity lies is handling imbalanced class. Since this is a image processing problem, repeating data with a
    little distortion would not affect the model. Hence is label 1 exceeds the count of label 0, then the code would
    actually re sample random data from the smaller label dataset and put them into the batch.
    '''
    
    if image_type not in ['assessor', 'aerial']:
        raise ValueError('Variable image_type not understood')
    
    land_image_path = os.path.join(pathDict[ '%s_image_path' %(image_type)], 'land')
    house_image_path = os.path.join(pathDict[ '%s_image_path' %(image_type)], 'house')
    output_batch_path = pathDict[ '%s_batch_path' %(image_type)]
    
    land_label = 0
    house_label = 1
    label_dict = {'0' :'land', '1' :'house'}
    land_pins = np.sort(np.intersect1d
        (np.array([img.split('.')[0] for img in os.listdir(land_image_path) if img != '.DS_Store'], dtype=str),
         valid_land_pins))
    
    house_pins = np.sort(np.intersect1d(
            np.array([img.split('.')[0] for img in os.listdir(house_image_path) if img != '.DS_Store'], dtype=str),
            valid_house_pins))
    
    my_seed = 873
    np.random.seed(my_seed)
    np.random.shuffle(land_pins)
    np.random.shuffle(house_pins)
    
    # print (len(land_pins), len(house_pins))
    
    if len(land_pins) > len(house_pins):
        bigger, b_label = land_pins, land_label
        smaller, s_label = house_pins, house_label
        b_land, b_house = True, False
    else:
        bigger, b_label = house_pins, house_label
        smaller, s_label = land_pins, land_label
        b_house, b_land = True, False
    
    
    img_per_label_per_batch = batch_size // 2
    
    num_batches = int(np.ceil(len(bigger ) /img_per_label_per_batch))
    
    dataBatchX = np.ndarray(shape=(num_batches, batch_size,
                                   img_resize_shape[0],
                                   img_resize_shape[1], 3), dtype='float32')
    dataBatchY = np.ndarray(shape=(num_batches, batch_size), dtype='float32')
    
    count = 1
    pin_batch_row_meta = None
    for batch_num in range(0 ,num_batches):
        logging.info('Creating BATCH %s .......... ', str(batch_num))
        idx1 = batch_num* img_per_label_per_batch
        idx2 = batch_num * img_per_label_per_batch + img_per_label_per_batch
        # print (idx1, idx2)
        if len(bigger) >= idx2:
            logging.info('Bigger: len(bigger) >= idx2 ')
            b_batch_pins = bigger[idx1: idx2]
        elif len(bigger) > idx1 and len(bigger) < idx2:
            logging.info('Bigger: idx1 < len(bigger) < idx2 ')
            b_batch_pins = bigger[idx1:len(bigger)]
            how_many = idx2 - len(bigger)
            random.seed(my_seed + count)
            b_idx = random.sample(range(0, len(bigger)), how_many)
            b_batch_pins = np.append(b_batch_pins, bigger[b_idx])
            count += 1
        else:
            logging.info('Bigger: len(bigger) < idx1 < idx2 ')
            how_many = img_per_label_per_batch
            random.seed(my_seed + count)
            b_idx = random.sample(range(0, len(bigger)), how_many)
            b_batch_pins = smaller[b_idx]
            count += 1
        
        if len(smaller) >= idx2:
            logging.info('Smaller: len(Smaller) >= idx2 ')
            s_batch_pins = smaller[idx1: idx2]
        elif len(smaller) > idx1 and len(smaller) < idx2:
            logging.info('Smaller: idx1 < len(Smaller) < idx2 ')
            s_batch_pins = smaller[idx1:len(smaller)]
            how_many = idx2 - len(smaller)
            random.seed(my_seed + count)
            s_idx = random.sample(range(0, len(smaller)), how_many)
            s_batch_pins = np.append(s_batch_pins, smaller[s_idx])
            count += 1
        else:
            logging.info('Smaller: len(Smaller) < idx1 < idx2 ')
            how_many = img_per_label_per_batch
            random.seed(my_seed + count)
            s_idx = random.sample(range(0, len(smaller)), how_many)
            s_batch_pins = smaller[s_idx]
            count += 1
        
        # GET IMAGE PATH
        if b_land:
            paths = [os.path.join(land_image_path, pin + '.jpg') for pin in b_batch_pins] + [
                os.path.join(house_image_path, pin + '.jpg') for pin in s_batch_pins]
            label = np.array(['land'] * len(b_batch_pins) + ['house'] * len(s_batch_pins))
        else:
            paths = [os.path.join(house_image_path, pin + '.jpg') for pin in b_batch_pins] + [
                os.path.join(land_image_path, pin + '.jpg') for pin in s_batch_pins]
            label = np.array(['house'] * len(b_batch_pins) + ['land'] * len(s_batch_pins))
        
        # We would like to create a metadata table that hods the list of pins and their corresponding batch number
        # and record number, so that we can perform some testing and validation.
        pins = np.vstack((b_batch_pins.reshape(-1, 1), s_batch_pins.reshape(-1, 1)))
        batch_no = np.tile(batch_num, len(pins))
        rownum = np.arange(len(batch_no))
        
        if batch_num == 0:
            pin_batch_row_meta = np.column_stack(
                    (pins, batch_no.reshape(-1, 1), rownum.reshape(-1, 1), label.reshape(-1, 1)))
        else:
            pin_batch_row_meta = np.vstack((pin_batch_row_meta,
                                            np.column_stack((pins, batch_no.reshape(-1, 1), rownum.reshape(-1, 1),
                                                             label.reshape(-1, 1)))))
        # print(pins.shape, batch_no.shape, rownum.shape, label.shape)
        # print (batch_no,'\n')
        # print (rownum, '\n')
        # print (label, '\n')
        # print ('')
        # print ('')
        
        # GET BATCH IMAGE ARRAY
        img_arr = []
        for pic_path in paths:
            image = ndimage.imread(pic_path, mode='RGB')
            image = resize_image(image, img_resize_shape)
            img_arr.append(image)
        
        dataBatchX[batch_num, :] = img_arr
        dataBatchY[batch_num, :] = np.append(
                np.tile(float(b_label), img_per_label_per_batch),
                np.tile(float(s_label), img_per_label_per_batch))
    
    if dump:
        # DUMP THE BATCH DATA
        if not os.path.exists(output_batch_path):
            os.makedirs(output_batch_path)
        logging.info('The Data batches dumped has shape: %s', str(dataBatchX.shape))
        logging.info('The Label batch dumped has shape: %s', str(dataBatchY.shape))
        dumpPickleFile(dataX=dataBatchX,
                       dataY=dataBatchY,
                       labelDict=label_dict,
                       folderPath=output_batch_path,
                       picklefileName=fileNames['batch_img_file'])
        
        # DUMP THE PIN_BATCH_ROW_META
        pin_batch_row_meta = pd.DataFrame(pin_batch_row_meta, columns=['pin', 'batch_no', 'rownum', 'indicator'])
        pin_batch_row_meta.to_csv(
            os.path.join(pathDict['pin_batch_row_meta_path'], '%s_pin_batch_row_meta.csv' % (image_type)), index=None)




