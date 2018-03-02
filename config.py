

import os
from collections import defaultdict
global api_call
global myNet
global pathDict
global fileNames


##################### Empty Global variables declaration
weight_seed_idx = 0
preprocess_seed_idx = 0




##################### External Data Fetch API keys

api_call ={}
myNet = {}
netParams=defaultdict(lambda: defaultdict())

api_call['zillow_zid'] = 'xyz'
api_call['bing_key'] = 'xyz'
api_call['google_streetside_key'] = 'xyz'
api_call['google_aerial_key'] = 'xyz'
api_call['google_meta_key'] = 'xyz'

api_call['google_aerial_key'] = str(api_call['google_aerial_key'][2])
api_call['google_meta_key'] = str(api_call['google_meta_key'][2])




#################### Seed Arrays
seed_arr = [553, 292, 394, 874, 445, 191, 161, 141, 213,436,754,991,302,992,223,645,724,944,
            232,123,321, 909,784,239,337,888,666, 400,912,255,983,902,846,345,
            854,989,291,486,444,101,202,304,505,607,707,808,905, 900, 774,272]

####################    PREPROCESSING PARAMETERS
pp_vars = {}
pp_vars['standardise'] = True
pp_vars['rand_brightness'] = False
pp_vars['rand_contrast'] = False
pp_vars['rand_rotate'] = False
pp_vars['rand_flip'] = True
pp_vars['rand_crop'] = False
pp_vars['central_crop'] = True



#####################   NET PARAMETERS
# myNet['image_shape'] = [160, 160, 3]
# myNet['crop_shape'] = [128, 128, 3]
myNet['image_shape'] = [260, 260, 3]
myNet['crop_shape'] = [224, 224, 3]
myNet['num_labels'] = 2
myNet['optimizer'] = 'ADAM'
myNet['learning_rate'] = 0.0009
myNet['momentum'] = 0.9
myNet['learning_rate_decay_rate'] = 0.95
myNet['batch_norm_decay'] = 0.9


#####################   BATCH PARAMETERS

vars = {}
# vars['epochs'] = 20
# vars['num_batches'] = 10
vars['num_img_per_label'] = 5000
vars['batch_size'] = 64
vars['train_size'] = vars['num_img_per_label'] * myNet['num_labels'] - vars['batch_size']



#################### CONV parameters
netParams['conv1']['conv_shape'] = [3,3,3,64]
netParams['conv1']['conv_stride'] = 1
netParams['conv1']['conv_pad'] = 'SAME'
netParams['conv1']['pool_size'] = 2
netParams['conv1']['pool_stride'] = 2
netParams['conv1']['pool_pad'] = 'SAME'
netParams['conv1']['keep_prob'] = 0.5

netParams['conv2']['conv_shape'] = [3,3,64,128]
netParams['conv2']['conv_stride'] = 1
netParams['conv2']['conv_pad'] = 'SAME'
netParams['conv2']['pool_size'] = 2
netParams['conv2']['pool_stride'] = 2
netParams['conv2']['pool_pad'] = 'SAME'
netParams['conv2']['keep_prob'] = 0.5

netParams['conv3']['conv_shape'] = [3,3,128,256]
netParams['conv3']['conv_stride'] = 1
netParams['conv3']['conv_pad'] = 'SAME'
netParams['conv3']['pool_size'] = 2
netParams['conv3']['pool_stride'] = 2
netParams['conv3']['pool_pad'] = 'SAME'
netParams['conv3']['keep_prob'] = 0.5

netParams['conv4']['conv_shape'] = [3,3,256,256]
netParams['conv4']['conv_stride'] = 1
netParams['conv4']['conv_pad'] = 'SAME'
netParams['conv4']['pool_size'] = 2
netParams['conv4']['pool_stride'] = 2
netParams['conv4']['pool_pad'] = 'SAME'
netParams['conv4']['keep_prob'] = 0.5

netParams['fc1']['shape'] = [None, 1280]
netParams['fc1']['keep_prob'] = 0.5
netParams['fc2']['shape'] = [1280, 1280]
netParams['fc2']['keep_prob'] = 0.5
netParams['fc3']['shape'] = [1280, 1000]
netParams['fc3']['keep_prob'] = 0.8

netParams['softmax']['shape'] = [1000, 2]



##################### Other important



#####################   IMAGE/MODEL PATH
fileNames = {}
fileNames['rsized_img_file'] = 'resized_image_arr.pickle'
fileNames['batch_img_file'] = 'batch_img_arr.pickle'


#####################   IMAGE PATHS
pathDict = {}
pathDict['parent_path'] = "/Users/sam/All-Program/App-DataSet/HouseClassification/"
pathDict['statistics_path'] = os.path.join(pathDict['parent_path'], "statistics")
pathDict['data_model_path'] = os.path.join(pathDict['parent_path'], 'data_models')
pathDict['pin_batch_row_meta_path'] = os.path.join(pathDict['statistics_path'], 'pin_batch_row_meta')

##### Aerial Images from Google
pathDict['google_aerial_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "aerial_images", "google")
pathDict['google_overlayed_image_path'] = os.path.join(pathDict['parent_path'],"input_images","overlayed_images","google")
pathDict['google_aerial_stats_path'] = os.path.join(pathDict['statistics_path'], "aerial_images", "google")
# pathDict['aerial_rsized_path'] = os.path.join(pathDict['data_model_path'], "aerial_images")
pathDict['google_aerial_batch_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google",'batch_data')
pathDict['google_aerial_ckpt_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google", 'checkpoint')
pathDict['google_aerial_smry_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google", 'summary')

##### Aerial Images from Bing
pathDict['bing_aerial_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "aerial_images", "bing")
pathDict['bing_aerial_stats_path'] = os.path.join(pathDict['statistics_path'], "aerial_images", "bing")
# pathDict['aerial_rsized_path'] = os.path.join(pathDict['data_model_path'], "aerial_images")
pathDict['bing_aerial_batch_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'batch_data')
pathDict['bing_aerial_ckpt_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'checkpoint')
pathDict['bing_aerial_smry_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'summary')



##### Assessor Images
pathDict['assessor_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "assessor_images")
pathDict['assessor_dl_stats_path'] = os.path.join(pathDict['statistics_path'], "assessor_images", 'data_loader')
pathDict['assessor_dp_stats_path'] = os.path.join(pathDict['statistics_path'], "assessor_images", 'data_prep')
pathDict['assessor_rsized_path'] = os.path.join(pathDict['data_model_path'], "assessor_images")
pathDict['assessor_batch_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'batch_data')
pathDict['assessor_ckpt_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'checkpoint')
pathDict['assessor_smry_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'summary')



##### Streetside Images
pathDict['streetside_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "streetside_images")
pathDict['streetside_dl_stats_path'] = os.path.join(pathDict['statistics_path'], "streetside_images", 'data_loader')
pathDict['streetside_dp_stats_path'] = os.path.join(pathDict['statistics_path'], "streetside_images", 'data_prep')
pathDict['streetside_rsized_path'] = os.path.join(pathDict['data_model_path'], "streetside_images")
pathDict['streetside_batch_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'batch_data')
pathDict['streetside_ckpt_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'checkpoint')
pathDict['streetside_smry_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'summary')


