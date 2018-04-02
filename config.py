

import os
import platform
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
#
# api_call['zillow_zid'] = 'xyz'
# api_call['bing_key'] = 'xyz'
# api_call['google_streetside_key'] = 'xyz'
# api_call['google_aerial_key'] = 'xyz'
# api_call['google_meta_key'] = 'xyz'




api_call['google_aerial_key'] = str(api_call['google_aerial_key'][0])
api_call['google_meta_key'] = str(api_call['google_meta_key'][0])




#################### Seed Arrays
seed_arr = [553, 292, 394, 874, 445, 191, 161, 141, 213,436,754,991,302,992,223,645,724,944,
            232,123,321, 909,784,239,337,888,666, 400,912,255,983,902,846,345,
            854,989,291,486,444,101,202,304,505,607,707,808,905, 900, 774,272]

####################    PREPROCESSING PARAMETERS
pp_vars = {}

# image_type = input('Input your image type OPTIONS: (assessor, assessor_code, google_aerial, google_overlayed \n IMAGE_TYPE = ')
image_type = 'assessor'
if image_type in ['assessor', 'google_aerial', 'bing_aerial', 'bing_streetside', 'google_overlayed']:
    pp_vars['standardise'] = True
    pp_vars['rand_brightness'] = False
    pp_vars['rand_contrast'] = False
    pp_vars['rand_rotate'] = False
    pp_vars['rand_flip'] = True
    pp_vars['rand_crop'] = False
    pp_vars['central_crop'] = True
elif image_type == 'assessor_code':
    pp_vars['standardise'] = True
    pp_vars['rand_brightness'] = False
    pp_vars['rand_contrast'] = False
    pp_vars['rand_rotate'] = False
    pp_vars['rand_flip'] = False
    pp_vars['rand_crop'] = False
    pp_vars['central_crop'] = True
else:
    raise ValueError('Please provide a valid image_type !!')



#####################   NET PARAMETERS
myNet['num_labels'] = 2
myNet['optimizer'] = 'RMSPROP'
myNet['learning_rate'] = 0.005
myNet['momentum'] = 0.9
myNet['learning_rate_decay_rate'] = 0.95
myNet['batch_norm_decay'] = 0.9
myNet['batch_size'] = 128
myNet['lr_decay_steps'] = 15000  # how many examples to see before making a decay
# If you are learning a very complex function then setting lr_decay_steps = train_size, makes sense. But if the
# function is not very complex and you feet that the function can be marginally learned in 1-3 steps than set it to
# train_size/5 or somthing like that. This would ensure that the high learning rate doesnt make the optimization just
# from minimas.

#####################   BATCH PARAMETERS



#################### CONV parameters
# netParams['conv1']['conv_shape'] = [3,3,3,64]
# netParams['conv1']['conv_stride'] = 1
# netParams['conv1']['conv_pad'] = 'SAME'
# netParams['conv1']['pool_size'] = 2
# netParams['conv1']['pool_stride'] = 2
# netParams['conv1']['pool_pad'] = 'SAME'
# netParams['conv1']['keep_prob'] = 0.5
#
# netParams['conv2']['conv_shape'] = [3,3,64,128]
# netParams['conv2']['conv_stride'] = 1
# netParams['conv2']['conv_pad'] = 'SAME'
# netParams['conv2']['pool_size'] = 2
# netParams['conv2']['pool_stride'] = 2
# netParams['conv2']['pool_pad'] = 'SAME'
# netParams['conv2']['keep_prob'] = 0.5
#
# netParams['conv3']['conv_shape'] = [3,3,128,256]
# netParams['conv3']['conv_stride'] = 1
# netParams['conv3']['conv_pad'] = 'SAME'
# netParams['conv3']['pool_size'] = 2
# netParams['conv3']['pool_stride'] = 2
# netParams['conv3']['pool_pad'] = 'SAME'
# netParams['conv3']['keep_prob'] = 0.5
#
# netParams['conv4']['conv_shape'] = [3,3,256,256]
# netParams['conv4']['conv_stride'] = 1
# netParams['conv4']['conv_pad'] = 'SAME'
# netParams['conv4']['pool_size'] = 2
# netParams['conv4']['pool_stride'] = 2
# netParams['conv4']['pool_pad'] = 'SAME'
# netParams['conv4']['keep_prob'] = 0.5
#
# netParams['fc1']['shape'] = [None, 1280]
# netParams['fc1']['keep_prob'] = 0.5
# netParams['fc2']['shape'] = [1280, 1280]
# netParams['fc2']['keep_prob'] = 0.5
# netParams['fc3']['shape'] = [1280, 1000]
# netParams['fc3']['keep_prob'] = 0.8
#
# netParams['softmax']['shape'] = [1000, 2]



##################### Other important



#####################   IMAGE/MODEL PATH
fileNames = {}
fileNames['rsized_img_file'] = 'resized_image_arr.pickle'
fileNames['batch_img_file'] = 'batch_img_arr.pickle'


#####################   IMAGE PATHSx
pathDict = {}

if platform.platform().split('-')[0] == 'Darwin':
    pathDict['parent_path'] = "/Users/sam/All-Program/App-DataSet/HouseClassification/training_data"
else:
    pathDict['parent_path'] = r"C:\Users\newline\Documents\ImageClassification\data\training_data"
pathDict['statistics_path'] = os.path.join(pathDict['parent_path'], "statistics")
pathDict['data_model_path'] = os.path.join(pathDict['parent_path'], 'data_models')
pathDict['pin_batch_row_meta_path'] = os.path.join(pathDict['statistics_path'], 'pin_batch_row_meta')
pathDict['chicago_bbox_shp_files'] = os.path.join(pathDict['parent_path'], "shape_files", "building_bbox")



##### Assessor Images
pathDict['assessor_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "assessor_images")
# pathDict['assessor_rsized_path'] = os.path.join(pathDict['data_model_path'], "assessor_images")
pathDict['assessor_batch_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'batch_data')
pathDict['assessor_ckpt_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'checkpoint')
pathDict['assessor_smry_path'] = os.path.join(pathDict['data_model_path'], "assessor_images", 'summary')
pathDict['assessor_pred_stats'] = os.path.join(pathDict['statistics_path'], 'prediction_stats', 'assessor_images')

##### Assessor Code Images
# Assessor ad assessor_code are same, however, assessor_code images are sent to Autoencoder to learn a latent
# representation of the input image.
pathDict['assessor_code_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "assessor_code_images")
# pathDict['assessor_code_rsized_path'] = os.path.join(pathDict['data_model_path'], "assessor_code_images")
pathDict['assessor_code_batch_path'] = os.path.join(pathDict['data_model_path'], "assessor_code_images", 'batch_data')
pathDict['assessor_code_ckpt_path'] = os.path.join(pathDict['data_model_path'], "assessor_code_images", 'checkpoint')
pathDict['assessor_code_smry_path'] = os.path.join(pathDict['data_model_path'], "assessor_code_images", 'summary')
pathDict['assessor_code_pred_stats'] = os.path.join(pathDict['statistics_path'], 'prediction_stats',
                                                    'assessor_code_images')


##### Aerial Images from Google
pathDict['aerial_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "aerial_images")
pathDict['aerial_batch_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google",'batch_data')
pathDict['aerial_ckpt_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google", 'checkpoint')
pathDict['aerial_smry_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "google", 'summary')
pathDict['aerial_pred_stats'] = os.path.join(pathDict['statistics_path'], 'prediction_stats', 'aerial_images')
pathDict['aerial_stats_path'] = os.path.join(pathDict['statistics_path'], 'aerial_collected_data_stat')


# OVERLAYED IMAGE PATHS
pathDict['overlayed_image_path'] = os.path.join(pathDict['parent_path'],"input_images","overlayed_images")
pathDict['overlayed_batch_path'] = os.path.join(pathDict['data_model_path'],"overlayed_images","google",'batch_data')
pathDict['overlayed_ckpt_path'] = os.path.join(pathDict['data_model_path'], "overlayed_images", "google", 'checkpoint')
pathDict['overlayed_smry_path'] = os.path.join(pathDict['data_model_path'], "overlayed_images", "google", 'summary')
pathDict['overlayed_pred_stats'] = os.path.join(pathDict['statistics_path'], 'prediction_stats',
                                                       'overlayed_images')

# AERIAL CROPPED IMAGE PATHS
pathDict['aerial_cropped_image_path'] = os.path.join(pathDict['parent_path'],"input_images","aerial_cropped_images")
pathDict['aerial_cropped_batch_path'] = os.path.join(pathDict['data_model_path'],"aerial_cropped_images","google",'batch_data')
pathDict['aerial_cropped_ckpt_path'] = os.path.join(pathDict['data_model_path'], "aerial_cropped_images", "google", 'checkpoint')
pathDict['aerial_cropped_path'] = os.path.join(pathDict['data_model_path'], "aerial_cropped_images", "google", 'summary')
pathDict['aerial_cropped_pred_stats'] = os.path.join(pathDict['statistics_path'], 'prediction_stats',
                                                       'aerial_cropped_images')

##### Other new directory
pathDict['assessor_code_house_path'] = os.path.join(pathDict['parent_path'], 'input_images','assessor_code_images','house')
pathDict['assessor_code_land_path'] = os.path.join(pathDict['parent_path'], 'input_images', 'assessor_code_images','land')






##### Aerial Images from Bing
# pathDict['bing_aerial_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "aerial_images", "bing")
# pathDict['bing_aerial_stats_path'] = os.path.join(pathDict['statistics_path'], "aerial_images", "bing")
# pathDict['bing_aerial_batch_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'batch_data')
# pathDict['bing_aerial_ckpt_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'checkpoint')
# pathDict['bing_aerial_smry_path'] = os.path.join(pathDict['data_model_path'], "aerial_images", "bing", 'summary')

##### Streetside Images
# pathDict['bing_streetside_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "streetside_images", "bing")
# pathDict['google_streetside_image_path'] = os.path.join(pathDict['parent_path'], "input_images", "streetside_images", "google")
# pathDict['streetside_rsized_path'] = os.path.join(pathDict['data_model_path'], "streetside_images")
# pathDict['streetside_batch_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'batch_data')
# pathDict['streetside_ckpt_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'checkpoint')
# pathDict['streetside_smry_path'] = os.path.join(pathDict['data_model_path'], "streetside_images", 'summary')




