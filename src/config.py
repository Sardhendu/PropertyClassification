

import os
import platform
from collections import defaultdict
# global conf['api_call']
# global conf['myNet']
# global conf['pp_vars']
# global conf['pathDict']
# global conf['fileNames']


##################### Empty Global variables declaration





##################### External Data Fetch API keys

# conf['api_call'] ={}
# conf['myNet'] = {}
# conf['pp_vars'] = {}
# conf['fileNames'] = {}
# conf['pathDict'] = {}



#
# conf['api_call']['zillow_zid'] = 'xyz'
# conf['api_call']['bing_key'] = 'xyz'
# conf['api_call']['google_streetside_key'] = 'xyz'
# conf['api_call']['google_aerial_key'] = 'xyz'
# conf['api_call']['google_meta_key'] = 'xyz'






#################### Seed Arrays





    
####################    PREPROCESSING PARAMETERS


# which_run = input('INPUT the RUN NAME: Options : ("A new run name" or "Any Previous Run Name"\n '
#                   'WHICH_RUN = ')
#
# img_type = input('INPUT: Image-type OPTIONS: (assessor, assessor_code, aerial, overlaid, aerial_cropped, '
#                    'streetside and ensemble \n '
#                    'img_type = ')



def get_config(which_run, img_type):
    conf = defaultdict(lambda: defaultdict(list))
    
    if img_type not in ['assessor', 'aerial', 'overlaid', 'aerial_cropped', 'assessor_encode', 'streetside',
                          'mixture_model']:
        raise ValueError("The image type doesnt match the used types, Here are few options options: ('assessor', 'aerial', 'overlaid', 'aerial_cropped', 'assessor_encode', 'streetside')")

    conf['weight_seed_idx'] = 0
    conf['preprocess_seed_idx'] = 0
    
    # Seed Arr:
    conf['seed_arr'] = [553, 292, 394, 874, 445, 191, 161, 141, 213, 436, 754, 991, 302, 992,
                        223, 645, 724, 944,
                        232, 123, 321, 909, 784, 239, 337, 888, 666, 400, 912, 255, 983, 902,
                        846, 345, 854, 989, 291, 486, 444, 101, 202, 304, 505, 607, 707, 808,
                        905, 900, 774, 272]
    
    # API's
    conf['api_call']['zillow_zid'] = 'xyz'
    conf['api_call']['bing_key'] = 'xyz'
    conf['api_call']['google_streetside_key'] = 'xyz'
    conf['api_call']['google_aerial_key'] = 'xyz'
    conf['api_call']['google_meta_key'] = 'xyz'
    

    conf['api_call']['google_aerial_key'] = str(conf['api_call']['google_aerial_key'][1])
    conf['api_call']['google_meta_key'] = str(conf['api_call']['google_meta_key'][1])
    
    
    # img_type = 'aerial'
    if img_type in ['assessor', 'aerial', 'streetside', 'overlaid', 'aerial_cropped']:
        conf['pp_vars']['standardise'] = True
        conf['pp_vars']['rand_brightness'] = False
        conf['pp_vars']['rand_contrast'] = False
        conf['pp_vars']['rand_rotate'] = False
        conf['pp_vars']['rand_Hflip'] = True
        conf['pp_vars']['rand_Vflip'] = True
        conf['pp_vars']['rand_crop'] = False
        conf['pp_vars']['central_crop'] = False
    
        conf['myNet']['num_labels'] = 2
        conf['myNet']['optimizer'] = 'ADAM'
        conf['myNet']['learning_rate'] = 0.0005
        conf['myNet']['momentum'] = 0.9
        conf['myNet']['learning_rate_decay_rate'] = 0.95
        conf['myNet']['batch_norm_decay'] = 0.9
        conf['myNet']['batch_size'] = 128
        conf['myNet']['lr_decay_steps'] = 5000  # how many examples to see before making a decay
        # If you are learning a very complex function then setting lr_decay_steps = train_size, makes sense. But if the
        # function is not very complex and you feet that the function can be marginally learned in 1-3 steps than set it to
        # train_size/5 or somthing like that. This would ensure that the high learning rate doesnt make the optimization
        # just from minimas.
    elif img_type =='mixture_model':
        conf['pp_vars']['standardise'] = True
        conf['pp_vars']['rand_brightness'] = False
        conf['pp_vars']['rand_contrast'] = False
        conf['pp_vars']['rand_rotate'] = False
        conf['pp_vars']['rand_Hflip'] = True
        conf['pp_vars']['rand_Vflip'] = True
        conf['pp_vars']['rand_crop'] = False
        conf['pp_vars']['central_crop'] = False
    
        conf['myNet']['num_labels'] = 2
        conf['myNet']['optimizer'] = 'ADAM'
        conf['myNet']['learning_rate'] = 0.0005
        conf['myNet']['momentum'] = 0.9
        conf['myNet']['learning_rate_decay_rate'] = 0.95
        conf['myNet']['batch_norm_decay'] = 0.9
        conf['myNet']['batch_size'] = 128
        conf['myNet']['lr_decay_steps'] = 9000
        
    elif img_type == 'assessor_code':
        conf['pp_vars']['standardise'] = True
        conf['pp_vars']['rand_brightness'] = False
        conf['pp_vars']['rand_contrast'] = False
        conf['pp_vars']['rand_rotate'] = False
        conf['pp_vars']['rand_flip'] = False
        conf['pp_vars']['rand_crop'] = False
        conf['pp_vars']['central_crop'] = True
    
        conf['myNet']['num_labels'] = 2
        conf['myNet']['optimizer'] = 'RMSPROP'
        conf['myNet']['learning_rate'] = 0.005
        conf['myNet']['momentum'] = 0.9
        conf['myNet']['learning_rate_decay_rate'] = 0.95
        conf['myNet']['batch_norm_decay'] = 0.9
        conf['myNet']['batch_size'] = 128
        conf['myNet']['lr_decay_steps'] = 15000  # how many examples to see before making a decay
        # If you are learning a very complex function then setting lr_decay_steps = train_size, makes sense. But if the
        # function is not very complex and you feet that the function can be marginally learned in 1-3 steps than set it to
        # train_size/5 or somthing like that. This would ensure that the high learning rate doesnt make the optimization
        # just
        # from minimas.
    else:
        raise ValueError('Please provide a valid img_type !!')
    
    #####################   IMAGE/MODEL PATH
    conf['fileNames']['rsized_img_file'] = 'resized_image_arr.pickle'
    conf['fileNames']['batch_img_file'] = 'batch_img_arr.pickle'
    
    
    #####################   IMAGE PATHS
    if platform.platform().split('-')[0] == 'Darwin':
        conf['pathDict']['parent_path'] = "/Users/sam/All-Program/App-DataSet/HouseClassification/"
    else:
        conf['pathDict']['parent_path'] = r"C:\Users\newline\Documents\ImageClassification\data"
    
    
    ######## Parent Paths
    conf['pathDict']['parent_checkpoint_path'] = os.path.join(conf['pathDict']['parent_path'], "checkpoints")
    conf['pathDict']['parent_statistics_path'] = os.path.join(conf['pathDict']['parent_path'], "statistics")
    conf['pathDict']['parent_csv_path'] = os.path.join(conf['pathDict']['parent_path'], "input_csv_files")
    
    ######## COMMON PATHS
    conf['pathDict']['chicago_bbox_shp_files'] = os.path.join(conf['pathDict']['parent_path'], "shape_files", "building_bbox")
    conf['pathDict']['input_image_run_dir'] = os.path.join(conf['pathDict']['parent_path'], "input_images", which_run)
    conf['pathDict']['general_stats_path'] = os.path.join(conf['pathDict']['parent_path'], "statistics", which_run)
    conf['pathDict']['general_batch_path'] = os.path.join(conf['pathDict']['parent_path'], "batch_data", which_run)
    
    
    ######### INPUT IMAGES PATH
    
    conf['pathDict']['image_path'] = os.path.join(conf['pathDict']['input_image_run_dir'], img_type)
    conf['pathDict']['batch_path'] = os.path.join(conf['pathDict']['general_batch_path'], img_type)

    conf['pathDict']['csv_path'] = os.path.join(conf['pathDict']['parent_csv_path'], which_run, img_type)
    conf['pathDict']['checkpoint_path'] = os.path.join(conf['pathDict']['parent_checkpoint_path'], which_run, img_type)
    conf['pathDict']['summary_path'] = os.path.join(conf['pathDict']['parent_path'],"summary", which_run, img_type)
    conf['pathDict']['statistics_path'] = os.path.join(conf['pathDict']['parent_path'],"statistics", which_run, img_type)
    
    path_arr = [conf['pathDict']['csv_path'], conf['pathDict']['image_path'], conf['pathDict']['batch_path'], conf['pathDict']['checkpoint_path'], conf['pathDict']['summary_path'], conf['pathDict']['statistics_path']]
    
    for paths in path_arr:
        if not os.path.exists(paths):
            os.makedirs(paths)
            
    return conf


# get_config(which_run='babu', img_type='aerial')
# print (conf['myNet'])
# print (conf['pathDict'])




# print (conf['pathDict']['statistics_path'])
#
# print (os.path.dirname(conf['pathDict']['statistics_path']))

# conf['pathDict']['pin_batch_row_meta_path'] = os.path.join(conf['pathDict']['statistics_path'], 'pin_batch_row_meta')


#
#
# ##### Assessor Imxages
#
# # conf['pathDict']['assessor_rsized_path'] = os.path.join(conf['pathDict']['data_model_path'], "assessor_images")
# conf['pathDict']['assessor_batch_path'] = os.path.join(conf['pathDict']['run_folder'], img_type, 'batch_data')
# conf['pathDict']['assessor_ckpt_path'] = os.path.join(conf['pathDict']['run_folder'], img_type, 'checkpoint')
# conf['pathDict']['assessor_smry_path'] = os.path.join(conf['pathDict']['run_folder'], img_type, 'summary')
# conf['pathDict']['assessor_pred_stats'] = os.path.join(conf['pathDict']['run_folder'], img_type, 'assessor_images')
#
# ##### Assessor Code Images
# # Assessor ad assessor_code are same, however, assessor_code images are sent to Autoencoder to learn a latent
# # representation of the input image.
#
# # conf['pathDict']['assessor_code_rsized_path'] = os.path.join(conf['pathDict']['data_model_path'], "assessor_code_images")
# conf['pathDict']['assessor_code_batch_path'] = os.path.join(conf['pathDict']['data_model_path'], "assessor_code_images", 'batch_data')
# conf['pathDict']['assessor_code_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "assessor_code_images", 'checkpoint')
# conf['pathDict']['assessor_code_smry_path'] = os.path.join(conf['pathDict']['data_model_path'], "assessor_code_images", 'summary')
# conf['pathDict']['assessor_code_pred_stats'] = os.path.join(conf['pathDict']['statistics_path'], 'prediction_stats',
#                                                     'assessor_code_images')
#
#
# ##### Aerial Images from Google
#
# conf['pathDict']['aerial_batch_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "google",'batch_data')
# conf['pathDict']['aerial_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "google", 'checkpoint')
# conf['pathDict']['aerial_smry_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "google", 'summary')
# conf['pathDict']['aerial_pred_stats'] = os.path.join(conf['pathDict']['statistics_path'], 'prediction_stats', 'aerial_images')
# conf['pathDict']['aerial_stats_path'] = os.path.join(conf['pathDict']['statistics_path'], 'aerial_collected_data_stat')
#
#
# # OVERLAYED IMAGE PATHS
# conf['pathDict']['overlaid_batch_path'] = os.path.join(conf['pathDict']['data_model_path'],"overlaid_images","google",'batch_data')
# conf['pathDict']['overlaid_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "overlaid_images", "google", 'checkpoint')
# conf['pathDict']['overlaid_smry_path'] = os.path.join(conf['pathDict']['data_model_path'], "overlaid_images", "google", 'summary')
# conf['pathDict']['overlaid_pred_stats'] = os.path.join(conf['pathDict']['statistics_path'], 'prediction_stats',
#                                                        'overlaid_images')
#
# # AERIAL CROPPED IMAGE PATHS
# conf['pathDict']['aerial_cropped_image_path'] = os.path.join(conf['pathDict']['parent_path'],"input_images","aerial_cropped_images")
# conf['pathDict']['aerial_cropped_batch_path'] = os.path.join(conf['pathDict']['data_model_path'],"aerial_cropped_images","google",'batch_data')
# conf['pathDict']['aerial_cropped_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_cropped_images", "google", 'checkpoint')
# conf['pathDict']['aerial_cropped_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_cropped_images", "google", 'summary')
# conf['pathDict']['aerial_cropped_pred_stats'] = os.path.join(conf['pathDict']['statistics_path'], 'prediction_stats',
#                                                        'aerial_cropped_images')
#
# ##### Other new directory
# conf['pathDict']['assessor_code_house_path'] = os.path.join(conf['pathDict']['parent_path'], 'input_images','assessor_code_images','house')
# conf['pathDict']['assessor_code_land_path'] = os.path.join(conf['pathDict']['parent_path'], 'input_images', 'assessor_code_images','land')
#





##### Aerial Images from Bing
# conf['pathDict']['bing_aerial_image_path'] = os.path.join(conf['pathDict']['parent_path'], "input_images", "aerial_images", "bing")
# conf['pathDict']['bing_aerial_stats_path'] = os.path.join(conf['pathDict']['statistics_path'], "aerial_images", "bing")
# conf['pathDict']['bing_aerial_batch_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "bing", 'batch_data')
# conf['pathDict']['bing_aerial_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "bing", 'checkpoint')
# conf['pathDict']['bing_aerial_smry_path'] = os.path.join(conf['pathDict']['data_model_path'], "aerial_images", "bing", 'summary')

##### Streetside Images
# conf['pathDict']['bing_streetside_image_path'] = os.path.join(conf['pathDict']['parent_path'], "input_images", "streetside_images", "bing")
# conf['pathDict']['google_streetside_image_path'] = os.path.join(conf['pathDict']['parent_path'], "input_images", "streetside_images", "google")
# conf['pathDict']['streetside_rsized_path'] = os.path.join(conf['pathDict']['data_model_path'], "streetside_images")
# conf['pathDict']['streetside_batch_path'] = os.path.join(conf['pathDict']['data_model_path'], "streetside_images", 'batch_data')
# conf['pathDict']['streetside_ckpt_path'] = os.path.join(conf['pathDict']['data_model_path'], "streetside_images", 'checkpoint')
# conf['pathDict']['streetside_smry_path'] = os.path.join(conf['pathDict']['data_model_path'], "streetside_images", 'summary')



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

