
import os
import shutil

from src.config import pathDict


def clean(clean_dict, which_vendor, which_model):
    dir_to_empty = []
    for img_type, file_types in clean_dict.items():
        if img_type == 'assessor':
            for files in file_types.split(','):
                if files == "summary":
                    dir_to_empty += [os.path.join(pathDict['assessor_smry_path'], which_model)]
                if files == 'checkpoint':
                    dir_to_empty += [os.path.join(pathDict['assessor_ckpt_path'], which_model)]
                if files == 'batch':
                    dir_to_empty += [pathDict['assessor_batch_path']]
                if files == 'images':
                    dir_to_empty += [pathDict['assessor_image_path']]
        
        if img_type == 'aerial':
            for files in file_types.split(','):
                if files == "summary":
                    dir_to_empty += [os.path.join(pathDict['%s_aerial_smry_path'%(which_vendor)], which_model)]
                if files == 'checkpoint':
                    dir_to_empty += [os.path.join(pathDict['%s_aerial_ckpt_path'%(which_vendor)], which_model)]
                if files == 'batch':
                    dir_to_empty += [pathDict['%s_aerial_batch_path'%(which_vendor)]]
                if files == 'images':
                    dir_to_empty += [pathDict['%s_aerial_image_path'%(which_vendor)]]
                
        if img_type == 'overlayed':
            for files in file_types.split(','):
                if files == "summary":
                    dir_to_empty += [os.path.join(pathDict['%s_overlayed_smry_path'%(which_vendor)], which_model)]
                if files == 'checkpoint':
                    dir_to_empty += [os.path.join(pathDict['%s_overlayed_ckpt_path'%(which_vendor)], which_model)]
                if files == 'batch':
                    dir_to_empty += [pathDict['%s_overlayed_batch_path'%(which_vendor)]]
                if files == 'images':
                    dir_to_empty += [pathDict['%s_overlayed_image_path'%(which_vendor)]]
                
        if img_type == 'streetside':
            for files in file_types.split(','):
                if files == "summary":
                    dir_to_empty += [os.path.join(pathDict['%s_streetside_smry_path'%(which_vendor)], which_model)]
                if files == 'checkpoint':
                    dir_to_empty += [os.path.join(pathDict['%s_streetside_ckpt_path'%(which_vendor)], which_model)]
                if files == 'batch':
                    dir_to_empty += [pathDict['%s_streetside_batch_path'%(which_vendor)]]
                if files == 'images':
                    dir_to_empty += [pathDict['%s_streetside_image_path'%(which_vendor)]]
                
              
    
            
    should_clean = input('The below directories would be emptied... %s \n Are you sure you want to clean directories? ('
                         'yes/no) \n'%str(dir_to_empty))
    
    if should_clean.lower() == 'yes':
        for path in dir_to_empty:
            if os.path.exists(path):
                if len([i for i in os.listdir(path) if i != '.DS_Store']) > 0:
                    shutil.rmtree(path)
                    print ('Removed files from: %s'%(path))

#
# debugg = False
#
# if debugg:
#     clean(dict(assessor='summary,batch', aerial='summary,batch', overlayed='summary'),
#           which_vendor='google', which_model='resnet')
#
