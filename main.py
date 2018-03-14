from __future__ import division, print_function, absolute_import
import logging

import os
import numpy as np
from config import pathDict
from conv_net.run import Train, Test
from data_transformation.data_prep import get_valid_land_house_ids, dumpStratifiedBatches_balanced_class




images_per_label = None # normally 5000 each label is good
assessor_img_type = 'assessor'
aerial_img_type = 'google_aerial' # 'bing_aerial'
overlayed_img_type = 'google_overlayed'
streetside_img_type = None





image_type = overlayed_img_type#aerial_img_type#assessor_img_type

if image_type == 'assessor':
    inp_image_shape = [260, 260, 3]
elif image_type == 'google_aerial':
    inp_image_shape = [400, 400, 3]
elif image_type == 'google_overlayed':
    inp_image_shape = [400, 400, 3]
elif image_type == 'google_streetside':
    inp_image_shape = [260, 260, 3]
else:
    raise ValueError('Not a valid image type provided')
    
batch_prepare = True
train = False
test = False
which_net = 'resnet'
max_batches = 5



if batch_prepare:
    cmn_land_pins, cmn_house_pins = get_valid_land_house_ids(
            aerial_img_type=aerial_img_type,
            streetside_img_type=streetside_img_type,
            overlayed_img_type=overlayed_img_type,
            images_per_label=images_per_label)
    print (len(cmn_land_pins), len(cmn_house_pins))

    cv_batch_size = (len(cmn_land_pins) + len(cmn_house_pins)) // 10
    
    dumpStratifiedBatches_balanced_class(cmn_land_pins, cmn_house_pins, img_resize_shape=inp_image_shape,
                                         image_type=image_type, cv_batch_size=cv_batch_size, tr_batch_size=128,
                                         shuffle_seed=873, get_stats=True, max_batches=max_batches)


if train:
    Train(dict(inp_img_shape=[400,400,3],
               crop_shape=[160,160,3],
               out_img_shape=[224, 224, 3],
               use_checkpoint=True,
               save_checkpoint=True,
               write_tensorboard_summary=True
               ),
          which_net=which_net,  # vgg
          image_type=image_type).run(num_epochs=3,
                                     num_batches=160)
if test:
    Test(params=dict(inp_img_shape=[400,400,3],
                     crop_shape=[160, 160, 3],
                     out_img_shape=[224, 224, 3]),
         which_net=which_net,
         image_type=image_type).run(dump_stats=True)