
import os
import numpy as np
import pandas as pd
from src.config import get_config
from src.conv_net.train import Train
from src.conv_net.test import Test
from src.prediction_n_mislabels import get_predictions_using_multiple_checkpoints
# from src.plot import Plot

# def save_plot_for_train(l_rate_arr, tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
#                         cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr, conf):
#
#     oj = Plot(rows=2, columns=3, fig_size=(30, 10))
#
#     l_rate_df = pd.DataFrame(l_rate_arr, columns=['learning_rate'])
#     oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line',
#                  params={'title': 'Learning Rate Decay'})
#
#     tr_loss_df = pd.DataFrame(tr_loss_arr, columns=['training_loss'])
#     oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line',
#                  params={'title': 'Training Loss'})
#     tr_data = pd.DataFrame(np.column_stack((tr_acc_arr, tr_precision_arr, tr_recall_arr)),
#                            columns=['accuracy', 'precision', 'recall'])
#     oj.vizualize(data=tr_data, colX=None, colY=None, label_col=None, viz_type='line',
#                  params={'title': 'Training Metrics'})
#
#     cv_loss_df = pd.DataFrame(cv_loss_arr, columns=['crossvalidation_loss'])
#     oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line',
#                  params={'title': 'Crossvalidation Loss'})
#     cv_data = pd.DataFrame(np.column_stack((cv_acc_arr, cv_precision_arr, cv_recall_arr)),
#                            columns=['accuracy', 'precision', 'recall'])
#     oj.vizualize(data=cv_data, colX=None, colY=None, label_col=None, viz_type='line',
#                  params={'title': 'Cross-Validation Metrics'})
#
#     save_path = os.path.join(conf['pathDict']['statistics_path'], 'viz_stats')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     oj.save(os.path.join(save_path,'train.png'))


def train(**kwargs):
    inp_params = kwargs['params']
    print(inp_params)
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')
    
    if 'use_checkpoint' not in inp_params.keys():
        raise ValueError('You should provide whether to train using previous checkpoints')
    
    if 'save_checkpoint' not in inp_params.keys():
        raise ValueError('You should provide, whether to save checkpoint or not')
    
    if 'write_tensorboard_summary' not in inp_params.keys():
        raise ValueError('You should provide, whether to write tensorboard summary or not')
    
    if 'which_net' not in inp_params.keys():
        raise ValueError('You should provide, which net to use')
    

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])

    train_batches = [file
                     for file in os.listdir(conf['pathDict']['batch_path'])
                     if file.split('.')[1] == 'h5']
    max_batches = len(train_batches) - 2

    tr_obj = Train(conf,
                   dict(pprocessor_inp_img_shape=[224, 224, 3],
                        pprocessor_inp_crop_shape=[],
                        model_inp_img_shape=[224, 224, 3],
                        use_checkpoint=inp_params['use_checkpoint'],
                        save_checkpoint=inp_params['save_checkpoint'],
                        write_tensorboard_summary=inp_params['write_tensorboard_summary']
                        ),
                   device_type='gpu',
                   which_net=inp_params['which_net'])
    # # (tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
    # #      cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr,
    # #      l_rate_arr) = \
    
    
    (tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
    cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr,
    l_rate_arr) = tr_obj.run(num_epochs=1, num_batches=max_batches, cv_num_batches=1, get_stats_at=2)

    print(
            'Mean Values: train_loss = %s, train_acc = %s, train_precision = %s, train_recall = %s, cv_loss = %s, '
            'cv_acc = %s, cv_precision = %s, cv_recall = %s' % (
                np.mean(tr_loss_arr), np.mean(tr_acc_arr), np.mean(tr_precision_arr), np.mean(tr_recall_arr),
                np.mean(cv_loss_arr), np.mean(cv_acc_arr), np.mean(cv_precision_arr), np.mean(cv_recall_arr)))
    
    return 'COMPLETE'

def cvalid(**kwargs):
    inp_params = kwargs['params']

    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')
    
    if 'which_net' not in inp_params.keys():
        raise ValueError('You should provide, which net to use')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])

    tsoj = Test(conf,
                params=dict(pprocessor_inp_img_shape=[224,224,3],
                            pprocessor_inp_crop_shape=[],
                            model_inp_img_shape=[224, 224, 3]),
                device_type = 'gpu',
                which_net=inp_params['which_net'])

    tsoj.run(use_checkpoint_for_run=inp_params['which_run'],
             use_checkpoint_for_imageType=inp_params['img_type'],
             optional_batch_name='cvalid',
             which_checkpoint='all',
             which_data='cvalid',
             dump_stats=True)

    return 'COMPLETE'

def test(**kwargs):
    inp_params = kwargs['params']

    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')
    
    if 'which_net' not in inp_params.keys():
        raise ValueError('You should provide, which net to use')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])

    tsoj = Test(
            conf,
            params=dict(pprocessor_inp_img_shape=[224, 224, 3],
                        pprocessor_inp_crop_shape=[],
                        model_inp_img_shape=[224, 224, 3]),
            device_type='gpu',
            which_net=inp_params['which_net'])
    
    tsoj.run(use_checkpoint_for_run=inp_params['which_run'],
             use_checkpoint_for_imageType=inp_params['img_type'],
             optional_batch_name='test',
             which_checkpoint='all',
             which_data='test',
             dump_stats=True)
    
    return 'COMPLETE'
#
#
def test_new(**kwargs):
    inp_params = kwargs['params']

    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    if 'use_checkpoint_for_run' not in inp_params.keys():
        raise ValueError('Provide the RUN name whose model you would wanna use')

    if 'use_checkpoint_for_imageType' not in inp_params.keys():
        raise ValueError('Provide the image_type name whose model you would wanna use')
    
    if 'which_net' not in inp_params.keys():
        raise ValueError('You should provide, which net to use')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])

    tsoj = Test(
            conf,
            params=dict(pprocessor_inp_img_shape=[224, 224, 3],
                        pprocessor_inp_crop_shape=[],
                        model_inp_img_shape=[224, 224, 3]),
            device_type='gpu',
            which_net=inp_params['which_net'])
    
    tsoj.run(
            use_checkpoint_for_run=inp_params['use_checkpoint_for_run'],
            use_checkpoint_for_imageType=inp_params['use_checkpoint_for_imageType'],
            optional_batch_name=None,
            which_checkpoint='all',
            which_data='test_new',
            dump_stats=True)

    return 'COMPLETE'


def predictions(**kwargs):
    inp_params = kwargs['params']
    
    if 'which_run' not in inp_params.keys():
        raise ValueError('You should provide the name of RUN')

    if 'img_type' not in inp_params.keys():
        raise ValueError('You should provide a valid image type to create batches')

    if 'use_checkpoint_for_run' not in inp_params.keys():
        raise ValueError('Provide the RUN name whose model you would wanna use')
    
    if 'use_checkpoint_for_imageType' not in inp_params.keys():
        raise ValueError('Provide the image_type name whose model you would wanna use')
    
    if 'which_net' not in inp_params.keys():
        raise ValueError('You should provide, which net to use')
    
    if 'use_checkpoint_for_prediction' not in inp_params.keys():
        raise ValueError('You should provide, whether you want prediction based on "all" checkpoints or "max"')

    conf = get_config(which_run=inp_params['which_run'], img_type=inp_params['img_type'])
    
    checkpoint_path = os.path.join(conf['pathDict']['parent_checkpoint_path'], inp_params['use_checkpoint_for_run'],
                                   inp_params['use_checkpoint_for_imageType'], inp_params['which_net'])
    
    get_predictions_using_multiple_checkpoints(conf, which_data='test_new', checkpoint_path=checkpoint_path,
                                               use_checkpoint_for_prediction='all')
    return 'COMPLETE'
    
