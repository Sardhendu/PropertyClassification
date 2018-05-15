import os
import numpy as np
import pandas as pd
from src.conv_net.utils import Score

class GetPrediction_N_Mislabels():
    def __init__(self, conf, which_data, inp_img_path=None, inp_stats_path=None):
        self.which_data = which_data
        
        if inp_img_path:
            self.input_img_path = inp_img_path
        else:
            self.input_img_path = conf['pathDict']['image_path']
        
        if not inp_stats_path:
            inp_stats_path = conf['pathDict']['statistics_path']
        
        if which_data == 'cvalid':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'cvalid_pred_outcomes.csv')
            self.meta_stats_path = os.path.join(inp_stats_path, 'prediction_stats', 'tr_cv_ts_pins_info.csv')
        elif which_data == 'test':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'test_pred_outcomes.csv')
            self.meta_stats_path = os.path.join(inp_stats_path, 'prediction_stats', 'tr_cv_ts_pins_info.csv')
        elif which_data == 'test_new':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'test_new_pred_outcomes.csv')
            self.meta_stats_path = os.path.join(inp_stats_path, 'prediction_stats', 'test_new_pins_info.csv')

        self.pred_stats = pd.read_csv(self.pred_stats_path)
        self.meta_stats = pd.read_csv(self.meta_stats_path, index_col=None)
        print('pred_stats.shape = %s, meta_stats.shape = %s'%(str(self.pred_stats.shape), str(self.meta_stats.shape)))
        
    def reform_labels_matrix_on_threshold(self, threshold):
        # self.pred_stats = self.pred_stats[self.pred_stats['checkpoint'] == 'epoch_15_batch_70']
        self.pred_stats = self.pred_stats.reset_index()
        # print (self.pred_stats.shape)
        # print(self.pred_stats[self.pred_stats['pred_label'] == 0].shape)
        # print(self.pred_stats[self.pred_stats['pred_label'] == 1].shape)

        # The below says:
        # 1. when the prediciton is land and its (1-prediction probility) is <= 0.68 then its land else its house
        # 2. when the prediction is house and prediction probility is < 0.68 then it should be a land
        # All in one if prediction probability is <= threshold then land else house
        land_at_thresh = self.pred_stats[(((self.pred_stats['pred_label'] == 0) &
                                          (1-self.pred_stats['pred_prob'] <= threshold)) |
                                          (((self.pred_stats['pred_label'] == 1) &
                                          (self.pred_stats['pred_prob'] <= threshold))))].index

        
        self.pred_stats.loc[(self.pred_stats['index'].isin(land_at_thresh)), 'pred_label'] = 0
        self.pred_stats.loc[(~self.pred_stats['index'].isin(land_at_thresh)), 'pred_label'] = 1
        self.pred_stats = self.pred_stats.drop('index', axis=1)
        #
        # print(self.pred_stats.shape)
        # print(self.pred_stats[self.pred_stats['pred_label'] == 0].shape)
        # print(self.pred_stats[self.pred_stats['pred_label'] == 1].shape)
        
    
    def concat_meta_n_pred_stats(self, checkpoint_name_arr, which_data):
        pred_meta_mrgd = self.pred_stats.merge(self.meta_stats, left_on=['rownum', "dataset_type"], right_on=['rownum', "dataset_type"], how='outer')
        
        if which_data != 'test_new':
            pred_meta_mrgd = pred_meta_mrgd[pred_meta_mrgd["dataset_type"] == which_data]
        
        column_names = ["property_pins", "property_type", "bbox_cropped", "true_label"]
        
        pred_prob_data = []
        self.checkpoint_label_col_arr = []
        for num, checkpoint_name in enumerate(checkpoint_name_arr):
            self.checkpoint_label_col_arr += ['%s_pred_label' % (checkpoint_name)]
            column_names += ['%s_pred_label' % (checkpoint_name), '%s_pred_prob' % (checkpoint_name)]
            
            if num == 0:
                pred_prob_data = np.array(
                        pred_meta_mrgd[pred_meta_mrgd["checkpoint"] == checkpoint_name][
                            ["property_pins", "property_type", "bbox_cropped", "true_label", "pred_label", "pred_prob"]
                        ]
                ).reshape(-1, 6)
            else:
                pred_prob_data = np.column_stack((
                    pred_prob_data,
                    np.array(
                            pred_meta_mrgd[pred_meta_mrgd["checkpoint"] == checkpoint_name][
                                ["pred_label", "pred_prob"]]).reshape(
                            -1, 2)
                ))
        pred_prob_data = pd.DataFrame(pred_prob_data, columns=column_names)
        float_columns = [col for col in column_names
                         if col not in ['property_pins', 'property_type', 'bbox_cropped', 'true_label']]
        
        pred_prob_data['bbox_cropped'] = pred_prob_data['bbox_cropped'].astype('int')
        pred_prob_data['true_label'] = pred_prob_data['true_label'].astype('int')
        for cols in float_columns:
            pred_prob_data[cols] = pred_prob_data[cols].astype('float')
        
        if pred_prob_data.isnull().values.any():
            raise ValueError('NaN Found! Seems the concat operation did merge properly (Check dataframe shapes)')
        return pred_prob_data
    
    def dynamic_rule_based_mislabel_correction(self, min_pred_prob=[], checkpoint_arr=[], bbox_cropped=True):
        '''
            min_pred_prob: The minimum prediction values for each checkpoint to qualify as a mislabeled data.
            checkpoint_arr: checkpoints to use while dynamically finding classification error due to mislabeled data.

        '''
        if len(min_pred_prob) != len(checkpoint_arr):
            raise ValueError('Provide min_pred_prob for each Checkpoint')
        
        dynamic_query = ''
        for num, (prob, checkpoint_name) in enumerate(zip(min_pred_prob, checkpoint_arr)):
            q = ' %s_pred_prob >= %s & true_label-%s_pred_label!=0 &' % (
                checkpoint_name, prob, checkpoint_name)
            dynamic_query += q
        dynamic_query = dynamic_query.strip('&').strip(' ')
        
        if bbox_cropped:
            q = "((property_type=='land' & bbox_cropped==1) | (property_type=='house' & bbox_cropped==0))"
            dynamic_query += " & " + q
        return dynamic_query
    
    def get_pin_path(self, dataIN):
        land_data = np.array(dataIN[dataIN['property_type'] == 'land']['property_pins'])
        house_data = np.array(dataIN[dataIN['property_type'] == 'house']['property_pins'])
        
        land_mis_pins_path = [os.path.join(self.input_img_path, 'land', pins + '.jpg') for pins in land_data]
        house_mis_pins_path = [os.path.join(self.input_img_path, 'house', pins + '.jpg') for pins in house_data]
        
        print(len(land_mis_pins_path), len(house_mis_pins_path))
        return land_mis_pins_path, house_mis_pins_path
    
    def get_title_array(self, dataIN):
        land_data = dataIN[dataIN['property_type'] == 'land'].reset_index().drop('index', axis=1)
        house_data = dataIN[dataIN['property_type'] == 'house'].reset_index().drop('index', axis=1)
        
        land_data['rownum'] = pd.Series(range(0, len(land_data)))
        house_data['rownum'] = pd.Series(range(0, len(house_data)))
        
        land_title_arr = np.array(land_data["rownum"].astype(str) + '--' +
                                  land_data["property_pins"].astype(str))
        
        house_title_arr = np.array(house_data["rownum"].astype(str) + '--' +
                                   house_data["property_pins"].astype(str))
        
        print(len(land_title_arr), len(house_title_arr))
        return land_title_arr, house_title_arr
    
    def main(self, checkpoint_min_prob_dict, bbox_cropped=True):
        min_pred_prob = list(checkpoint_min_prob_dict.values())
        checkpoint_name_arr = list(checkpoint_min_prob_dict.keys())
        
        concat_meta_pred_data = self.concat_meta_n_pred_stats(checkpoint_name_arr=checkpoint_name_arr,
                                                              which_data=self.which_data)
        dynamic_query = self.dynamic_rule_based_mislabel_correction(min_pred_prob=min_pred_prob,
                                                                    checkpoint_arr=checkpoint_name_arr,
                                                                    bbox_cropped=bbox_cropped)
        print(dynamic_query)
        mislabeled_data = concat_meta_pred_data.query(dynamic_query)
        land_mis_pins_path, house_mis_pins_path = self.get_pin_path(dataIN=mislabeled_data)
        land_title_arr, house_title_arr = self.get_title_array(dataIN=mislabeled_data)
        return mislabeled_data, land_mis_pins_path, house_mis_pins_path, land_title_arr, house_title_arr


def get_misclassified_images(which_data, input_img_path, input_stats_path, checkpoint_arr):
    obj_ms = GetPrediction_N_Mislabels(which_data=which_data, inp_img_path=input_img_path, inp_stats_path=input_stats_path)
    
    concat_meta_pred_data = obj_ms.concat_meta_n_pred_stats(checkpoint_arr, which_data)
    
    dynamic_query = obj_ms.dynamic_rule_based_mislabel_correction(min_pred_prob=[0] * len(checkpoint_arr), checkpoint_arr=checkpoint_arr, bbox_cropped=False)
    
    mislabeled_data = concat_meta_pred_data.query(dynamic_query)
    land_mis_pins_path, house_mis_pins_path = obj_ms.get_pin_path(dataIN=mislabeled_data)
    land_title_arr, house_title_arr = obj_ms.get_title_array(dataIN=mislabeled_data)
    return concat_meta_pred_data, mislabeled_data, land_mis_pins_path, house_mis_pins_path, land_title_arr, house_title_arr


def get_predictions_using_multiple_checkpoints(conf, which_data, checkpoint_path, use_checkpoint_for_prediction, threshold):
    ## GET CHECKPOINTS
    checkpoints = [str(filename.split('.')[0]) for filename in os.listdir(checkpoint_path)
                   if filename.endswith('meta')]
    print('Checkpoint LISTS .. %s', str(checkpoints))
    checkpoint_arr = []
    if len(checkpoints) > 0:
        if use_checkpoint_for_prediction == 'all':
            for i in checkpoints:
                checkpoint_arr += ['_'.join([k for j,k in enumerate(i.split('_')) if j >0])]
        elif use_checkpoint_for_prediction == 'max':
            epoch_batch = np.array([[int(ckpt.split('_')[2]), int(ckpt.split('_')[4])]
                                    for ckpt in checkpoints], dtype=int)
            max_epoch_ = epoch_batch[np.where(epoch_batch[:, 0] == max(epoch_batch[:, 0]))[0]]  # .reshape(1,-1)
            
            max_epoch, max_batch = np.squeeze(
                    max_epoch_[np.where(max_epoch_[:, 1] == max(max_epoch_[:, 1]))[0]])
            checkpoint_arr = ['epoch_%s_batch_%s' % (str(max_epoch),max_batch)]
            print('Checkpoint latest at: ', str(checkpoint_path))
        else:
            raise ValueError('Provide valid checkpoint type')
    else:
        print ('No Checkpoint found, make sure the path %s is correct'%(str(checkpoint_path)))
    
    ## GET THE COMBINED DATA
    obj_ms = GetPrediction_N_Mislabels(conf, which_data=which_data, inp_img_path=None, inp_stats_path=None)
    obj_ms.reform_labels_matrix_on_threshold(threshold = threshold)
    meta_join_pred = obj_ms.concat_meta_n_pred_stats(checkpoint_arr, which_data)
    
    ## GET THE PREDICTION SUPPORTED BY MOST CHECKPOINTS
    if (len(obj_ms.checkpoint_label_col_arr) % 2) == 0 and len(obj_ms.checkpoint_label_col_arr) > 1:
        obj_ms.checkpoint_label_col_arr.pop(0)

    pred_checkpoints = meta_join_pred[obj_ms.checkpoint_label_col_arr]
    
    labels = meta_join_pred['true_label']
    predictions = np.array(pred_checkpoints.T.mode()).reshape(-1,1)
    
    print ('The accuracy using predictions from multiple checkpoint is: ', Score.accuracy(labels, predictions))
    
  
    fnl_pred_out = pd.DataFrame(np.column_stack((np.array(meta_join_pred['property_pins']).reshape(-1,1),
                                                 predictions)), columns=['property_pins', 'label'])
    fnl_pred_out.to_csv(os.path.join(conf['pathDict']['statistics_path'], 'prediction_stats', 'PREDICTIONS.csv'))


# from src.config import get_config
#
# conf = get_config(which_run='PreTestRun', img_type='aerial_cropped')
# print (conf)