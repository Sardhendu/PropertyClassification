import os
import numpy as np
import pandas as pd
from config import pathDict


import os
import numpy as np
import pandas as pd
from config import pathDict


class GetMislabels():
    def __init__(self, which_data, inp_img_path=None, inp_stats_path=None):
        self.which_data = which_data
        
        
        if inp_img_path:
            self.input_img_path = inp_img_path
        else:
            self.input_img_path = pathDict['image_path']

        if inp_stats_path:
            stats_path = inp_stats_path
        else:
            stats_path = pathDict['statistics_path']
            
        if which_data == 'cvalid':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'cvalid_pred_outcomes.csv')
        elif which_data == 'test':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'test_pred_outcomes.csv')
        elif which_data == 'new':
            self.pred_stats_path = os.path.join(inp_stats_path, 'prediction_stats',
                                                'test_pred_outcomes.csv')
            
        self.meta_stats_path = os.path.join(inp_stats_path, 'prediction_stats', 'tr_cv_ts_pins_info.csv')
#         else:
#             raise ValueError('Provide proper data name: Option: cvalid, test')
        self.pred_stats = pd.read_csv(self.pred_stats_path)
        self.meta_stats = pd.read_csv(self.meta_stats_path, index_col=None)
        print (self.pred_stats.shape, self.meta_stats.shape)
        
        
    def concat_meta_n_pred_stats(self, checkpoint_name_arr, which_data):
        pred_meta_mrgd = self.pred_stats.merge(self.meta_stats, left_on=['rownum', "dataset_type"], right_on=['rownum', "dataset_type"], how='outer')
        if which_data != 'new':
            pred_meta_mrgd = pred_meta_mrgd[pred_meta_mrgd["dataset_type"] == which_data]
      
        column_names = ["property_pins", "property_type", "bbox_cropped", "true_label"]
        pred_prob_data = []
        for num, checkpoint_name in enumerate(checkpoint_name_arr):
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
                        pred_meta_mrgd[pred_meta_mrgd["checkpoint"] == checkpoint_name][["pred_label", "pred_prob"]]).reshape(
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
        print (dynamic_query)
        mislabeled_data = concat_meta_pred_data.query(dynamic_query)
        land_mis_pins_path, house_mis_pins_path = self.get_pin_path(dataIN=mislabeled_data)
        land_title_arr, house_title_arr = self.get_title_array(dataIN=mislabeled_data)
        return mislabeled_data, land_mis_pins_path, house_mis_pins_path, land_title_arr, house_title_arr
    
    
def get_misclassified_images(which_data, input_img_path, input_stats_path, checkpoint_arr):
    obj_ms = GetMislabels(which_data=which_data, inp_img_path=input_img_path, inp_stats_path=input_stats_path)
    
    concat_meta_pred_data = obj_ms.concat_meta_n_pred_stats(checkpoint_arr, which_data)
    
    dynamic_query = obj_ms.dynamic_rule_based_mislabel_correction(min_pred_prob=[0]*len(checkpoint_arr), 
                                                           checkpoint_arr=checkpoint_arr, bbox_cropped=False)
    
    mislabeled_data = concat_meta_pred_data.query(dynamic_query)
    land_mis_pins_path, house_mis_pins_path = obj_ms.get_pin_path(dataIN=mislabeled_data)
    land_title_arr, house_title_arr = obj_ms.get_title_array(dataIN=mislabeled_data)
    return concat_meta_pred_data, mislabeled_data, land_mis_pins_path, house_mis_pins_path, land_title_arr, house_title_arr
    