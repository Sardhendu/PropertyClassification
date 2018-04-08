import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from conv_net.utils import Score
from conv_net.resnet import resnet
from conv_net.convnet import conv_net
from config import pathDict


from data_transformation.preprocessing import Preprocessing
from conv_net.train import PropertyClassification, load_batch_data

class Test(PropertyClassification):
    
    def _init__(self, params, device_type, which_net):
        PropertyClassification.__init__(self, params, device_type, which_net)

    
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph)
        
        batchY_1hot = self.to_one_hot(batchY)
        feed_dict = {
            self.computation_graph['inpX']: preprocessed_data,
            self.computation_graph['inpY']: batchY_1hot,
        }
        
        out_prob, ts_acc, ts_loss = sess.run([self.computation_graph['outProbs'],
                                              self.computation_graph['accuracy'],
                                              self.computation_graph['loss']], feed_dict=feed_dict)
        
        ts_pred = sess.run(tf.argmax(out_prob, 1))
        ts_recall_score = Score.recall(batchY, ts_pred, reverse=True)
        ts_precsion_score = Score.precision(batchY, ts_pred, reverse=True)
        # We use reverse as true becasue, the business case is to identify more lands than houses and lands are
        # labeled as 0. When we do reverse 0 turns 1 and we get a recall on lands

        logging.info(
                "VALIDATION METRICs : Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s, Precision: %s, "
                "Recall: %s",
                str(1),
                str(self.epoch),
                str(self.batch_num),
                str("{:.6f}".format(ts_loss)),
                str("{:.5f}".format(ts_acc)),
                str("{:.5f}".format(ts_precsion_score)),
                str("{:.5f}".format(ts_recall_score)))
        
        return out_prob, ts_loss, ts_acc, ts_precsion_score, ts_recall_score
    
    def test(self,  checkpoint_path):
        saver = tf.train.Saver()
        self.max_checkpoint_num = 0
        
        config_ = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config_) as sess:
            sess.run(tf.global_variables_initializer())
            self.epoch = os.path.basename(checkpoint_path).split('.')[0].split('_')[2]
            self.batch_num = os.path.basename(checkpoint_path).split('.')[0].split('_')[4]
            
            self.restore_checkpoint(checkpoint_path, saver, sess)

            if self.batch_name:
                batch_name_arr = [self.batch_name]
            else:
                batch_name_arr = [dirs.split('.')[0] for dirs in os.listdir(pathDict['batch_path']) if
                                  dirs!='.DS_Store']
                
            print('Batch path %s, batch_names: %s'%(str(pathDict['batch_path']), str(batch_name_arr)))

            tst_metric_stack = np.ndarray((len(batch_name_arr), 5), dtype=object)
            true_pred_prob_stack = []
            
            for batch_num, batch_name in enumerate(batch_name_arr):
                batchX, batchY = load_batch_data(which_data=batch_name)
                out_prob, tst_loss, tst_acc, ts_precsion_score, ts_recall_score = self.cvalid(batchX, batchY, sess)

                tst_metric_stack[batch_num] = [batch_name, round(tst_loss, 3), round(tst_acc, 3),
                                               round(ts_precsion_score, 3), round(ts_recall_score, 3)]

                y_hat = np.argmax(out_prob, 1).reshape(-1, 1)
                out_pred_prob = np.maximum(out_prob[:, 0], out_prob[:, 1]).round(decimals=3).reshape(-1, 1)
                
                if batch_num == 0:
                    true_pred_prob_stack = np.column_stack((np.tile(batch_name, len(out_prob)), batchY.reshape(-1 ,1), y_hat,out_pred_prob))
                else:
                    true_pred_prob_stack = np.vstack((
                        true_pred_prob_stack,
                        np.column_stack((np.tile(batch_name, len(out_prob)), batchY.reshape(-1 ,1), y_hat, out_pred_prob))))

            return true_pred_prob_stack, tst_metric_stack
            #
    
    def run(self, use_checkpoint_for_run, use_checkpoint_for_imageType, optional_batch_name=None,
            which_checkpoint='max', dump_stats=False):
        logging.info('INITIATING TEST ........')
        self.stats_path = os.path.join(pathDict['statistics_path'], 'prediction_stats')
        # Override the checkpoint path.
        self.ckpt_path = os.path.join(pathDict['parent_checkpoint_path'], use_checkpoint_for_run,
                                      use_checkpoint_for_imageType, self.which_net)
        
        tf.reset_default_graph()
        self.dump_stats = dump_stats
        self.batch_name = optional_batch_name

        checkpoint_paths = self.get_checkpoint_path(which_checkpoint)
        
        if which_checkpoint=='all':
            dir_name = os.path.dirname(list(checkpoint_paths)[0])
            cmn_filename = os.path.basename(checkpoint_paths[0]).split('_')
            cmn_filename[2] = '%s'
            cmn_filename[4] = '%s'
            cmn_filename = ('_').join(cmn_filename)
    
            nums = np.array([[os.path.basename(chk).split('_')[2], os.path.basename(chk).split('_')[4]] for chk in checkpoint_paths], dtype=int)
            nums_sort = nums[np.lexsort((nums[:,1], nums[:,0]))]
    
            checkpoint_paths = [os.path.join(dir_name, cmn_filename%(str(i), str(j))) for i, j in nums_sort]
        else:
            checkpoint_paths = [checkpoint_paths]

       

        fnl_true_pred_prob_stack = []
        fnl_tst_metric_stack = []
        colnames1 = ['checkpoint', 'test_batch', 'true_label', 'pred_label', 'pred_prob']
        colnames2 = ['checkpoint', 'test_batch', 'test_loss', 'test_acc', 'test_precsion', 'test_recall']

        for path_num, chk_path in enumerate(checkpoint_paths):
            ########## Create Grephs
            self.preprocess_graph = Preprocessing(
                    pprocessor_inp_img_shape=self.pprocessor_inp_img_shape,
                    pprocessor_inp_crop_shape=self.pprocessor_inp_crop_shape,
                    model_inp_img_shape=self.model_inp_img_shape).preprocessImageGraph(is_training=False)

            if self.which_net == 'resnet':
                print('Test Graphs: RESNET')
                self.computation_graph = resnet(img_shape=self.model_inp_img_shape, device_type=self.device_type,
                                                use_dropout=False)
            elif self.which_net == 'convnet':
                print('Test Graphs: CONVNET')
                self.computation_graph = conv_net(img_shape=self.model_inp_img_shape, device_type=self.device_type)
            else:
                raise ValueError('Provide a valid Net type options ={vgg, resnet}')
            
            # ########   RUN THE SESSION
            true_pred_prob_stack, tst_metric_stack = self.test(chk_path)
            
            chkpnt_name = 'epoch_%s_batch_%s' %(str(self.epoch), str(self.batch_num))
            
            if path_num == 0:
                fnl_true_pred_prob_stack = np.column_stack((np.tile(chkpnt_name, len(true_pred_prob_stack)), true_pred_prob_stack))
                fnl_tst_metric_stack = np.column_stack((np.tile(chkpnt_name, len(tst_metric_stack)), tst_metric_stack))
            else:
                fnl_true_pred_prob_stack = np.vstack((
                    fnl_true_pred_prob_stack, np.column_stack((np.tile(chkpnt_name, len(true_pred_prob_stack)),true_pred_prob_stack))
                ))
                fnl_tst_metric_stack = np.vstack((
                    fnl_tst_metric_stack,  np.column_stack((np.tile(chkpnt_name, len(tst_metric_stack)), tst_metric_stack))
                ))
            
            tf.reset_default_graph()

        if self.dump_stats:
            fnl_true_pred_prob_df = pd.DataFrame(fnl_true_pred_prob_stack, columns=colnames1)
            pred_path = os.path.join(self.stats_path, 'pred_outcomes.csv')
            fnl_true_pred_prob_df.to_csv(pred_path, index=None)

            fnl_tst_stats_df = pd.DataFrame(fnl_tst_metric_stack, columns=colnames2)
            metric_path = os.path.join(self.stats_path, 'pred_metrics.csv')
            fnl_tst_stats_df.to_csv(metric_path, index=None)
        #
        return fnl_tst_metric_stack

# params, device_type, which_net, use_checkpoint_for_run, use_checkpoint_for_imageType
#
# which_data = 'cvalid'
# tsoj = Test(params=dict(pprocessor_inp_img_shape=[224,224,3],
#                         pprocessor_inp_crop_shape=[],
#                         model_inp_img_shape=[224, 224, 3]),
#             device_type = 'gpu',
#             which_net='resnet')
# fnl_tst_metric_stack = tsoj.run(
#         use_checkpoint_for_run='sam_new',
#         use_checkpoint_for_imageType='aerial_cropped',
#         optional_batch_name=None,
#         which_checkpoint='max',
#         dump_stats=True)
#
# print (fnl_tst_metric_stack)
