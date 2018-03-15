import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from conv_net.utils import Score
from conv_net.vgg import vgg
from conv_net.resnet import resnet
from config import pathDict


from data_transformation.preprocessing import Preprocessing
from conv_net.train import PropertyClassification, load_batch_data

class Test(PropertyClassification):
    
    def _init__(self, params, which_net, image_type):
        PropertyClassification.__init__(self, params, which_net, image_type)
    
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph, is_training=False)
        
        batchY_1hot = self.to_one_hot(batchY)
        feed_dict = {
            self.computation_graph['inpX']: preprocessed_data,
            self.computation_graph['inpY']: batchY_1hot,
            self.computation_graph['is_training']: False
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
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            batchX, batchY = load_batch_data(
                    image_type=self.image_type,
                    image_shape=self.inp_img_shape,
                    which_data='cvalid')
            
            self.epoch = os.path.basename(checkpoint_path).split('.')[0].split('_')[2]
            self.batch_num = os.path.basename(checkpoint_path).split('.')[0].split('_')[2]
            
            self.restore_checkpoint(checkpoint_path, saver, sess)
            
            out_prob, tst_loss, tst_acc, ts_precsion_score, ts_recall_score = self.cvalid(batchX, batchY, sess)
        
        return batchY, out_prob, tst_loss, tst_acc, ts_precsion_score, ts_recall_score
    
    
    def run(self, dump_stats=False):
        logging.info('INITIATING TEST ........')
        tf.reset_default_graph()
        self.dump_stats = dump_stats
        checkpoint_paths = self.get_checkpoint_path(which_checkpoint='all')
        checkpoint_paths = np.sort(checkpoint_paths)
        
        stats_matrix = []
        colnames = []
        ts_loss_arr = []
        ts_acc_arr = []
        ts_precision_arr = []
        ts_recall_arr = []
        for path_num, chk_path in enumerate(checkpoint_paths):

            self.preprocess_graph = Preprocessing(inp_img_shape=self.inp_img_shape,
                                                  crop_shape=self.crop_shape,
                                                  out_img_shape=self.out_img_shape).preprocessImageGraph()
            if self.which_net == 'vgg':
                print ('Test Graph: VGG')
                self.computation_graph = vgg(training=False)
            elif self.which_net == 'resnet':
                print('Test Graphs: RESNET')
                self.computation_graph = resnet(img_shape=self.out_img_shape)
            else:
                raise ValueError('Provide a valid Net type options ={vgg, resnet}')

            # ########   RUN THE SESSION
            batchY, out_prob, tst_loss, tst_acc, tr_precision_score, ts_recall_score = self.test(chk_path)
            ts_loss_arr.append(tst_loss)
            ts_acc_arr.append(tst_acc)
            ts_precision_arr.append(tr_precision_score)
            ts_recall_arr.append(ts_recall_score)


            y_hat = np.argmax(out_prob, 1).reshape(-1 ,1)

            if self.dump_stats:

                if path_num == 0:
                    colnames.append('Label')
                    stats_matrix = batchY.reshape(-1 ,1)

                colnames.append( 'epoch_%s_batch_%s' %(str(self.epoch), str(self.batch_num)) + '_pred')
                colnames.append( 'epoch_%s_batch_%s' %(str(self.epoch), str(self.batch_num)) + '_prob')
                stats_matrix = np.column_stack((
                    stats_matrix, y_hat,
                    np.maximum(out_prob[: ,0], out_prob[: ,1]).reshape(-1 ,1))
                )

            tf.reset_default_graph()

            # if path_num == 5:
            #     break

        if self.dump_stats:
            stats_matrix = pd.DataFrame(stats_matrix, columns=colnames)
            stats_path = os.path.join(pathDict['%s_pred_stats' % str(self.image_type)], 'pred_stats.csv')
            stats_matrix.to_csv(stats_path, index=None)


        return ts_loss_arr, ts_acc_arr, ts_precision_arr, ts_recall_arr

