import logging
import os

import numpy as np
import tensorflow as tf

from conv_net.utils import Score
from conv_net.vgg import vgg
from conv_net.resnet import resnet
from config import pathDict, myNet
from conv_net.ops import summary_builder
from data_transformation.data_io import getH5File
from data_transformation.preprocessing import Preprocessing
from data_transformation.data_prep import unison_shuffled_copies


logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def load_batch_data(image_type, image_shape, which_data='cvalid'):
    if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside','google_overlayed']:
        raise ValueError('Can not identify the image type %s, Please provide a valid one' % (str(image_type)))
    
    data_path = pathDict['%s_batch_path' % (str(image_type))]
    batch_file_name = '%s' % (which_data)
   
    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY = getH5File(data_path, batch_file_name)
    
    return dataX, dataY


class PropertyClassification(object):
    def __init__(self, params, which_net, image_type):
        params_keys = list(params.keys())
        self.which_net = which_net
        if 'inp_img_shape' in  params_keys:
            self.inp_img_shape = params['inp_img_shape']

        if 'crop_shape' in params_keys:
            self.crop_shape = params['crop_shape']

        if 'out_img_shape' in params_keys:
            self.out_img_shape = params['out_img_shape']


        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']
        
        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']
        
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
        
        if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside', 'google_overlayed']:
            raise ValueError('Can not identify the image type %s, Please provide a valid one'%(str(image_type)))
        
        self.ckpt_path = os.path.join(pathDict['%s_ckpt_path'%(str(image_type))], self.which_net )
        self.smry_path = os.path.join(pathDict['%s_smry_path'%(str(image_type))], self.which_net )
        
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            
        if not os.path.exists(self.smry_path):
            os.makedirs(self.smry_path)
        
        self.image_type = image_type
        print('Dumping Checkpoints to %s', self.ckpt_path)
        print('Dumping Tensorboard Summary to %s', self.smry_path)
        

    def reshape(self, x, y):
        return (x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]),
                y.reshape(y.shape[0] * y.shape[1]))

    def to_one_hot(self, y):
        y = np.array(y, dtype=int)
        n_values = int(np.max(y)) + 1
        y = np.eye(n_values)[y]
        return y

    def accuracy(self, y, y_hat):
        # Both the predictions and the labels should be in One-Hot vector format.
        return (100.0 * np.sum(np.argmax(y_hat, 1) == np.argmax(y, 1)) / y_hat.shape[0])
    

    def restore_checkpoint(self, checkpoint_path, saver, sess):
        saver.restore(sess, checkpoint_path)
   

    def get_checkpoint_path(self, which_checkpoint):
        checkpoints = [str(filename.split('.')[0]) for filename in os.listdir(self.ckpt_path)
                       if filename.endswith('meta')]
        logging.info('Checkpoint LISTS .. %s', str(checkpoints))
        if len(checkpoints) > 0:
            if which_checkpoint == 'all':
                checkpoint_path = [os.path.join(self.ckpt_path, pth) for pth in checkpoints]
            elif which_checkpoint == 'max':
                epoch_batch  = np.array([[int(ckpt.split('_')[2]), int(ckpt.split('_')[4])]
                                         for ckpt in checkpoints], dtype=int)
                max_epoch_ = epoch_batch[np.where(epoch_batch[:,0] == max(epoch_batch[:,0]))[0]]#.reshape(1,-1)

                self.max_epoch, self.max_batch = np.squeeze(max_epoch_[np.where(max_epoch_[:,1] == max(max_epoch_[:,1]))[0]])
                checkpoint_path = os.path.join(self.ckpt_path,
                                               '%s_epoch_%s_batch_%s'%(self.which_net ,str(self.max_epoch),self.max_batch))
                print('Checkpoint latest at: ', str(checkpoint_path))
            else:
                raise ValueError('Provide valid checkpoint type')
            return checkpoint_path
        else:
            print('Checkpoints not found, Hence starting at batch 0 and epoch 0........')
            logging.info('No Checkpoint found, hence initializing random weights')
            return []


    def run_preprocessor(self, sess, dataIN, preprocess_graph, is_training):
        out_shape = [dataIN.shape[0]] + self.out_img_shape
        pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
        for img_no in np.arange(dataIN.shape[0]):
            feed_dict = {
                preprocess_graph['imageIN']: dataIN[img_no, :],
                preprocess_graph['is_training']: is_training
            }
            pp_imgs[img_no, :] = sess.run(
                    preprocess_graph['imageOUT'],
                    feed_dict=feed_dict
            )
            
        return pp_imgs



class Train(PropertyClassification):
    
    def _init__(self, params, which_net, image_type):
        PropertyClassification.__init__(self, params, which_net, image_type)
    
    def train(self, batchX, batchY, sess):
        batchX, batchY = unison_shuffled_copies(batchX, batchY)
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph, is_training=True)

        batchY_1hot = self.to_one_hot(batchY)
        feed_dict = {
            self.computation_graph['inpX']: preprocessed_data,
            self.computation_graph['inpY']: batchY_1hot,
            self.computation_graph['is_training']: True
        }

        _, out_prob, tr_acc, tr_loss, l_rate = sess.run(
                [self.computation_graph['optimizer'],
                 self.computation_graph['outProbs'],
                 self.computation_graph['accuracy'],
                 self.computation_graph['loss'],
                 self.computation_graph['l_rate']], feed_dict=feed_dict)

        tr_pred = sess.run(tf.argmax(out_prob, 1))
        tr_recall_score = Score.recall(batchY, tr_pred, reverse=True)
        tr_precision_score = Score.precision(batchY, tr_pred, reverse=True)
        
        
        if self.write_tensorboard_summary:
            out_prob, tr_acc, ls, _, l_rt, smry = sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)
            
        logging.info("Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s, Precision: %s, Recall: %s",
                     str(self.foldNUM),
                     str(self.epoch),
                     str(self.batch_num),
                     str("{:.6f}".format(tr_loss)),
                     str("{:.5f}".format(tr_acc)),
                     str("{:.5f}".format(tr_precision_score)),
                     str("{:.5f}".format(tr_recall_score)))
        
        return tr_loss, tr_acc, tr_precision_score, tr_recall_score, l_rate
    
    
    def cvalid(self, sess):
        feed_dict = {
            self.computation_graph['inpX']: self.cv_preprocessed_data,
            self.computation_graph['inpY']: self.cvbatchY_1hot,
            self.computation_graph['is_training']: False
        }

        cv_prob, cv_acc, cv_loss = sess.run([self.computation_graph['outProbs'],
                                              self.computation_graph['accuracy'],
                                              self.computation_graph['loss']], feed_dict=feed_dict)
        if self.write_tensorboard_summary:
            cv_prob, acc, smry = sess.run([self.computation_graph['outProbs'],
                                            self.computation_graph['accuracy'],
                                            self.merged_summary], feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)

        cv_pred = sess.run(tf.argmax(cv_prob, 1))
        cv_recall_score = Score.recall(self.cvbatchY, cv_pred, reverse=True)
        cv_precision_score = Score.precision(self.cvbatchY, cv_pred, reverse=True)

        logging.info("VALIDATION METRICs : Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s, Precision: %s, Recall: %s",
                     str(self.foldNUM),
                     str(self.epoch),
                     str(self.batch_num),
                     str("{:.6f}".format(cv_loss)),
                     str("{:.5f}".format(cv_acc)),
                     str("{:.5f}".format(cv_precision_score)),
                     str("{:.5f}".format(cv_recall_score)))
       
        return cv_loss, cv_acc, cv_precision_score, cv_recall_score


    def some_stuff(self, saver, sess):
        # LOAD CHECKPOINTS (WEIGHTS IF NEEDED)
        if self.use_checkpoint:
            checkpoint_path = self.get_checkpoint_path(which_checkpoint='max')
            if len(checkpoint_path) > 0:
                self.restore_checkpoint(checkpoint_path, saver, sess)
    
        # CREATE TENSOR BOARD SUMMARY WRITER (Writer opens up a file and starts writing summary for every epoch
        if self.write_tensorboard_summary:
            logging.info('TENSOR BOARD SUMMARY: Dumping Tensorboard summary')
            self.merged_summary, self.writer = summary_builder(sess, self.smry_path)
        
        # When we have already processed all the batches then we need to start a new epoch
        if self.max_batch == self.num_batches - 1:
            self.max_batch = 0
            self.max_epoch += 1
        elif self.max_batch == 0:
            self.max_batch = 0
        else:
            self.max_batch += 1  # When we have already run some batches for the epoch then we need to run from
            # the next batch
            
    def run_epoch(self, get_stats_at):
        saver = tf.train.Saver(max_to_keep=10)  # max_to_keep specifies the number of latest checkpoint to maintain
        self.max_batch = 0
        self.max_epoch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # GET LATEST CHECKPOINT, BATCH NUMBER TO START FROM AND ETC
            self.some_stuff(saver, sess)
            
            # CROSS-VALIDATION We load and pre-process the Cross-Validation set once, since we have to use it many times
            cvbatchX, self.cvbatchY = load_batch_data(image_type=self.image_type, image_shape=self.inp_img_shape,
                                                  which_data='cvalid')
            
            self.cvbatchY_1hot = self.to_one_hot(self.cvbatchY)
            self.cv_preprocessed_data = self.run_preprocessor(sess, cvbatchX, self.preprocess_graph, is_training=False)
            del cvbatchX

            
            # INITIATE EXECUTION (TRAINING AND TESTING)
            tr_acc_arr = []
            tr_loss_arr = []
            tr_precision_arr = []
            tr_recall_arr = []
            cv_acc_arr = []
            cv_loss_arr = []
            cv_precision_arr = []
            cv_recall_arr = []
            l_rate_arr = []
            for epoch in range(self.max_epoch, self.max_epoch + self.epochs):
                self.epoch = epoch
                
                for batch_num in range(self.max_batch, self.num_batches):
                    self.batch_num = batch_num
                    batchX, batchY = load_batch_data(image_type=self.image_type, image_shape=self.inp_img_shape, which_data='train_%s'%(batch_num))

                    tr_loss, tr_acc, tr_precision_score, tr_recall_score, l_rate = self.train(batchX, batchY, sess)
                    tr_loss_arr.append(tr_loss)
                    tr_acc_arr.append(tr_acc)
                    tr_precision_arr.append(tr_precision_score)
                    tr_recall_arr.append(tr_recall_score)
                    l_rate_arr.append(l_rate)
                    
                    if ((batch_num+1)%get_stats_at == 0) or (batch_num == self.num_batches -1):
                        
                        ## VALIDATION ACCURACY
                        cv_loss, cv_acc, cv_precision_score, cv_recall_score = self.cvalid(sess)
                        cv_loss_arr.append(cv_loss)
                        cv_acc_arr.append(cv_acc)
                        cv_precision_arr.append(cv_precision_score)
                        cv_recall_arr.append(cv_recall_score)
                        
                        # SAVE CHECKPOINTS TO THE PATH FOR EVERY EPOCH
                        if self.save_checkpoint:
                            logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                            checkpoint_path = os.path.join(
                                    self.ckpt_path,
                                    '%s_epoch_%s_batch_%s'%(self.which_net, str(epoch),str(batch_num))
                            )

                            saver.save(sess, checkpoint_path)#, write_meta_graph=False)
                  
        return tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr, l_rate_arr

    def run(self, num_epochs, num_batches, get_stats_at = 10):
        logging.info('INITIATING RUN ........')
        tf.reset_default_graph()
        self.foldNUM = 1
        self.epochs = num_epochs
        self.num_batches = num_batches + 1

        self.preprocess_graph = Preprocessing(inp_img_shape=self.inp_img_shape,
                                              crop_shape=self.crop_shape,
                                              out_img_shape=self.out_img_shape).preprocessImageGraph()
        
        if self.which_net == 'vgg':
            self.computation_graph = vgg(training=True)
        elif self.which_net == 'resnet':
            self.computation_graph = resnet(img_shape=self.out_img_shape)
        else:
            raise ValueError('Provide a valid Net type options ={vgg, resnet}')
        ########   RUN THE SESSION
        tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, \
            cv_recall_arr, l_rate_arr = self.run_epoch(get_stats_at)
        
        return tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr, l_rate_arr



