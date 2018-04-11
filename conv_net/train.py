import logging
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from conv_net.utils import Score

from conv_net.utils import Score
from conv_net.resnet import resnet
from conv_net.convnet import conv_net
from conv_net.conv_autoencoder import conv_autoencoder
from config import pathDict, myNet
from conv_net.ops import summary_builder
from data_transformation.data_io import getH5File
from data_transformation.preprocessing import Preprocessing
from conv_net.utils import unison_shuffled_copies

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def load_batch_data(which_data='cvalid', force_dir_fetch=None):
    if force_dir_fetch:
        data_path =force_dir_fetch
    else:
        data_path = pathDict['batch_path']  # % (str(image_type))]
        
    batch_file_name = '%s' % (which_data)
    
    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY = getH5File(data_path, batch_file_name)
    
    return dataX, dataY


class PropertyClassification(object):
    def __init__(self, params, device_type, which_net):
        params_keys = list(params.keys())
        self.which_net = which_net
        self.device_type = device_type
        
        if 'learning_rate' in params_keys:
            myNet['learning_rate'] = params['learning_rate']
        
        if 'pprocessor_inp_img_shape' in params_keys:
            self.pprocessor_inp_img_shape = params['pprocessor_inp_img_shape']
        else:
            raise ValueError('You should provide the input image shape')
        
        if 'model_inp_img_shape' in params_keys:
            self.model_inp_img_shape = params['model_inp_img_shape']
        else:
            raise ValueError('You should provide the input image shape')
        
        if 'pprocessor_inp_crop_shape' in params_keys:
            self.pprocessor_inp_crop_shape = params['pprocessor_inp_crop_shape']
        else:
            self.pprocessor_inp_crop_shape = []
        
        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']
        else:
            self.use_checkpoint = False
        
        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']
        else:
            self.save_checkpoint = False
        
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
        else:
            self.write_tensorboard_summary = False
        
        if self.save_checkpoint or self.use_checkpoint:
            self.ckpt_path = os.path.join(pathDict['checkpoint_path'], self.which_net)
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            print('Dumping/Retreiving Checkpoints to/from %s', self.ckpt_path)
        
        if self.write_tensorboard_summary:
            self.smry_path = os.path.join(pathDict['summary_path'], self.which_net)
            if not os.path.exists(self.smry_path):
                os.makedirs(self.smry_path)
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
                epoch_batch = np.array([[int(ckpt.split('_')[2]), int(ckpt.split('_')[4])]
                                        for ckpt in checkpoints], dtype=int)
                max_epoch_ = epoch_batch[np.where(epoch_batch[:, 0] == max(epoch_batch[:, 0]))[0]]  # .reshape(1,-1)
                
                self.max_epoch, self.max_batch = np.squeeze(
                        max_epoch_[np.where(max_epoch_[:, 1] == max(max_epoch_[:, 1]))[0]])
                checkpoint_path = os.path.join(self.ckpt_path,
                                               '%s_epoch_%s_batch_%s' % (
                                                   self.which_net, str(self.max_epoch), self.max_batch))
                print('Checkpoint latest at: ', str(checkpoint_path))
            else:
                raise ValueError('Provide valid checkpoint type')
            return checkpoint_path
        else:
            print('Checkpoints not found, Hence starting at batch 0 and epoch 0........')
            logging.info('No Checkpoint found, hence initializing random weights')
            return []
    
    def run_preprocessor(self, sess, dataIN, preprocess_graph):
        out_shape = [dataIN.shape[0]] + self.model_inp_img_shape
        pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
        for img_no in np.arange(dataIN.shape[0]):
            feed_dict = {
                preprocess_graph['imageIN']: dataIN[img_no, :]
            }
            pp_imgs[img_no, :] = sess.run(
                    preprocess_graph['imageOUT'],
                    feed_dict=feed_dict
            )
        
        return pp_imgs


class Train(PropertyClassification):
    def _init__(self, params, device_type, which_net):
        PropertyClassification.__init__(self, params, device_type, which_net)
    
    def train(self, batchX, batchY, sess):
        batchX, batchY = unison_shuffled_copies(batchX, batchY)
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph)
        
        batchY_1hot = self.to_one_hot(batchY)
        feed_dict = {
            self.computation_graph['inpX']: preprocessed_data,
            self.computation_graph['inpY']: batchY_1hot,
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
        len_cv_data = self.cv_preprocessed_data.shape[0]
        cv_batches_size = int(np.ceil(len_cv_data / self.cv_num_batches))
        
        inter_loss = 0
        inter_acc = 0
        inter_precision = 0
        inter_recall = 0
        tot_len = 0
        for ite in range(0, self.cv_num_batches):
            
            if ite != (self.cv_num_batches - 1):
                from_idx = ite * cv_batches_size
                to_idx = (ite * cv_batches_size) + cv_batches_size
            else:
                from_idx = ite * cv_batches_size
                to_idx = (ite * cv_batches_size) + (len_cv_data - (ite * cv_batches_size))
            
            cvX = self.cv_preprocessed_data[from_idx: to_idx, :]
            cvY = self.cvbatchY_1hot[from_idx: to_idx, :]
            
            logging.info('Running Cross Validation batch%s: cvX.shape = %s, cvY.shape = %s', str(ite), str(cvX.shape),
                         str(cvY.shape))
            
            feed_dict = {
                self.computation_graph['inpX']: cvX,
                self.computation_graph['inpY']: cvY
            }
            
            cv_prob, cv_acc, cv_loss = sess.run([self.computation_graph['outProbs'],
                                                 self.computation_graph['accuracy'],
                                                 self.computation_graph['loss']], feed_dict=feed_dict)
            
            if self.write_tensorboard_summary:
                cv_prob, acc, smry = sess.run([self.computation_graph['outProbs'],
                                               self.merged_summary], feed_dict=feed_dict)
                self.writer.add_summary(smry, self.epoch)
            
            cv_pred = sess.run(tf.argmax(cv_prob, 1))
            cv_recall_score = Score.recall(self.cvbatchY[from_idx: to_idx], cv_pred, reverse=True)
            cv_precision_score = Score.precision(self.cvbatchY[from_idx: to_idx], cv_pred, reverse=True)
            
            # We do the below to produce the average
            inter_loss += cv_loss * len(cvX)
            inter_acc += cv_acc * len(cvX)
            inter_recall += cv_recall_score * len(cvX)
            inter_precision += cv_precision_score * len(cvX)
            tot_len += len(cvX)
        
        inter_loss = inter_loss / tot_len
        inter_acc = inter_acc / tot_len
        inter_recall = inter_recall / tot_len
        inter_precision = inter_precision / tot_len
        
        logging.info(
                "VALIDATION METRICs : Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s, Precision: %s, "
                "Recall: %s",
                str(self.foldNUM),
                str(self.epoch),
                str(self.batch_num),
                str("{:.6f}".format(inter_loss)),
                str("{:.5f}".format(inter_acc)),
                str("{:.5f}".format(inter_precision)),
                str("{:.5f}".format(inter_recall)))
        
        return inter_loss, inter_acc, inter_precision, inter_recall
    
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
        if self.max_batch == self.num_batches:
            self.max_batch = 0
            self.max_epoch += 1
        elif self.max_batch == 0:
            self.max_batch = 0
        else:
            self.max_batch += 1  # When we have already run some batches for the epoch then we need to run from
            # the next batch
    
    def run_epoch(self, get_stats_at):
        saver = tf.train.Saver(max_to_keep=5)  # max_to_keep specifies the number of latest checkpoint to maintain
        self.max_batch = 0
        self.max_epoch = 0
        
        config_ = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config_) as sess:
            sess.run(tf.global_variables_initializer())
            
            # GET LATEST CHECKPOINT, BATCH NUMBER TO START FROM AND ETC
            self.some_stuff(saver, sess)
            
            # CROSS-VALIDATION We load and pre-process the Cross-Validation set once, since we have to use it many times
            cvbatchX, self.cvbatchY = load_batch_data(which_data='cvalid')
            
            self.cvbatchY_1hot = self.to_one_hot(self.cvbatchY)
            self.cv_preprocessed_data = self.run_preprocessor(sess, cvbatchX, self.preprocess_graph)
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
                
                for batch_num in range(self.max_batch, self.num_batches + 1):
                    self.batch_num = batch_num
                    batchX, batchY = load_batch_data(which_data='train_%s' % (batch_num))
                    
                    tr_loss, tr_acc, tr_precision_score, tr_recall_score, l_rate = self.train(batchX, batchY, sess)
                    tr_loss_arr.append(tr_loss)
                    tr_acc_arr.append(tr_acc)
                    tr_precision_arr.append(tr_precision_score)
                    tr_recall_arr.append(tr_recall_score)
                    l_rate_arr.append(l_rate)
                    
                    if ((batch_num + 1) % get_stats_at == 0) or (batch_num == self.num_batches):
                        
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
                                    '%s_epoch_%s_batch_%s' % (self.which_net, str(epoch), str(batch_num))
                            )
                            
                            saver.save(sess, checkpoint_path)  # , write_meta_graph=False)
        
        return tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, \
               cv_recall_arr, l_rate_arr
    
    def run(self, num_epochs, num_batches, cv_num_batches, get_stats_at=10):
        logging.info('INITIATING RUN ........')
        tf.reset_default_graph()
        self.foldNUM = 1
        self.epochs = num_epochs
        self.num_batches = num_batches
        self.cv_num_batches = cv_num_batches
        
        self.preprocess_graph = Preprocessing(
                pprocessor_inp_img_shape=self.pprocessor_inp_img_shape,
                pprocessor_inp_crop_shape=self.pprocessor_inp_crop_shape,
                model_inp_img_shape=self.model_inp_img_shape).preprocessImageGraph(is_training = True)
        
        # if self.which_net == 'vgg':
        #     self.computation_graph = vgg(training=True)
        if self.which_net == 'resnet':
            self.computation_graph = resnet(img_shape=self.model_inp_img_shape, device_type=self.device_type, use_dropout=True)
        elif self.which_net == 'convnet':
            self.computation_graph = conv_net(img_shape=self.model_inp_img_shape, device_type=self.device_type)
        else:
            raise ValueError('Provide a valid Net type options ={vgg, resnet}')
        ########   RUN THE SESSION
        tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, \
        cv_recall_arr, l_rate_arr = self.run_epoch(
            get_stats_at)
        
        return tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, \
               cv_recall_arr, l_rate_arr





class TrainConvEnc(PropertyClassification):
    def _init__(self, params, device_type, which_net):
        PropertyClassification.__init__(self, params, device_type, which_net)
    
    def plot(self, X_true, X_reconstructed):
        n = 10
        fig, ax = plt.subplots(2, n, figsize=(20, 4))
        ax = ax.ravel()
        
        imaga_indexlist = np.arange(120, 130)
        for i in range(0, n):
            ax[i].imshow(X_true[imaga_indexlist[i]])
        
        for i in range(n, n + n):
            ax[i].imshow(X_reconstructed[imaga_indexlist[i - n]])
        
        fig.show()
        plt.pause(6)
        plt.close()
    
    def cluster_score(self, out):
        random_state_arr = [443, 882]
        for i in range(0, 2):
            kmeans = KMeans(n_clusters=2, n_init=100, random_state=random_state_arr[i])
            kmeans = kmeans.fit(out)
            labels = kmeans.predict(out)
            centroids = kmeans.cluster_centers_
            
            scr = Score.accuracy(self.cvbatchY, labels)
            print('i = %s Score = ' % str(i), scr)
    
    def train(self, batchX, batchY, sess):
        batchX, batchY = unison_shuffled_copies(batchX, batchY)
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph)
        
        # batchY_1hot = self.to_one_hot(batchY)
        feed_dict = {
            self.computation_graph['inpX']: preprocessed_data,
            # self.computation_graph['inpY']: batchY_1hot,
        }
        
        tr_lr, tr_loss, _ = sess.run([self.computation_graph['learning_rate'],
                                      self.computation_graph['loss'],
                                      self.computation_graph['optimizer']], feed_dict=feed_dict)
        
        logging.info("Fold: %s, epoch: %s, batch: %s, Loss: %s",
                     str(self.foldNUM),
                     str(self.epoch),
                     str(self.batch_num),
                     str("{:.6f}".format(tr_loss)))
        
        return tr_loss, tr_lr
    
    def cvalid(self, sess):
        feed_dict = {self.computation_graph['inpX']: self.cv_preprocessed_data}
        (cv_enc, cv_dec, sig_logits, rec_mse,
         rec_entrpy, cv_loss) = sess.run([self.computation_graph['encoded'],
                                          self.computation_graph['decoded'],
                                          self.computation_graph['sigmoid_logits'],
                                          self.computation_graph['reconstructionMSE'],
                                          self.computation_graph['reconstructionEntropy'],
                                          self.computation_graph['loss']], feed_dict=feed_dict)
        
        logging.info(
                "VALIDATION METRICs : Fold: %s, epoch: %s, batch: %s, Loss: %s",
                str(self.foldNUM),
                str(self.epoch),
                str(self.batch_num),
                str("{:.6f}".format(cv_loss)))
        
        return cv_enc, cv_dec, sig_logits, rec_mse, rec_entrpy, cv_loss
    
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
    
    def run_epoch(self, get_stats_at, plot=False, cluster_score=True):
        saver = tf.train.Saver(max_to_keep=10)  # max_to_keep specifies the number of latest checkpoint to maintain
        self.max_batch = 0
        self.max_epoch = 0
        
        config_ = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config_) as sess:
            sess.run(tf.global_variables_initializer())
            
            # GET LATEST CHECKPOINT, BATCH NUMBER TO START FROM AND ETC
            self.some_stuff(saver, sess)
            
            # CROSS-VALIDATION We load and pre-process the Cross-Validation set once, since we have to use it many times
            cvbatchX, self.cvbatchY = load_batch_data(which_data='cvalid')
            print('cvalid_shape: ', cvbatchX.shape)
            self.cv_preprocessed_data = self.run_preprocessor(sess, cvbatchX, self.preprocess_graph)
            del cvbatchX
            
            # INITIATE EXECUTION (TRAINING AND TESTING)
            tr_loss_arr = []
            cv_loss_arr = []
            l_rate_arr = []
            cv_sigmoid_logits = []
            cv_reconstruction_mse = []
            cv_reconstruction_entropy = []
            for epoch in range(self.max_epoch, self.max_epoch + self.epochs):
                self.epoch = epoch
                
                for batch_num in range(self.max_batch, self.num_batches):
                    self.batch_num = batch_num
                    batchX, batchY = load_batch_data(which_data='train_%s' % (batch_num))
                    
                    tr_loss, l_rate = self.train(batchX, batchY, sess)
                    tr_loss_arr.append(tr_loss)
                    l_rate_arr.append(l_rate)
                
                if ((epoch) % get_stats_at == 0) or (epoch == self.max_epoch + self.epochs - 1):
                    
                    ## VALIDATION ACCURACY
                    (cv_encodings, cv_decodings, cv_sigmoid_logits, cv_reconstruction_mse,
                     cv_reconstruction_entropy, cv_loss) = self.cvalid(sess)
                    # print (cv_encodings)
                    cv_loss_arr.append(cv_loss)
                    
                    if plot:
                        self.plot(X_true=self.cv_preprocessed_data, X_reconstructed=cv_sigmoid_logits)
                    
                    if cluster_score:
                        self.cluster_score(out=cv_encodings)
                    
                    # SAVE CHECKPOINTS TO THE PATH FOR EVERY EPOCH
                    if self.save_checkpoint:
                        logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                        checkpoint_path = os.path.join(
                                self.ckpt_path,
                                '%s_epoch_%s_batch_%s' % (self.which_net, str(epoch), str(batch_num))
                        )
                        
                        saver.save(sess, checkpoint_path)  # , write_meta_graph=False)
        
        return tr_loss_arr, cv_loss_arr, l_rate_arr, cv_reconstruction_mse, cv_reconstruction_entropy, cv_encodings
    
    def run(self, num_epochs, num_batches, get_stats_at=10, plot=False, cluster_score=True):
        logging.info('INITIATING RUN ........')
        tf.reset_default_graph()
        self.foldNUM = 1
        self.epochs = num_epochs
        self.num_batches = num_batches
        
        self.preprocess_graph = Preprocessing(
                pprocessor_inp_img_shape=self.pprocessor_inp_img_shape,
                pprocessor_inp_crop_shape=self.pprocessor_inp_crop_shape,
                model_inp_img_shape=self.model_inp_img_shape).preprocessImageGraph(is_training=True)
        
        if self.which_net == 'autoencoder':
            self.computation_graph = conv_autoencoder(img_shape=self.model_inp_img_shape, device_type=self.device_type)
        else:
            raise ValueError('Net type not understood, Make sure you typed : "autoencoder"')
        ########   RUN THE SESSION
        tr_loss_arr, cv_loss_arr, l_rate_arr, cv_reconstruction_mse, reconstruction_entropy, cv_encodings = \
            self.run_epoch(
                    get_stats_at, plot=plot, cluster_score=cluster_score)
        
        return tr_loss_arr, cv_loss_arr, l_rate_arr, cv_reconstruction_mse, reconstruction_entropy, self.cvbatchY, \
               cv_encodings

#
# debugg = False
# encoder = False
#
# if debugg:
#     if encoder:
#         max_batches = 2
#         # if train:
#         tr_obj = TrainConvEnc(dict(pprocessor_inp_img_shape=[224, 400, 3],
#                                    pprocessor_inp_crop_shape=[128, 128, 3],
#                                    model_inp_img_shape=[128, 128, 3],
#                                    use_checkpoint=True,
#                                    save_checkpoint=True,
#                                    write_tensorboard_summary=False
#                                    ),
#                               device_type='cpu',
#                               which_net='autoencoder',  # vgg
#                               image_type='assessor_code')
#         (tr_loss_arr, cv_loss_arr, l_rate_arr, cv_reconstruction_mse,
#          cv_reconstruction_entropy, cvY_label) = tr_obj.run(num_epochs=100, num_batches=max_batches + 1, get_stats_at=3,
#                                                             plot=True)  # + 1)
#         print('')
#         print('')
#         print(cv_reconstruction_mse)
#         print('')
#         print('')
#         print(cv_reconstruction_entropy)
#
#     else:
#         max_batches = 66
#         # if train:
#         tr_obj = Train(dict(pprocessor_inp_img_shape=[400, 400, 3],
#                             pprocessor_inp_crop_shape=[96, 96, 3],
#                             model_inp_img_shape=[96, 96, 3],
#                             use_checkpoint=True,
#                             save_checkpoint=True,
#                             write_tensorboard_summary=False
#                             ),
#                        device_type='cpu',
#                        which_net='convnet',  # vgg
#                        image_type='google_overlayed')
#         (tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
#          cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr,
#          l_rate_arr) = tr_obj.run(num_epochs=3, num_batches=max_batches, get_stats_at=10)  # + 1)
#
