from __future__ import division, print_function, absolute_import

import logging
import os

import numpy as np
import tensorflow as tf

from src.config import pathDict
from src.conv_net.ops import summary_builder
from src.conv_net.resnet import mixture_of_experts
from src.conv_net.train import PropertyClassification
from src.conv_net.train import load_batch_data
from src.conv_net.utils import Score
from src.data_transformation.preprocessing import Preprocessing

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

m1_batch_dir = os.path.join(pathDict['general_batch_path'], 'aerial_cropped')
m2_batch_dir = os.path.join(pathDict['general_batch_path'], 'overlaid')


class MixtureModels(PropertyClassification):
    def _init__(self, params, device_type, which_net):
        PropertyClassification.__init__(self, params, device_type, which_net)

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

    def get_cvalid_data(self, sess):
        cvbatchX_m1, self.cvbatchY_m1 = load_batch_data(which_data='cvalid', force_dir_fetch=m1_batch_dir)
        cvbatchY_1hot_m1 = self.to_one_hot(self.cvbatchY_m1)
        cvbatchX_m1 = self.run_preprocessor(sess, cvbatchX_m1, self.preprocess_graph)

        cvbatchX_m2, cvbatchY_m2 = load_batch_data(which_data='cvalid', force_dir_fetch=m2_batch_dir)
        cvbatchY_1hot_m2 = self.to_one_hot(cvbatchY_m2)
        cvbatchX_m2 = self.run_preprocessor(sess, cvbatchX_m2, self.preprocess_graph)

        if not np.array_equal(np.array(self.cvbatchY_m1), np.array(cvbatchY_m2)):
            return ValueError('The response (label) array from the expert crossvalidation batch doesnt match')
        return cvbatchX_m1, cvbatchY_1hot_m1, cvbatchX_m2, cvbatchY_1hot_m2

    def get_training_batches(self, batch_num):
        tr_batchX_m1, tr_batchY_m1 = load_batch_data(which_data='train_%s' % (batch_num),
                                                     force_dir_fetch=m1_batch_dir)
        tr_batchX_m2, tr_batchY_m2 = load_batch_data(which_data='train_%s' % (batch_num),
                                                     force_dir_fetch=m2_batch_dir)

        if not np.array_equal(np.array(tr_batchY_m1), np.array(tr_batchY_m2)):
            return ValueError('The response (label) array from the expert training batch doesnt match')
        return tr_batchX_m1, tr_batchY_m1, tr_batchX_m2, tr_batchY_m2

    def train(self, batchX_m1, batchX_m2, batchY_m1, sess):
        processed_m1 = self.run_preprocessor(sess, batchX_m1, self.preprocess_graph)

        processed_m2 = self.run_preprocessor(sess, batchX_m2, self.preprocess_graph)

        batchY_1hot = self.to_one_hot(batchY_m1)

        feed_dict = {
            self.computation_graph['inpX1']: processed_m1,
            self.computation_graph['inpX2']: processed_m2,
            self.computation_graph['inpY']: batchY_1hot
        }

        _, out_prob, tr_acc, tr_loss, l_rate = sess.run(
            [self.computation_graph['optimizer'],
             self.computation_graph['outProbs'],
             self.computation_graph['accuracy'],
             self.computation_graph['loss'],
             self.computation_graph['l_rate']], feed_dict=feed_dict)

        tr_pred = sess.run(tf.argmax(out_prob, 1))
        tr_recall_score = Score.recall(batchY_m1, tr_pred, reverse=True)
        tr_precision_score = Score.precision(batchY_m1, tr_pred, reverse=True)

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

    def cvalid(self, cvbatchX_m1, cvbatchY_1hot_m1, cvbatchX_m2, cvbatchY_1hot_m2, sess):
        len_cv_data = cvbatchX_m1.shape[0]
        cv_batches_size = int(np.ceil(len_cv_data / self.cv_num_batches))
        cv_pred_arr = []
        cv_loss = 10000
        for ite in range(0, self.cv_num_batches):

            if ite != (self.cv_num_batches - 1):
                from_idx = ite * cv_batches_size
                to_idx = (ite * cv_batches_size) + cv_batches_size
            else:
                from_idx = ite * cv_batches_size
                to_idx = (ite * cv_batches_size) + (len_cv_data - (ite * cv_batches_size))

            cvX_m1 = cvbatchX_m1[from_idx: to_idx, :]
            cvY_m1 = cvbatchY_1hot_m1[from_idx: to_idx, :]

            cvX_m2 = cvbatchX_m2[from_idx: to_idx, :]
            cvY_m2 = cvbatchY_1hot_m2[from_idx: to_idx, :]

            logging.info('Running Cross Validation batch%s: cvX_m1.shape = %s, cvY_m1.shape = %s cvX_m2.shape = %s, '
                         'cvY_m2.shape = %s', str(ite), str(cvX_m1.shape), str(cvY_m1.shape), str(cvX_m2.shape),
                         str(cvY_m2.shape))

            feed_dict = {
                self.computation_graph['inpX1']: cvX_m1,
                self.computation_graph['inpX2']: cvX_m2,
                self.computation_graph['inpY']: cvY_m1
            }

            cv_prob, cv_acc, cv_loss = sess.run([self.computation_graph['outProbs'],
                                                 self.computation_graph['accuracy'],
                                                 self.computation_graph['loss']], feed_dict=feed_dict)

            if self.write_tensorboard_summary:
                cv_prob, acc, smry = sess.run([self.computation_graph['outProbs'],
                                               self.merged_summary], feed_dict=feed_dict)
                self.writer.add_summary(smry, self.epoch)

            cv_pred = sess.run(tf.argmax(cv_prob, 1))
            # print (len(list(cv_pred)), list(cv_pred))
            cv_pred_arr += list(cv_pred)
            # print (len(cv_pred_arr), cv_pred_arr)

        # print (len(self.cvbatchY_m1), len(cv_pred_arr))
        cv_acc_ = Score.accuracy(self.cvbatchY_m1, np.array(cv_pred_arr))
        cv_recall_score = Score.recall(self.cvbatchY_m1, np.array(cv_pred_arr), reverse=True)
        cv_precision_score = Score.precision(self.cvbatchY_m1, np.array(cv_pred_arr), reverse=True)

        logging.info(
            "VALIDATION METRICs : Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s, Precision: %s, "
            "Recall: %s",
            str(self.foldNUM),
            str(self.epoch),
            str(self.batch_num),
            str("{:.6f}".format(cv_loss)),
            str("{:.5f}".format(cv_acc_)),
            str("{:.5f}".format(cv_precision_score)),
            str("{:.5f}".format(cv_recall_score)))

        return cv_loss, cv_acc_, cv_precision_score, cv_recall_score

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
            cvbatchX_m1, cvbatchY_1hot_m1, cvbatchX_m2, cvbatchY_1hot_m2 = self.get_cvalid_data(
                sess)

            # INITIATE EXECUTION (TRAINING AND TESTING)
            tr_acc_arr, tr_loss_arr, tr_precision_arr, tr_recall_arr = [], [], [], []
            cv_acc_arr, cv_loss_arr, cv_precision_arr, cv_recall_arr = [], [], [], []
            l_rate_arr = []

            # GET TRAINING BATCH FROM OVERLAYED
            for epoch in range(self.max_epoch, self.max_epoch + self.epochs):
                self.epoch = epoch

                for batch_num in range(self.max_batch, self.num_batches + 1):
                    self.batch_num = batch_num
                    batchX_m1, batchY_m1, batchX_m2, _ = self.get_training_batches(self.batch_num)

                    tr_loss, tr_acc, tr_precision_score, tr_recall_score, l_rate = self.train(batchX_m1, batchX_m2,
                                                                                              batchY_m1, sess)
                    tr_loss_arr.append(tr_loss)
                    tr_acc_arr.append(tr_acc)
                    tr_precision_arr.append(tr_precision_score)
                    tr_recall_arr.append(tr_recall_score)
                    l_rate_arr.append(l_rate)

                    if ((batch_num + 1) % get_stats_at == 0) or (batch_num == self.num_batches):

                        ## VALIDATION ACCURACY
                        cv_loss, cv_acc, cv_precision_score, cv_recall_score = self.cvalid(cvbatchX_m1,
                                                                                           cvbatchY_1hot_m1,
                                                                                           cvbatchX_m2,
                                                                                           cvbatchY_1hot_m2, sess)

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
            model_inp_img_shape=self.model_inp_img_shape).preprocessImageGraph(is_training=True)

        # if self.which_net == 'vgg':
        #     self.computation_graph = vgg(training=True)
        self.computation_graph = mixture_of_experts(img_shape=self.model_inp_img_shape,
                                                    device_type=self.device_type, use_dropout=True)

        ########   RUN THE SESSION
        # self.run_epoch(get_stats_at)
        tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, cv_precision_arr, \
        cv_recall_arr, l_rate_arr = self.run_epoch(get_stats_at)

        return tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, cv_loss_arr, cv_acc_arr, \
               cv_precision_arr, \
               cv_recall_arr, l_rate_arr

        # MixtureModels(dict(pprocessor_inp_img_shape=[224,224,3],
        #                         pprocessor_inp_crop_shape=[],
        #                         model_inp_img_shape=[224,224,3],
        #                         learning_rate=0.0005,
        #                         use_checkpoint=True,
        #                         save_checkpoint=True,
        #                         write_tensorboard_summary=False
        #                         ),
        #               device_type='cpu',
        #               which_net='resnetMoE').run(num_epochs=10, num_batches=70, cv_num_batches=1, get_stats_at=10)