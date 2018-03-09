import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from conv_net.vgg import vgg
from conv_net.resnet import resnet
from config import pathDict, myNet
from conv_net.ops import summary_builder
from data_transformation.data_io import getPickleFile
from data_transformation.preprocessing import Preprocessing

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def load_batch_data(image_type, image_shape, which_data='cv'):
    batch_file_name = None

    if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside','google_overlayed']:
        raise ValueError('Can not identify the image type %s, Please provide a valid one' % (str(image_type)))
    
    
    data_path = pathDict['%s_batch_path' % (str(image_type))]
    
    batch_file_name = '%s.pickle' % (which_data)
    # print (os.path.exists(os.path.join(data_path, batch_file_name)))
    if not os.path.exists(os.path.join(data_path, batch_file_name)):
        raise ValueError('The batch file doesnt seem to exists')
    else:
        logging.info('Loading the data from path %s ', str(os.path.join(data_path, batch_file_name)))

    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY, label_dict = getPickleFile(data_path, batch_file_name)
    
    return dataX, dataY, label_dict



class PropertyClassification():
    def __init__(self, params, which_net, image_type, inp_image_shape):
        params_keys = list(params.keys())
        self.which_net = which_net
        myNet['inp_image_shape'] = inp_image_shape
        
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
        if len(checkpoints) > 0:
            if which_checkpoint == 'all':
                checkpoint_path = [os.path.join(self.ckpt_path, pth) for pth in checkpoints]
            elif which_checkpoint == 'max':
                epoch_batch  = np.array([[int(ckpt.split('_')[2]), int(ckpt.split('_')[4])]
                                         for ckpt in checkpoints], dtype=int)
                # print (epoch_batch[:,0], '\n', epoch_batch[:,1], '\n')
                max_epoch_ = epoch_batch[np.where(epoch_batch[:,0] == max(epoch_batch[:,0]))[0]]#.reshape(1,-1)
                # print(max_epoch_, '\n')
                self.max_epoch, self.max_batch = np.squeeze(max_epoch_[np.where(max_epoch_[:,1] == max(max_epoch_[:,1]))[0]])
                checkpoint_path = os.path.join(self.ckpt_path,
                                               '%s_epoch_%s_batch_%s'%(self.which_net ,str(self.max_epoch),self.max_batch))
                print ('Checkpoint latest at: ', checkpoint_path)
            else:
                raise ValueError('Provide valid checkpoint type')
            return checkpoint_path
        else:
            print('Checkpoints not found, Hence starting at batch 0 and epoch 0........')
            logging.info('No Checkpoint found, hence initializing random weights')
            return []


    def run_preprocessor(self, sess, dataIN, preprocess_graph, is_training):
        # logging.info('INITIATING PREPROCESSING.................')
        out_shape = [dataIN.shape[0]] + myNet['crop_shape']
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
    
    def _init__(self, params, which_net, image_type, inp_image_shape):
        PropertyClassification.__init__(self, params, which_net, image_type, inp_image_shape)
    
    def train(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph, is_training=True)
        feed_dict = {
            self.train_graph['inpX']: preprocessed_data,
            self.train_graph['inpY']: batchY,
            self.train_graph['is_training']: True
        }
        
        if self.write_tensorboard_summary:
            out_prob, tacc, ls, _, l_rt, smry = sess.run([self.train_graph['outProbs'],
                                                            self.train_graph['accuracy'],
                                                            self.train_graph['loss'],
                                                            self.train_graph['optimizer'],
                                                            self.train_graph['l_rate'],
                                                            self.merged_summary], feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)
        else:
            out_prob, tacc, ls, _, l_rt = sess.run([self.train_graph['outProbs'],
                                              self.train_graph['accuracy'],
                                              self.train_graph['loss'],
                                              self.train_graph['optimizer'],
                                              self.train_graph['l_rate']], feed_dict=feed_dict)
        
        # acc1 = self.accuracy(y=batchY, y_hat=out_prob)
        
        logging.info("Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s",
                     str(self.foldNUM),
                     str(self.epoch),
                     str(self.batch_num),
                     str("{:.6f}".format(ls)),
                     str("{:.5f}".format(tacc)))
        
        return tacc
    
    def cvalid(self, sess):
        feed_dict = {
            self.train_graph['inpX']: self.cv_preprocessed_data,
            self.train_graph['inpY']: self.cvbatchY,
            self.train_graph['is_training']: False
        }
    
        if self.write_tensorboard_summary:
            out_prob, acc, smry = sess.run([self.train_graph['outProbs'],
                                                         self.train_graph['accuracy'],
                                                         self.merged_summary], feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)
        else:
            out_prob, acc, = sess.run([self.train_graph['outProbs'],
                                                   self.train_graph['accuracy']], feed_dict=feed_dict)
    
        acc1 = self.accuracy(y=self.cvbatchY, y_hat=out_prob)
        return acc1

    def run_epoch(self):
        saver = tf.train.Saver(max_to_keep=20)
        self.max_batch = 0
        self.max_epoch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # LOAD CHECKPOINTS (WEIGHTS IF NEEDED)
            if self.use_checkpoint:
                checkpoint_path = self.get_checkpoint_path(which_checkpoint='max')
                if len(checkpoint_path) > 0 :
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
                self.max_batch += 1   # When we have already run some batches for the epoch then we need to run from
                # the next batch
            
            # CROSS-VALIDATION We load and Preprocess the Cross-Validation set once, since we have to use it many times
            cvbatchX, cvbatchY,_ = load_batch_data(image_type=self.image_type, image_shape=myNet['inp_image_shape'],
                                                   which_data='cv')
            self.cvbatchY = self.to_one_hot(cvbatchY)
            self.cv_preprocessed_data = self.run_preprocessor(sess, cvbatchX, self.preprocess_graph, is_training=False)
            del cvbatchX
            del cvbatchY
            
            
            # INITIATE EXECUTION (TRAINING AND TESTING)
            for epoch in range(self.max_epoch, self.max_epoch + self.epochs):
                self.epoch = epoch
                avg_accuracy = 0
                get_stats_at = 20

                for batch_num in range(self.max_batch, self.num_batches):
                    self.batch_num = batch_num
                    batchX, batchY, label_dict = load_batch_data(
                            image_type=self.image_type,
                            image_shape=myNet['inp_image_shape'],
                            which_data='tr%s'%(batch_num))
                    
                    batchY = self.to_one_hot(batchY)
                    train_acc = self.train(batchX, batchY, sess)

                    avg_accuracy += train_acc

                    if ((batch_num+1)%get_stats_at == 0) or (batch_num == self.num_batches -1):
                        print("Epoch: " + str(epoch) + ", Batch: " + str(batch_num) +
                              ", AVG Training Accuracy= " + "{:.5f}".format(avg_accuracy/get_stats_at))
                        avg_accuracy = 0
                        
                        
                        ## VALIDATION ACCURACY
                        valid_acc = self.cvalid( sess)
                        print("Epoch: " + str(epoch) + ", Batch: " + str(batch_num) +
                              ", Validation Accuracy= " + "{:.5f}".format(valid_acc))
                        
                        # SAVE CHECKPOINTS TO THE PATH FOR EVERY EPOCH
                        if self.save_checkpoint:
                            logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                            checkpoint_path = os.path.join(
                                    self.ckpt_path,
                                    '%s_epoch_%s_batch_%s'%(self.which_net, str(epoch),str(batch_num))
                            )

                            saver.save(sess, checkpoint_path, write_meta_graph=False)

    def run(self, num_epochs, num_batches):
        logging.info('INITIATING RUN ........')
        self.foldNUM = 1
        self.epochs = num_epochs
        self.num_batches = num_batches

        self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['inp_image_shape'])
        
        if self.which_net == 'vgg':
            self.train_graph = vgg(training=True)
        elif self.which_net == 'resnet':
            self.train_graph = resnet()
        else:
            raise ValueError('Provide a valid Net type options ={vgg, resnet}')
        ########   RUN THE SESSION
        self.run_epoch()


class Test(PropertyClassification):
    
    def _init__(self, params, which_net, image_type, inp_image_shape):
        PropertyClassification.__init__(self, params, which_net, image_type, inp_image_shape)
        
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(sess, batchX, self.preprocess_graph, is_training=False)
        feed_dict = {
            self.test_graph['inpX']: preprocessed_data
        }

        out_prob = sess.run(self.test_graph['outProbs'], feed_dict=feed_dict)
        

        v_acc = self.accuracy(y=batchY, y_hat=out_prob)

        print("Epoch: " + str(self.epoch) +
              ", Cross Validation Accuracy= " + "{:.5f}".format(v_acc))

        return out_prob, v_acc

    def test(self,  checkpoint_path):
        saver = tf.train.Saver()
        self.max_checkpoint_num = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            batchX, batchY, label_dict = load_batch_data(
                    image_type=self.image_type,
                    image_shape=myNet['inp_image_shape'],
                    which_data='cv')
            
            
            self.epoch = int(str(checkpoint_path.split('.')[0]).split('_')[-1])
            self.restore_checkpoint(checkpoint_path, saver, sess)
            batchY_1hot = self.to_one_hot(batchY)
            out_prob, acc = self.cvalid(batchX, batchY_1hot, sess)
        
        return batchY, out_prob, acc
        

    def run(self, dump_stats=False):
        logging.info('INITIATING TEST ........')
        self.dump_stats = dump_stats
        checkpoint_paths = self.get_checkpoint_path(which_checkpoint='all')

        stats_matrix = []
        colnames = []
        for path_num, chk_path in enumerate(checkpoint_paths):
            
            self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['inp_image_shape'])

            if self.which_net == 'vgg':
                print ('Test Graph: VGG')
                self.test_graph = vgg(training=False)
            elif self.which_net == 'resnet':
                print('Test Graphs: RESNET')
                self.test_graph = resnet()
            else:
                raise ValueError('Provide a valid Net type options ={vgg, resnet}')
            
            ########   RUN THE SESSION
            batchY, out_prob, acc = self.test(chk_path)
            y_hat = np.argmax(out_prob, 1).reshape(-1,1)
            
            # print('1 ', len(batchY))
            # print('2 ',batchY.shape)
            # print ('3 ',len(out_prob))
            # print ('4 ',out_prob.shape)
            
            if self.dump_stats:

                if path_num == 0:
                    colnames.append('Label')
                    stats_matrix = batchY.reshape(-1,1)
                
                # print (out_prob[0:5])
                # print (np.argmax(out_prob, 1))
                print(stats_matrix.shape)
                colnames.append(os.path.basename(chk_path.split('.')[0] + '_pred'))
                colnames.append(os.path.basename(chk_path.split('.')[0] + '_prob'))
                stats_matrix = np.column_stack((
                    stats_matrix, y_hat,
                    np.maximum(out_prob[:,0], out_prob[:,1]).reshape(-1,1))
                )

            tf.reset_default_graph()

        if self.dump_stats:
            stats_matrix = pd.DataFrame(stats_matrix, columns=colnames)
            stats_path = os.path.join(pathDict['%s_pred_stats' % str(self.image_type)], 'pred_stats.csv')
            stats_matrix.to_csv(stats_path, index=None)
        # print(len(colnames), colnames)
        # print(stats_matrix.shape)
                
                
                
        
            
   
debugg  = False

if debugg:
    # image_type = 'aerial'
    # image_shape = [224,224,3]
    image_type = 'assessor'
    inp_image_shape = [260, 260, 3]
    # image_type = 'streetside'
    # image_shape = [260,260,3]
    
    run = True
    if run:
        Train(dict(use_checkpoint=True,
                   save_checkpoint=True,
                   write_tensorboard_summary=True
                   ),
              which_net='resnet', # vgg
              image_type=image_type,
              inp_image_shape=inp_image_shape).run(num_epochs=3,
                                         num_batches=149)
        
        
        # Test(params={}, which_net='resnet', image_type=image_type).run()