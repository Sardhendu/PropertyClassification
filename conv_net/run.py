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
    if image_type == 'aerial':
        data_path = pathDict['aerial_batch_path']
    elif image_type == 'assessor':
        data_path = pathDict['assessor_batch_path']
    elif image_type == 'streetside':
        data_path = pathDict['streetside_batch_path']
    else:
        raise ValueError('The image type doesnt match the type handled')
    
    batch_file_name = '%s.pickle' % (which_data)
    
    if not os.path.exists(os.path.join(data_path, batch_file_name)):
        raise ValueError('The batch file doesnt seem to exists')
    else:
        logging.info('Loading the data from path %s ', str(os.path.join(data_path, batch_file_name)))

    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY, label_dict = getPickleFile(data_path, batch_file_name)
    
    return dataX, dataY, label_dict



class PropertyClassification():
    def __init__(self, params, which_net, image_type):
        params_keys = list(params.keys())
        self.which_net = which_net
        
        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']
        
        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']
        
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
        
        if image_type == 'aerial':
            self.ckpt_path = os.path.join(pathDict['aerial_ckpt_path'], self.which_net )
            self.smry_path = os.path.join(pathDict['aerial_smry_path'], self.which_net )
            # aerial images are small in size, so to accomodate it, we override the input shape them here
            myNet['image_shape'] = [224, 224, 3]  # override the shape
        elif image_type == 'assessor':
            self.ckpt_path = os.path.join(pathDict['assessor_ckpt_path'], self.which_net )
            self.smry_path = os.path.join(pathDict['assessor_smry_path'], self.which_net )
        elif image_type == 'streetside':
            self.ckpt_path = os.path.join(pathDict['streetside_ckpt_path'], self.which_net )
            self.smry_path = os.path.join(pathDict['streetside_smry_path'], self.which_net )
        else:
            raise ValueError('Provide a valid image type')

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
                checkpoint_path = [os.path.join(self.ckpt_path, pth+'.ckpt') for pth in checkpoints]
            elif which_checkpoint == 'max':
                epoch_batch  = np.array([[int(ckpt.split('_')[2]), int(ckpt.split('_')[4])]
                                         for ckpt in checkpoints], dtype=int)
                # print (epoch_batch[:,0], '\n', epoch_batch[:,1], '\n')
                max_epoch_ = epoch_batch[np.where(epoch_batch[:,0] == max(epoch_batch[:,0]))[0]]#.reshape(1,-1)
                # print(max_epoch_, '\n')
                self.max_epoch, self.max_batch = np.squeeze(max_epoch_[np.where(max_epoch_[:,1] == max(max_epoch_[:,1]))[0]])
                checkpoint_path = os.path.join(self.ckpt_path,
                                               '%s_epoch_%s_batch_%s.ckpt'%(self.which_net ,str(self.max_epoch),self.max_batch))
                print ('Checkpoint latest at: ', checkpoint_path)
            else:
                raise ValueError('Provide valid checkpoint type')
            return checkpoint_path
        else:
            print('Checkpoints not found, Hence starting at batch 0 and epoch 0........')
            logging.info('No Checkpoint found, hence initializing random weights')
            return []


    def run_preprocessor(self, dataIN, preprocess_graph, sess):
        # logging.info('INITIATING PREPROCESSING.................')
        out_shape = [dataIN.shape[0]] + myNet['crop_shape']
        pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
        for img_no in np.arange(dataIN.shape[0]):
            feed_dict = {
                preprocess_graph['imageIN']: dataIN[img_no, :]
            }
            pp_imgs[img_no, :] = sess.run(
                    preprocess_graph['imageOUT'],
                    feed_dict=feed_dict
            )
            
        # logging.info('Preprocessed data Shape: %s', str(pp_imgs.shape))
        return pp_imgs



class Train(PropertyClassification):
    
    def _init__(self, params, which_net, image_type):
        PropertyClassification.__init__(self, params, which_net, image_type)
    
    def train(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, self.preprocess_graph, sess)
        feed_dict = {
            self.train_graph['inpX']: preprocessed_data,
            self.train_graph['inpY']: batchY
        }
        
        if self.write_tensorboard_summary:
            out_prob, ls, _, l_rt, smry = sess.run([self.train_graph['outProbs'],
                                                    self.train_graph['loss'],
                                                    self.train_graph['optimizer'],
                                                    self.train_graph['l_rate'],
                                                    self.merged_summary],
                                                   feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)
        else:
            out_prob, ls, _, l_rt = sess.run([self.train_graph['outProbs'],
                                              self.train_graph['loss'],
                                              self.train_graph['optimizer'],
                                              self.train_graph['l_rate']], feed_dict=feed_dict)
        
        acc = self.accuracy(y=batchY, y_hat=out_prob)
        # print("Fold: " + str(self.foldNUM) +
        #       ", epoch: " + str(self.epoch) +
        #       ", batch: " + str(self.batch_num) +
        #       ", Loss= " + "{:.6f}".format(ls) +
        #       ", Training Accuracy= " + "{:.5f}".format(acc))
        
        logging.info("Fold: %s, epoch: %s, batch: %s, Loss: %s, Accuracy: %s",
                     str(self.foldNUM),
                     str(self.epoch),
                     str(self.batch_num),
                     str("{:.6f}".format(ls)),
                     str("{:.5f}".format(acc)))
        
        return acc
            
    def run_epoch(self):
        saver = tf.train.Saver()
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
                self.max_batch += 1   # When we have already run some batches for the epoch then we nees to run from
                # the next batch
            
            for epoch in range(self.max_epoch, self.max_epoch + self.epochs):
                self.epoch = epoch
                avg_accuracy = 0
                get_stats_at = 10
                for batch_num in range(self.max_batch, self.num_batches):
                    self.batch_num = batch_num
                    batchX, batchY, label_dict = load_batch_data(image_type=image_type,
                                                                 image_shape=image_shape,
                                                                 which_data='tr%s'%(batch_num))
                    batchY = self.to_one_hot(batchY)
                    acc = self.train(batchX, batchY, sess)

                    avg_accuracy += acc

                    if ((batch_num+1)%get_stats_at == 0) or (batch_num == self.num_batches -1):
                        print("Epoch: " + str(epoch) + ", Batch: " + str(batch_num) +
                              ", AVG Training Accuracy= " + "{:.5f}".format(avg_accuracy/get_stats_at))
                        avg_accuracy = 0
                        # SAVE CHECKPOINTS TO THE PATH FOR EVERY EPOCH
                        if self.save_checkpoint:
                            logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                            checkpoint_path = os.path.join(
                                    self.ckpt_path, '%s_epoch_%s_batch_%s.ckpt'%(self.which_net, str(epoch),
                                                                                 str(batch_num)))

                            saver.save(sess, checkpoint_path)
    
    def run(self, num_epochs, num_batches):
        logging.info('INITIATING RUN ........')
        self.foldNUM = 1
        self.epochs = num_epochs
        self.num_batches = num_batches

        self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['image_shape'])
        
        if self.which_net == 'vgg':
            self.train_graph = vgg(training=True)
        elif self.which_net == 'resnet':
            self.train_graph = resnet(training=True)
        else:
            raise ValueError('Provide a valid Net type options ={vgg, resnet}')
        ########   RUN THE SESSION
        self.run_epoch()


class Test(PropertyClassification):
    def _init__(self, params, which_net, image_type):
        PropertyClassification.__init__(self, params, image_type)
        
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, self.preprocess_graph, sess)
        feed_dict = {
            self.test_graph['inpX']: preprocessed_data
        }

        out_prob = sess.run(self.test_graph['outProbs'], feed_dict=feed_dict)

        v_acc = self.accuracy(y=batchY, y_hat=out_prob)

        print("Epoch: " + str(self.epoch) +
              ", Cross Validation Accuracy= " + "{:.5f}".format(v_acc))

        return v_acc

    def test(self,  checkpoint_path):
        saver = tf.train.Saver()
        self.max_checkpoint_num = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            batchX, batchY, label_dict = load_batch_data(image_type=image_type,
                                                         image_shape=image_shape,
                                                         which_data='cv')
           
            
            self.epoch = int(str(checkpoint_path.split('.')[0]).split('_')[-1])
            self.restore_checkpoint(checkpoint_path, saver, sess)
            batchY = self.to_one_hot(batchY)
            accuracy = self.cvalid(batchX, batchY, sess)

    def run(self):
        logging.info('INITIATING RUN ........')
        checkpoint_paths = self.get_checkpoint_path(which_checkpoint='all')
        for path in checkpoint_paths:
            self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['image_shape'])

            if self.which_net == 'vgg':
                print ('Test Graph: VGG')
                self.test_graph = vgg(training=False)
            elif self.which_net == 'resnet':
                print('Test Graphs: RESNET')
                self.test_graph = resnet(training=False)
            else:
                raise ValueError('Provide a valid Net type options ={vgg, resnet}')
            ########   RUN THE SESSION
            self.test(path)
            tf.reset_default_graph()
            
        
# image_type = 'aerial'
# image_shape = [224,224,3]
# image_type = 'assessor'
# image_shape = [260, 260, 3]
image_type = 'streetside'
image_shape = [260,260,3]

run = True
if run:
    Train(dict(use_checkpoint=True,
               save_checkpoint=True,
               write_tensorboard_summary=True
               ),
          which_net='resnet', # vgg
          image_type=image_type).run(num_epochs=3,
                                     num_batches=149)
    
    
    # Test(params={}, which_net='resnet', image_type=image_type).run()