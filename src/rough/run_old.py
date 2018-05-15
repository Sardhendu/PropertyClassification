import logging
import os

import numpy as np
import tensorflow as tf

from src.config import pathDict, fileNames, myNet, netParams, vars
from src.conv_net.ops import loss_optimization, summary_builder
from src.conv_net.vgg import conv_1, conv_2, conv_3, conv_4, fc1, fc2, fc3, softmax
from src.data_transformation.data_io import getPickleFile
from src.data_transformation.preprocessing import Preprocessing

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def graph():
    inpX = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['crop_shape'][0], myNet['crop_shape'][1], myNet['crop_shape'][2]],
                          name='X')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['num_labels']],
                          name='Y')
    
    X = conv_1(inpX)
    X = conv_2(X)
    X = conv_3(X)
    X = conv_4(X)
    X = tf.contrib.layers.flatten(X, scope='flatten')
    netParams['fc1']['shape'][0] = X.get_shape().as_list()[1]
    
    X = fc1(X)
    X = fc2(X)
    X = fc3(X)
    logging.info('The X till this point is: shape %s', str(X.get_shape().as_list()))
    
    logits, probs = softmax(X)
    logging.info('Logits: shape %s', str(logits.get_shape().as_list()))
    logging.info('Probabilities: shape %s', str(probs.get_shape().as_list()))
    
    lossCE, optimizer, l_rate = loss_optimization(X=logits, y=inpY, learning_rate_decay=True)
    
    return dict(inpX=inpX, inpY=inpY, outProbs=probs,
                loss=lossCE, optimizer=optimizer, l_rate=l_rate)


class Train():
    def __init__(self, params):
        
        params_keys = list(params.keys())
        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']
        
        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']
        
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
    
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
    
    def run_preprocessor(self, dataIN, sess):
        # logging.info('INITIATING PREPROCESSING.................')
        out_shape = [dataIN.shape[0]] + myNet['crop_shape']
        pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
        for img_no in np.arange(dataIN.shape[0]):
            feed_dict = {
                self.preprocess_graph['imageIN']: dataIN[img_no, :]
            }
            pp_imgs[img_no, :] = sess.run(
                    self.preprocess_graph['imageOUT'],
                    feed_dict=feed_dict
            )
        # logging.info('Preprocessed data Shape: %s', str(pp_imgs.shape))
        return pp_imgs
    
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, sess)
        feed_dict = {
            self.train_graph['inpX']: preprocessed_data
        }
        
        out_prob = sess.run(self.train_graph['outProbs'], feed_dict=feed_dict)
        
        v_acc = self.accuracy(y=batchY, y_hat=out_prob)
        
        print("Fold: " + str(self.foldNUM + 1) +
              ", Step: " + str(self.step + 1) +
              ", Cross Validation Validation Accuracy= " + "{:.5f}".format(v_acc))
        
        return v_acc
    
    def train(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, sess)
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
            self.writer.add_summary(smry, self.step)
        else:
            out_prob, ls, _, l_rt = sess.run([self.train_graph['outProbs'],
                                              self.train_graph['loss'],
                                              self.train_graph['optimizer'],
                                              self.train_graph['l_rate']], feed_dict=feed_dict)
        
        acc = self.accuracy(y=batchY, y_hat=out_prob)
        print("Fold: " + str(self.foldNUM + 1) +
              ", Step: " + str(self.step + 1) +
              ", Loss= " + "{:.6f}".format(ls) +
              ", Training Accuracy= " + "{:.5f}".format(acc))
        
        logging.info("Fold: %s, Step: %s, Loss: %s, Accuracy: %s",
                     str(self.foldNUM + 1),
                     str(self.step + 1),
                     str("{:.6f}".format(ls)),
                     str("{:.5f}".format(acc)))
        
        return acc
    
    def run_step(self):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # LOAD CHECKPOINTS (WEIGHTS IF NEEDED)
            # checkpoints = [ck for ck in os.listdir(pathDict['checkpoint_path']) if ck != '.DS_Store']
            checkpoints=[]
            if len(checkpoints) > 0 and self.use_checkpoint:
                logging.info('CHECKPOINT SAVER: Fetching model updated parameters')
                checkpoint_path = os.path.join(pathDict['checkpoint_path'],
                                               fileNames['checkpoint_file_name'] + '.ckpt'
                                               if len(fileNames['checkpoint_file_name'].split('.')) == 1
                                               else fileNames['checkpoint_file_name'])
                saver.restore(sess, checkpoint_path)
            
            # CREATE TENSOR BOARD SUMMARY WRITER (Writer opens up a file and starts writing summary for every step or
            #  so)
            if self.write_tensorboard_summary:
                logging.info('TENSOR BOARD SUMMARY: Dumping Tensorboard summary')
                self.merged_summary, self.writer = summary_builder(sess, pathDict["summary_path"])
            
            for step in np.arange(self.num_steps):
                self.step = step
                offset = (step * vars['batch_size']) % (self.train_size - self.remainder)
                
                if offset == (self.train_size - self.remainder - vars['batch_size']):
                    batchX = self.trnX[offset:(offset + vars['batch_size'] + self.remainder), :]
                    batchY = self.trnY[offset:(offset + vars['batch_size'] + self.remainder), :]
                else:
                    batchX = self.trnX[offset:(offset + vars['batch_size']), :]
                    batchY = self.trnY[offset:(offset + vars['batch_size']), :]
                
                acc = self.train(batchX, batchY, sess)
                
                # if ((step +1) % 10) == 0:
                # v_acc = self.cvalid(batchX=self.cvX, batchY=self.cvY, sess=sess)
                if step == 30:
                    break
            
            # SAVE CHECKPOINTS TO THE PATH
            if self.save_checkpoint:
                logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                checkpoint_path = os.path.join(pathDict['checkpoint_path'],
                                               fileNames['checkpoint_file_name'] + '.ckpt'
                                               if len(fileNames['checkpoint_file_name'].split('.')) == 1
                                               else fileNames['checkpoint_file_name'])
                saver.save(sess, checkpoint_path)
    
    def run(self, dataX, dataY):
        logging.info('INITIATING RUN ........')
        trnBatch_idx = [list(np.setdiff1d(np.arange(len(dataX)), np.array(i))) for i in np.arange(len(dataX))]
        
        cvBatch_idx = [i for i in np.arange(len(dataX))]
        print(trnBatch_idx)
        print(cvBatch_idx)
        
        print('dataX shape: ', dataX.shape)
        print('dataY shape: ', dataY.shape)
        
        logging.info('dataX.shape = %s, dataY.shape = %s', str(dataX.shape), str(dataY.shape))
        
        for nFold, (trn_batch_idx, cv_batch_idx) in enumerate(zip(trnBatch_idx, cvBatch_idx)):
            self.foldNUM = nFold
            self.trnX, self.trnY = self.reshape(dataX[trn_batch_idx, :], dataY[trn_batch_idx, :])
            
            # We random shuffle the training set to avoid the same label going into a minibatch.
            #######################################
            # shuffle_idx = np.arange(10)
            # np.random.shuffle(shuffle_idx)
            # self.trnX = self.trnX[shuffle_idx, :]
            # self.trnY = self.trnY[shuffle_idx, :]
            #######################################
            
            self.cvX, self.cvY = dataX[cv_batch_idx, :], dataY[cv_batch_idx, :]
            self.trnY = self.to_one_hot(self.trnY)
            self.cvY = self.to_one_hot(self.cvY)
            print('trnX shape: ', self.trnX.shape)
            print('trnY shape: ', self.trnY.shape)
            print('cvX shape: ', self.cvX.shape)
            print('cvY shape: ', self.cvY.shape)
            
            # print (trnY)
            self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['image_shape'])
            self.train_graph = graph()
            
            self.train_size = self.trnX.shape[0]
            self.num_steps = int(vars["epochs"] * self.train_size) // vars['batch_size']
            self.remainder = self.train_size % vars['batch_size']
            
            print('Total number of step: ', self.num_steps)
            print(self.remainder)
            
            ########   RUN THE SESSION
            self.run_step()
            
            break
            #


dataX, dataY, label_dict = getPickleFile('/Users/sam/All-Program/App-DataSet/HouseClassification/data_models/backup/batch_data',
                                         fileNames['batch_img_file'])
Train(dict(
        cv_fold_num=3,
        use_checkpoints=False,
        save_checkpoints=True,
        write_tensorboard_summary=False)).run(
        dataX=dataX,
        dataY=dataY
)